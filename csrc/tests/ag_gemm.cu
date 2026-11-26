// AllGather + GEMM with the Mode A consume hook (readyFlags / awaitChunk).
//   Y[M_total x N] = gather(X)[M_total x K] * W[K x N], bf16 in, fp32 accumulate, bf16 out.
// Part 1 (correctness): checker CTAs run alongside the AllGather, wait on each chunk's ready
//   flag, then compare that chunk in dst against the expected data. dst starts as 0xFF, so a
//   flag that fires early shows up as mismatches.
// Part 2 (timing): NCCL AG + cuBLAS, Purlin AG + cuBLAS, and Purlin AG overlapped with cuBLAS
//   GEMMs that each start once their rows' chunks have landed (split SMs: AG keeps its CTAs,
//   GEMMs use the rest). Each fused run's Y must equal the NCCL baseline's Y bit for bit.
//   Fused is chunk-major: chunks land on all peers in lockstep, so once a group of chunks has
//   landed everywhere, one batched GEMM multiplies those rows of every peer.
// Run: mpirun -n 4 ./testAGGEMM
#include <cstdio>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <mpi.h>
#include <nccl.h>
#include <purlin/core.cuh>
#include <purlin/benchmark/kernel.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>

constexpr auto nArch = purlin::normalizeArch<ARCH>();
using Atom = purlin::Atom<nArch, purlin::Configuration<128, 16, 8, 8, 2>>; // same as examples/ag
constexpr size_t CHUNK = 4 * 1024 * 1024;
constexpr int PUT_BLOCKS = 8;
constexpr int S = 2; // consumer CTAs per peer. 16 CTAs total: as fast as 64, leaves more SMs to the GEMMs
using Cfg = purlin::CollectiveConfig<purlin::CollectiveType::chunked, PUT_BLOCKS, purlin::UNUSED, CHUNK>;
constexpr int M = 4096, K = 4096; // rows per rank, columns of X: 32 MiB per rank
constexpr int ROWS_PER_CHUNK = CHUNK / (K * 2), CHUNKS = M / ROWS_PER_CHUNK;

#define CK(x) do { auto e_ = (x); if (e_ != 0) { std::printf("error %d at line %d\n", int(e_), __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

struct Args { const cuda::std::byte* src; cuda::std::byte* dst; size_t bytes; cuda::fast_mod_div<long int> blocks; };

__launch_bounds__(Atom::THREADS, 1)
__global__ void allGather(const __grid_constant__ Args a, const __grid_constant__ purlin::Context ctx) {
  extern __shared__ __align__(bench::sharedMemoryAlignment) cuda::std::byte workspace[];
  const purlin::SnacArgs<cuda::fast_mod_div<long int>> args{.dst = a.dst, .src = a.src, .bytes = a.bytes,
    .workspace = workspace, .blocks = a.blocks, .collBlocks = static_cast<int>(a.blocks)};
  purlin::allGather<Atom, Cfg>(args, ctx);
}

// Exact small values (multiples of 1/8 in [-1, 1]) so every fp32 sum is exact and Y is the
// same bits whatever order cuBLAS adds in.
__device__ __host__ inline float val(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t h = a * 2654435761u ^ b * 2246822519u ^ c * 3266489917u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return static_cast<float>(static_cast<int>(h % 17) - 8) / 8.0f;
}
__global__ void fill(__nv_bfloat16* x, int rows, int cols, uint32_t tag) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size_t(rows) * cols; i += size_t(gridDim.x) * blockDim.x)
    x[i] = __float2bfloat16(val(tag, i / cols, i % cols));
}
// One CTA per (peer, chunk): wait for the chunk, then check it.
__global__ void checkChunks(const __grid_constant__ purlin::Context ctx, const __nv_bfloat16* dst, unsigned long long* bad) {
  const int peer = blockIdx.x / CHUNKS, chunk = blockIdx.x % CHUNKS;
  purlin::awaitChunk(ctx, peer, chunk, S);
  const size_t row0 = size_t(chunk) * ROWS_PER_CHUNK;
  const auto* p = dst + (size_t(peer) * M + row0) * K;
  unsigned long long n = 0;
  for (size_t i = threadIdx.x; i < size_t(ROWS_PER_CHUNK) * K; i += blockDim.x)
    n += __bfloat162float(p[i]) != val(peer, row0 + i / K, i % K);
  atomicAdd(bad, n);
}
// Holds its stream until chunks [0, chunk] of every peer have landed; the next GEMM waits for it.
__global__ void waitChunkAllPeers(const __grid_constant__ purlin::Context ctx, int world, int chunk) {
  for (int p = 0; p < world; ++p) purlin::awaitChunk(ctx, p, chunk, S);
}
__global__ void countDiff(const __nv_bfloat16* a, const __nv_bfloat16* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    c += reinterpret_cast<const uint16_t*>(a)[i] != reinterpret_cast<const uint16_t*>(b)[i];
  atomicAdd(bad, c);
}

void run() {
  bench::PurlinRuntime rt;
  const int world = rt.world, rank = rt.rank;
  const auto s1 = rt.stream;
  cudaStream_t s2; CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  auto ctx = rt.context;
  CK(cudaMalloc(&ctx.readyFlags, sizeof(uint64_t) * world * purlin::READY_SLOTS));
  const size_t flagBytes = sizeof(uint64_t) * world * purlin::READY_SLOTS;

  const size_t localBytes = size_t(M) * K * 2, rowsAll = size_t(M) * world;
  __nv_bfloat16 *x, *xAll, *w, *y, *yRef; unsigned long long* bad;
  CK(cudaMalloc(&x, localBytes)); CK(cudaMalloc(&xAll, localBytes * world));
  CK(cudaMalloc(&w, size_t(K) * 8192 * 2)); CK(cudaMalloc(&y, rowsAll * 8192 * 2)); CK(cudaMalloc(&yRef, rowsAll * 8192 * 2));
  CK(cudaMalloc(&bad, sizeof(*bad)));
  fill<<<1024, 256, 0, s1>>>(x, M, K, rank);
  fill<<<1024, 256, 0, s1>>>(w, K, 8192, 1000);

  constexpr auto smem = purlin::copySmemBytes<Atom>();
  bench::configureKernel(allGather, smem, rt.deviceProperties);
  const int blocks = PUT_BLOCKS + S * world;
  const Args args{.src = reinterpret_cast<cuda::std::byte*>(x), .dst = reinterpret_cast<cuda::std::byte*>(xAll),
    .bytes = localBytes, .blocks = cuda::fast_mod_div<long int>{blocks}};
  const auto purlinAG = [&] {
    CK(cudaMemsetAsync(ctx.readyFlags, 0, flagBytes, s1));
    allGather<<<blocks, Atom::THREADS, smem, s1>>>(args, ctx);
  };

  // ---- Part 1: correctness of the ready flags, 20 calls in a row ----
  cudaEvent_t go, done; CK(cudaEventCreateWithFlags(&go, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
  CK(cudaMemsetAsync(bad, 0, sizeof(*bad), s1));
  for (int it = 0; it < 20; ++it) {
    CK(cudaMemsetAsync(xAll, 0xFF, localBytes * world, s1));
    CK(cudaMemsetAsync(ctx.readyFlags, 0, flagBytes, s1));
    CK(cudaEventRecord(go, s1)); CK(cudaStreamWaitEvent(s2, go));
    allGather<<<blocks, Atom::THREADS, smem, s1>>>(args, ctx);
    checkChunks<<<world * CHUNKS, 256, 0, s2>>>(ctx, xAll, bad);
    CK(cudaEventRecord(done, s2)); CK(cudaStreamWaitEvent(s1, done));
  }
  unsigned long long nBad = 0;
  CK(cudaMemcpyAsync(&nBad, bad, sizeof(nBad), cudaMemcpyDeviceToHost, s1)); CK(cudaStreamSynchronize(s1));
  CK(cudaGetLastError());
  unsigned long long allBad = 0; MPI_Reduce(&nBad, &allBad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (!rank) std::printf("correctness: %d ranks x 20 calls x %d chunks checked right after their flag, mismatched elements = %llu -> %s\n",
                         world, world * CHUNKS, allBad, allBad ? "FAIL" : "PASS");

  // ---- Part 2: timing ----
  ncclUniqueId id; if (!rank) CK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm; CK(ncclCommInitRank(&comm, world, id, rank));
  cublasHandle_t h1, h2; CK(cublasCreate(&h1)); CK(cublasCreate(&h2));
  CK(cublasSetStream(h1, s1)); CK(cublasSetStream(h2, s2));
  // Keep split-K partial sums in fp32 so Y is the same bits for any row count.
  for (auto hd : {h1, h2}) CK(cublasSetMathMode(hd, cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  const float one = 1, zero = 0;
  // Row-major Y[rows x n] = X[rows x K] * W[K x n] is column-major Y^T = W^T X^T.
  const auto gemm = [&](cublasHandle_t hd, size_t row0, int rows, int n, __nv_bfloat16* out) {
    CK(cublasGemmEx(hd, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, K, &one, w, CUDA_R_16BF, 8192,
                    xAll + row0 * K, CUDA_R_16BF, K, &zero, out + row0 * n, CUDA_R_16BF, n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  const auto timeMs = [&](auto&& op) {  // ms per call, slowest rank
    for (int i = 0; i < 5; ++i) op();
    CK(cudaStreamSynchronize(s1)); MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, s1)); for (int i = 0; i < 20; ++i) op(); CK(cudaEventRecord(b, s1));
    CK(cudaEventSynchronize(b)); float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); ms /= 20;
    float worst = 0; MPI_Allreduce(&ms, &worst, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    return worst;
  };
  const auto sameAsRef = [&](int n) {
    CK(cudaMemsetAsync(bad, 0, sizeof(*bad), s1));
    countDiff<<<1024, 256, 0, s1>>>(y, yRef, rowsAll * n, bad);
    unsigned long long c = 0; CK(cudaMemcpyAsync(&c, bad, sizeof(c), cudaMemcpyDeviceToHost, s1)); CK(cudaStreamSynchronize(s1));
    unsigned long long t = 0; MPI_Allreduce(&c, &t, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return t;  // elements of Y that differ from the NCCL baseline, all ranks
  };
  const auto ncclAG = [&] { CK(ncclAllGather(x, xAll, size_t(M) * K, ncclBfloat16, comm, s1)); };

  // Chunk groups per GEMM step, e.g. {1, 1, 2, 4}: small first so the GEMMs start early.
  const std::vector<std::pair<const char*, std::vector<int>>> schedules{
    {"2-2-2-2", {2, 2, 2, 2}}, {"4-4", {4, 4}}, {"1-1-2-4", {1, 1, 2, 4}}};
  if (!rank) std::printf("N,schedule,ncclAG_ms,purlinAG_ms,gemm_ms,nccl+gemm_ms,purlin+gemm_ms,fused_ms,"
                         "fused_vs_nccl,Y_elements_differing\n");
  for (int n : {1024, 2048, 4096, 8192}) {
    const float tNccl = timeMs(ncclAG), tPurlin = timeMs(purlinAG);
    const float tGemm = timeMs([&] { gemm(h1, 0, rowsAll, n, yRef); });
    const float tNcclGemm = timeMs([&] { ncclAG(); gemm(h1, 0, rowsAll, n, yRef); });  // also leaves yRef
    const float tPurlinGemm = timeMs([&] { purlinAG(); gemm(h1, 0, rowsAll, n, y); });
    for (const auto& [name, schedule] : schedules) {
      const auto& groups = schedule;  // a plain reference, so the lambda below can capture it
      // One batched GEMM per group: batch = peers, stride = one peer's rows.
      const auto fused = [&] {
        CK(cudaMemsetAsync(ctx.readyFlags, 0, flagBytes, s1));
        CK(cudaEventRecord(go, s1)); CK(cudaStreamWaitEvent(s2, go));
        allGather<<<blocks, Atom::THREADS, smem, s1>>>(args, ctx);
        int c = 0;
        for (const int g : groups) {
          const size_t row0 = size_t(c) * ROWS_PER_CHUNK;
          c += g;
          waitChunkAllPeers<<<1, 32, 0, s2>>>(ctx, world, c - 1);
          CK(cublasGemmStridedBatchedEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, n, g * ROWS_PER_CHUNK, K, &one,
                                        w, CUDA_R_16BF, 8192, 0,
                                        xAll + row0 * K, CUDA_R_16BF, K, (long long)M * K, &zero,
                                        y + row0 * n, CUDA_R_16BF, n, (long long)M * n, world,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        CK(cudaEventRecord(done, s2)); CK(cudaStreamWaitEvent(s1, done));
      };
      CK(cudaMemsetAsync(y, 0, rowsAll * n * 2, s1));
      const float tFused = timeMs(fused);
      const auto diff = sameAsRef(n);
      if (!rank) std::printf("%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu\n", n, name, tNccl, tPurlin, tGemm,
                             tNcclGemm, tPurlinGemm, tFused, tNcclGemm / tFused, diff);
    }
  }
  CK(cudaGetLastError());
  ncclCommDestroy(comm);
}

int main(int argc, char** argv) {
  int up = 0; MPI_Initialized(&up); if (!up) MPI_Init(&argc, &argv);
  run();  // Purlin/NVSHMEM shut down when run() returns
  int down = 0; MPI_Finalized(&down); if (!down) MPI_Finalize();
  return 0;
}
