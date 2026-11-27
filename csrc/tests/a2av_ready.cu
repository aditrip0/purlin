// AllToAllV ready flags: correctness only.
// Checker CTAs run on a second stream next to purlin::all2allV. Each waits on the ready flag
// for one (source GPU, 1 MiB chunk) with awaitChunkV, then compares that chunk of dst with the
// expected bytes. dst starts as 0xFF, so a flag that fires early shows up as mismatches.
// Splits: rank r sends `hot` of its bytes to rank r+1 and splits the rest evenly (hot = 1/world
// is uniform), which covers the packet, staged and skew-weighted paths.
// Run: mpirun -n 4 ./testA2AVREADY
#include <algorithm>
#include <cstdio>
#include <vector>

#include <mpi.h>

#include <purlin/benchmark/data.cuh>
#include <purlin/benchmark/device_buffer.cuh>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/benchmark/variable_counts.cuh>
#include <purlin/host/all2all.cuh>

constexpr int WORLD = 4;
constexpr size_t CHUNK = purlin::host::All2AllVCodesign<purlin::normalizeArch<ARCH>(), WORLD>::CHUNK_SIZE;
struct Recv { size_t bytes[8], offset[8]; };

__global__ void checkChunks(const __grid_constant__ purlin::Context ctx, const __grid_constant__ Recv r,
                            const cuda::std::byte* dst, const cuda::std::byte* ref, unsigned long long* bad) {
  const int peer = blockIdx.x, chunk = blockIdx.y;
  if (static_cast<size_t>(chunk) * CHUNK >= r.bytes[peer]) return;
  purlin::awaitChunkV(ctx, peer, chunk);
  const size_t start = r.offset[peer] + chunk * CHUNK;
  const size_t end = r.offset[peer] + cuda::std::min(r.bytes[peer], (chunk + 1) * CHUNK);
  unsigned long long n = 0;
  for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) n += dst[i] != ref[i];
  if (n) atomicAdd(bad, n);
}

// Bytes rank `from` sends to rank `to`, out of `total`, rounded down to 256 B.
size_t split(size_t total, double hot, int from, int to, int world) {
  const double share = to == (from + 1) % world ? hot : (1 - hot) / (world - 1);
  return static_cast<size_t>(total * share) / 256 * 256;
}

int main() {
  bench::PurlinRuntime rt;
  const int world = rt.world, rank = rt.rank;
  if (world != WORLD) { if (!rank) std::printf("run with mpirun -n %d\n", WORLD); return 1; }
  cudaStream_t s2;
  CHECK_CUDA(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  const size_t flagBytes = sizeof(uint64_t) * world * purlin::READY_SLOTS;
  CHECK_CUDA(cudaMalloc(&rt.context.readyFlags, flagBytes));
  const size_t maxTotal = 64UL << 20;
  bench::DeviceBuffer<cuda::std::byte> src(maxTotal, rt.stream), dst(maxTotal, rt.stream), ref(maxTotal, rt.stream);
  bench::DeviceBuffer<size_t> dSends(world, rt.stream), dRecvs(world, rt.stream);
  bench::DeviceBuffer<unsigned long long> bad(1, rt.stream);
  int failed = 0;
  if (!rank) std::printf("total_per_rank,hot,max_stream,mismatches\n");
  for (const size_t total : {256UL << 10, 4UL << 20, 64UL << 20}) {
    for (const double hot : {1.0 / world, 0.7}) {
      std::vector<size_t> sends(world), recvs(world);
      for (int p = 0; p < world; ++p) { sends[p] = split(total, hot, rank, p, world); recvs[p] = split(total, hot, p, rank, world); }
      const auto sOff = bench::offsets(sends), rOff = bench::offsets(recvs);
      Recv r{};
      size_t maxStream = 0, maxChunks = 1;
      for (int p = 0; p < world; ++p) {
        r.bytes[p] = recvs[p]; r.offset[p] = rOff[p];
        maxStream = std::max(maxStream, std::max(sends[p], recvs[p]));
        maxChunks = std::max(maxChunks, (recvs[p] + CHUNK - 1) / CHUNK);
      }
      rt.context.vState = bench::makePurlinAllToAllVState(sends, recvs, sOff, rank);
      CHECK_CUDA(cudaMemcpyAsync(dSends.get(), sends.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, rt.stream));
      CHECK_CUDA(cudaMemcpyAsync(dRecvs.get(), recvs.data(), sizeof(size_t) * world, cudaMemcpyHostToDevice, rt.stream));
      for (int p = 0; p < world; ++p) {
        bench::fillRandomBytes(src.get() + sOff[p], sends[p], bench::pairSeed(42, rank, p), rt.stream);
        bench::fillRandomBytes(ref.get() + rOff[p], recvs[p], bench::pairSeed(42, p, rank), rt.stream);
      }
      unsigned long long mismatches = 0;
      for (int call = 0; call < 10; ++call) {
        CHECK_CUDA(cudaMemsetAsync(dst.get(), 0xFF, maxTotal, rt.stream));
        CHECK_CUDA(cudaMemsetAsync(rt.context.readyFlags, 0, flagBytes, rt.stream));
        CHECK_CUDA(cudaMemsetAsync(bad.get(), 0, sizeof(unsigned long long), rt.stream));
        CHECK_CUDA(cudaStreamSynchronize(rt.stream));
        MPI_Barrier(MPI_COMM_WORLD);
        // all2allV first: it needs all its CTAs resident at once, so checkers can't take SMs before it.
        purlin::all2allV<ARCH>(src.get(), dst.get(), dSends.get(), dRecvs.get(), rt.context, rt.stream);
        checkChunks<<<dim3(world, maxChunks), 128, 0, s2>>>(rt.context, r, dst.get(), ref.get(), bad.get());
        CHECK_CUDA(cudaStreamSynchronize(s2));
        CHECK_CUDA(cudaStreamSynchronize(rt.stream));
        unsigned long long h = 0;
        CHECK_CUDA(cudaMemcpy(&h, bad.get(), sizeof(h), cudaMemcpyDeviceToHost));
        mismatches += h;
      }
      unsigned long long all = 0;
      MPI_Allreduce(&mismatches, &all, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      failed |= all != 0;
      if (!rank) std::printf("%zu,%.2f,%zu,%llu\n", total, hot, maxStream, all);
    }
  }
  if (!rank) std::printf(failed ? "FAIL\n" : "PASS\n");
  CHECK_CUDA(cudaFree(rt.context.readyFlags));
  rt.context.readyFlags = nullptr;
  return failed;
}
