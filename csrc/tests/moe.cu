// MoE layer benchmark on 4 GPUs: pack -> dispatch AllToAllV -> expert GEMMs -> reduce -> combine AllToAllV -> unpack,
// with NCCL or Purlin for both AllToAllVs.
// Configs (MOE_CONFIG):
//   toy   1 expert per GPU, top-1, T = 8192 tokens per GPU, H = 4096, one GEMM H -> 4096, tokens split evenly.
//   qwen  one Qwen3-30B-A3B MoE layer on EP 4: T = 2048 tokens per GPU, H = 2048, 128 experts (32 per GPU), top-8,
//         gate+up GEMM (H -> 2 x 768), relu(gate) * up, down GEMM (768 -> H), grouped over the 32 local experts.
//         Routing: uniform random top-8.
// Simplifications: routing weights are 1, relu instead of silu, and all values are small integers, so every fp32
//   sum is exact and both paths give the same bits. Routing is built from per-expert counts: expert e gets a run of
//   count_e consecutive tokens in a cyclic token order that interleaves the GPUs, so each GPU holds 1/4 of each
//   expert's tokens.
// Steps: pack (each token once per destination GPU) -> dispatch -> expert (gather received rows by local expert,
//   grouped GEMMs) -> reduce (sum a token's local experts) -> combine -> unpack (sum a token's results from all GPUs).
// Times are ms per call on the slowest GPU; dg = dispatch + expert. The Purlin layer must equal the NCCL layer bit for bit.
// Run: MOE_CONFIG=qwen mpirun -n 4 ./testMOE
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <mpi.h>
#include <nccl.h>
#include <purlin/benchmark/purlin_runtime.cuh>
#include <purlin/host/all2all.cuh>

using bf16 = __nv_bfloat16;
#define CK(x) do { auto e_ = (x); if (e_ != 0) { std::printf("error %d at line %d\n", int(e_), __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// T tokens per GPU, hidden H, E experts, top-k, first GEMM width N1, inter = down GEMM input (0: no down GEMM, N1 = H).
struct Cfg { int T, H, E, topk, N1, inter; };

__global__ void fill(bf16* x, size_t n, int cols, uint32_t tag) {  // values in {-1, 0, 1}
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = tag * 2654435761u ^ uint32_t(i / cols) * 2246822519u ^ uint32_t(i % cols) * 3266489917u;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    x[i] = __float2bfloat16(float(int(h % 3) - 1));
  }
}
// Rows r0 <= r < r1: out[r] = sum of in[idx[k]] for k in [ptr[r], ptr[r + 1]), in fp32, rounded to bf16 once.
__global__ void gatherSum(bf16* out, const bf16* in, const int* ptr, const int* idx, int r0, int r1, int cols) {
  for (int r = r0 + blockIdx.x; r < r1; r += gridDim.x)
    for (int c = threadIdx.x * 8; c < cols; c += blockDim.x * 8) {
      float s[8] = {};
      for (int k = ptr[r]; k < ptr[r + 1]; ++k) {
        const uint4 v = *reinterpret_cast<const uint4*>(in + size_t(idx[k]) * cols + c);
        for (int q = 0; q < 8; ++q) s[q] += __bfloat162float(reinterpret_cast<const bf16*>(&v)[q]);
      }
      uint4 o; for (int q = 0; q < 8; ++q) reinterpret_cast<bf16*>(&o)[q] = __float2bfloat16(s[q]);
      *reinterpret_cast<uint4*>(out + size_t(r) * cols + c) = o;
    }
}
// Rows [r0, r1): a = relu(gate) * up, where h = [gate | up].
__global__ void reluMul(bf16* a, const bf16* h, size_t r0, size_t r1, int inter) {
  for (size_t i = r0 * inter + blockIdx.x * blockDim.x + threadIdx.x; i < r1 * inter; i += size_t(gridDim.x) * blockDim.x) {
    const size_t r = i / inter, c = i % inter;
    a[i] = __float2bfloat16(fmaxf(__bfloat162float(h[r * 2 * inter + c]), 0.f) * __bfloat162float(h[r * 2 * inter + inter + c]));
  }
}
__global__ void countDiff(const bf16* a, const bf16* b, size_t n, unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    c += reinterpret_cast<const uint16_t*>(a)[i] != reinterpret_cast<const uint16_t*>(b)[i];
  atomicAdd(bad, c);
}

std::vector<void*> allocs;  // freed after each routing case
template <class T> T* dev(const std::vector<T>& v) {
  T* p; CK(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); allocs.push_back(p); return p;
}
struct Csr { int *ptr, *idx; };
Csr csr(const std::vector<std::vector<int>>& rows) {
  std::vector<int> p{0}, i;
  for (auto& r : rows) { i.insert(i.end(), r.begin(), r.end()); p.push_back(int(i.size())); }
  return {dev(p), dev(i)};
}
// One cuBLAS grouped GEMM, one group per expert. Row-major C[n x m] = B[n x k] * A[k x m] is column-major C^T = A^T B^T.
struct Grouped {
  std::vector<int> m, n, k, lda, ldb, ldc; std::vector<const void*> a, b; std::vector<void*> c; void** d = nullptr;
  void add(const bf16* A, const bf16* B, bf16* C, int m_, int n_, int k_) {
    m.push_back(m_); n.push_back(n_); k.push_back(k_); lda.push_back(m_); ldb.push_back(k_); ldc.push_back(m_);
    a.push_back(A); b.push_back(B); c.push_back(C);
  }
  void upload() { std::vector<void*> p; for (auto* v : a) p.push_back(const_cast<void*>(v)); for (auto* v : b) p.push_back(const_cast<void*>(v)); p.insert(p.end(), c.begin(), c.end()); d = dev(p); }
  void run(cublasHandle_t hd) const {
    const int g = int(m.size()); if (!g) return;
    if (g == 1) {  // one expert (toy): plain cublasGemmEx, ~25% faster here than a one-group grouped GEMM
      const float one = 1, zero = 0;
      CK(cublasGemmEx(hd, CUBLAS_OP_N, CUBLAS_OP_N, m[0], n[0], k[0], &one, a[0], CUDA_R_16BF, lda[0], b[0], CUDA_R_16BF, ldb[0],
                      &zero, c[0], CUDA_R_16BF, ldc[0], CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      return;
    }
    const std::vector<cublasOperation_t> op(g, CUBLAS_OP_N); const std::vector<float> one(g, 1), zero(g, 0); const std::vector<int> size(g, 1);
    CK(cublasGemmGroupedBatchedEx(hd, op.data(), op.data(), m.data(), n.data(), k.data(), one.data(), (const void* const*)d, CUDA_R_16BF, lda.data(),
                                  (const void* const*)(d + g), CUDA_R_16BF, ldb.data(), zero.data(), d + 2 * g, CUDA_R_16BF, ldc.data(),
                                  g, size.data(), CUBLAS_COMPUTE_32F));
  }
};
// Expert of slot j of global token t (on GPU t % world, local index t / world) at ex[t * topk + j]: expert e takes
// count[e] consecutive positions of a cyclic walk over the tokens, so a token never gets the same expert twice.
std::vector<int> route(const std::vector<int>& count, int tokens, int topk, bool shuffle) {
  std::vector<int> order(count.size()), ex(size_t(tokens) * topk);
  std::iota(order.begin(), order.end(), 0);
  if (shuffle) std::shuffle(order.begin(), order.end(), std::mt19937(7));
  long pos = 0;
  for (int e : order) {
    if (count[e] > tokens) { std::printf("expert %d has more than %d tokens\n", e, tokens); MPI_Abort(MPI_COMM_WORLD, 1); }
    for (int c = 0; c < count[e]; ++c, ++pos) ex[(pos % tokens) * topk + pos / tokens] = e;
  }
  if (pos != long(tokens) * topk) { std::printf("counts sum to %ld, need %d\n", pos, tokens * topk); MPI_Abort(MPI_COMM_WORLD, 1); }
  return ex;
}

void run() {
  bench::PurlinRuntime rt;
  const int world = rt.world, me = rt.rank;
  if (world != 4) { if (!me) std::printf("run with mpirun -n 4\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
  const std::string name = std::getenv("MOE_CONFIG") ? std::getenv("MOE_CONFIG") : "toy";
  const Cfg cfg = name == "qwen" ? Cfg{2048, 2048, 128, 8, 2 * 768, 768} : Cfg{8192, 4096, 4, 1, 4096, 0};
  const int T = cfg.T, H = cfg.H, El = cfg.E / world, tokens = world * T;

  // Routing cases: name and per-expert token counts (summing to tokens * topk).
  std::vector<std::pair<std::string, std::vector<int>>> cases;
  if (name == "qwen") {
    std::vector<int> c(cfg.E); std::mt19937 rng(42);
    for (int t = 0; t < tokens; ++t) {  // uniform random top-k
      std::vector<int> pick;
      while (int(pick.size()) < cfg.topk) { const int e = int(rng() % cfg.E); if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e); }
      for (int e : pick) ++c[e];
    }
    cases.push_back({"uniform", c});
  } else {
    for (const double hot : {0.25}) {  // per source GPU: int(T * hot) tokens to GPU 0, the rest split evenly
      std::vector<int> cnt(world); cnt[0] = int(T * hot);
      for (int d = 1; d < world; ++d) cnt[d] = (T - cnt[0]) / (world - 1);
      cnt[world - 1] += T - std::accumulate(cnt.begin(), cnt.end(), 0);
      for (auto& c : cnt) c *= world;
      char buf[32]; std::snprintf(buf, sizeof(buf), "hot%.2f", hot); cases.push_back({buf, cnt});
    }
  }

  const auto s1 = rt.stream;
  auto& ctx = rt.context;
  ncclUniqueId id; if (!me) CK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm; CK(ncclCommInitRank(&comm, world, id, me));
  cublasHandle_t h1; CK(cublasCreate(&h1)); CK(cublasSetStream(h1, s1));
  CK(cublasSetMathMode(h1, cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));

  // Buffers (rows x width): x, out [T x H]; xs, xr, y, cr [tokens x H]; xe [tokens*topk x H], h [.. x N1], a [.. x inter], eo [.. x H].
  const size_t maxE = size_t(tokens) * cfg.topk;
  bf16 *x, *out, *outRef, *xs, *xr, *y, *cr, *xe, *h, *a = nullptr, *eo, *w1, *w2 = nullptr; unsigned long long* bad;
  for (auto [p, n] : std::initializer_list<std::pair<bf16**, size_t>>{{&x, size_t(T) * H}, {&out, size_t(T) * H}, {&outRef, size_t(T) * H}, {&xs, size_t(tokens) * H},
                      {&xr, size_t(tokens) * H}, {&y, size_t(tokens) * H}, {&cr, size_t(tokens) * H}, {&xe, maxE * H}, {&h, maxE * cfg.N1},
                      {&w1, size_t(El) * H * cfg.N1}}) CK(cudaMalloc(p, n * 2));
  if (cfg.inter) { CK(cudaMalloc(&a, maxE * cfg.inter * 2)); CK(cudaMalloc(&eo, maxE * H * 2)); CK(cudaMalloc(&w2, size_t(El) * cfg.inter * H * 2)); }
  else eo = h;  // toy: the one GEMM's output is the expert output (N1 = H)
  CK(cudaMalloc(&bad, sizeof(*bad)));
  size_t* dSplits; CK(cudaMalloc(&dSplits, 4 * world * sizeof(size_t)));  // dispatch send/recv, combine send/recv
  fill<<<1024, 256, 0, s1>>>(x, size_t(T) * H, H, me);
  fill<<<1024, 256, 0, s1>>>(w1, size_t(El) * H * cfg.N1, cfg.N1, 1000 + me);
  if (cfg.inter) fill<<<1024, 256, 0, s1>>>(w2, size_t(El) * cfg.inter * H, H, 2000 + me);

  if (!me) std::printf("config %s: T=%d H=%d experts=%d top-%d N1=%d inter=%d\n", name.c_str(), T, H, cfg.E, cfg.topk, cfg.N1, cfg.inter);
  if (!me) std::printf("case,variant,median_ms,min_ms,max_ms,vs_nccl,out_diffs\n");
  for (const auto& [cname, count] : cases) {
    const auto ex = route(count, tokens, cfg.topk, name == "qwen");
    const auto uses = [&](int t, int gpu) { for (int j = 0; j < cfg.topk; ++j) if (ex[size_t(t) * cfg.topk + j] / El == gpu) return true; return false; };
    // N[s][d]: tokens GPU s sends GPU d; tok[s][d]: their local indices, in order.
    std::vector<std::vector<std::vector<int>>> tok(world, std::vector<std::vector<int>>(world));
    for (int t = 0; t < tokens; ++t) for (int d = 0; d < world; ++d) if (uses(t, d)) tok[t % world][d].push_back(t / world);
    if (!me) {
      std::printf("routing %s: GPU recv rows / expert rows:", cname.c_str());
      for (int d = 0; d < world; ++d) {
        size_t r = 0, e = 0; for (int s = 0; s < world; ++s) r += tok[s][d].size();
        for (size_t i = 0; i < ex.size(); ++i) e += ex[i] / El == d;
        std::printf(" %zu/%zu", r, e);
      }
      std::printf("\n");
    }
    std::vector<size_t> dSend(world), dRecv(world), dSendOff(world), dRecvOff(world), cSend(world), cRecv(world), cSendOff(world), cRecvOff(world);
    std::vector<int> rOff(world + 1), sOff(world + 1);  // row offsets of each source in xr, of each destination in xs
    for (int p = 0; p < world; ++p) {
      dSend[p] = tok[me][p].size() * H * 2; dRecv[p] = tok[p][me].size() * H * 2; cSend[p] = dRecv[p]; cRecv[p] = dSend[p];
      rOff[p + 1] = rOff[p] + int(tok[p][me].size()); sOff[p + 1] = sOff[p] + int(tok[me][p].size());
    }
    for (int p = 1; p < world; ++p) {
      dSendOff[p] = dSendOff[p - 1] + dSend[p - 1]; dRecvOff[p] = dRecvOff[p - 1] + dRecv[p - 1];
      cSendOff[p] = cSendOff[p - 1] + cSend[p - 1]; cRecvOff[p] = cRecvOff[p - 1] + cRecv[p - 1];
    }
    std::vector<size_t> all; for (auto* v : {&dSend, &dRecv, &cSend, &cRecv}) all.insert(all.end(), v->begin(), v->end());
    CK(cudaMemcpyAsync(dSplits, all.data(), all.size() * sizeof(size_t), cudaMemcpyHostToDevice, s1));
    const auto dState = bench::makePurlinAllToAllVState(dSend, dRecv, dSendOff, me);
    const auto cState = bench::makePurlinAllToAllVState(cSend, cRecv, cSendOff, me);
    // pack: xs row <- x row; unpack: out row t <- sum of its cr rows (one per destination GPU).
    std::vector<std::vector<int>> packL, unpackL(T);
    for (int d = 0; d < world; ++d) for (size_t i = 0; i < tok[me][d].size(); ++i) { packL.push_back({tok[me][d][i]}); unpackL[tok[me][d][i]].push_back(sOff[d] + int(i)); }
    const Csr packC = csr(packL), unpackC = csr(unpackL);

    // Expert rows: xe rows ordered [local expert][source][row]; reduce sums a received row's xe rows.
    Grouped g1, g2; std::vector<std::vector<int>> gl, red(rOff[world]);
    for (int e = 0; e < El; ++e) {
      const int off = int(gl.size());
      for (int s = 0; s < world; ++s)
        for (int i = 0; i < int(tok[s][me].size()); ++i) {
          const int t = tok[s][me][i] * world + s;
          for (int j = 0; j < cfg.topk; ++j) if (ex[size_t(t) * cfg.topk + j] == me * El + e) { red[rOff[s] + i].push_back(int(gl.size())); gl.push_back({rOff[s] + i}); }
        }
      const int rows = int(gl.size()) - off;
      if (!rows) continue;
      g1.add(w1 + size_t(e) * H * cfg.N1, xe + size_t(off) * H, h + size_t(off) * cfg.N1, cfg.N1, rows, H);
      if (cfg.inter) g2.add(w2 + size_t(e) * cfg.inter * H, a + size_t(off) * cfg.inter, eo + size_t(off) * H, H, rows, cfg.inter);
    }
    const int xeRows = int(gl.size()); g1.upload(); g2.upload();
    const Csr gatherC = csr(gl), reduceC = csr(red);

    const auto B = [](auto* p) { return reinterpret_cast<cuda::std::byte*>(p); };
    const auto gather = [&](bf16* o, const bf16* in, Csr c, int r0, int r1, int cols, cudaStream_t st) {
      if (r1 > r0) gatherSum<<<std::min(r1 - r0, 4096), 128, 0, st>>>(o, in, c.ptr, c.idx, r0, r1, cols);
    };
    const auto pack = [&] { gather(xs, x, packC, 0, sOff[world], H, s1); };
    const auto unpack = [&] { gather(out, cr, unpackC, 0, T, H, s1); };
    const auto dispatchPurlin = [&] { ctx.vState = dState; purlin::all2allV<ARCH>(B(xs), B(xr), dSplits, dSplits + world, ctx, s1); };
    const auto combinePurlin = [&] { ctx.vState = cState; purlin::all2allV<ARCH>(B(y), B(cr), dSplits + 2 * world, dSplits + 3 * world, ctx, s1); };
    const auto nccl = [&](auto* src, auto* dst, const std::vector<size_t>& s, const std::vector<size_t>& so,
                          const std::vector<size_t>& r, const std::vector<size_t>& ro) {
      CK(ncclGroupStart());
      for (int p = 0; p < world; ++p) {
        CK(ncclSend(B(src) + so[p], s[p], ncclUint8, p, comm, s1));
        CK(ncclRecv(B(dst) + ro[p], r[p], ncclUint8, p, comm, s1));
      }
      CK(ncclGroupEnd());
    };
    const auto dispatchNccl = [&] { nccl(xs, xr, dSend, dSendOff, dRecv, dRecvOff); };
    const auto combineNccl = [&] { nccl(y, cr, cSend, cSendOff, cRecv, cRecvOff); };
    const auto expert = [&] {
      gather(xe, xr, gatherC, 0, xeRows, H, s1);
      g1.run(h1);
      if (cfg.inter && xeRows) { reluMul<<<1024, 256, 0, s1>>>(a, h, 0, xeRows, cfg.inter); g2.run(h1); }
    };
    const auto reduce = [&] { gather(y, eo, reduceC, 0, rOff[world], H, s1); };
    const auto layer = [&](std::function<void()> dg, std::function<void()> comb) -> std::function<void()> {
      return [&, dg, comb] { pack(); dg(); reduce(); comb(); unpack(); };
    };

    // Reference output: NCCL layer.
    layer([&] { dispatchNccl(); expert(); }, combineNccl)();
    CK(cudaMemcpyAsync(outRef, out, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, s1));
    // Output elements differing from the reference over 5 layers, all GPUs; xr starts as 0xFF (NaN) so an early GEMM shows.
    const auto diffs = [&](const std::function<void()>& op) {
      CK(cudaMemsetAsync(bad, 0, sizeof(*bad), s1));
      for (int i = 0; i < 5; ++i) {
        CK(cudaMemsetAsync(xr, 0xFF, size_t(tokens) * H * 2, s1)); CK(cudaMemsetAsync(out, 0, size_t(T) * H * 2, s1));
        op();
        countDiff<<<1024, 256, 0, s1>>>(out, outRef, size_t(T) * H, bad);
      }
      unsigned long long nb = 0; CK(cudaMemcpyAsync(&nb, bad, sizeof(nb), cudaMemcpyDeviceToHost, s1)); CK(cudaStreamSynchronize(s1));
      unsigned long long t = 0; MPI_Allreduce(&nb, &t, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      return t;
    };
    const auto timeMs = [&](const std::function<void()>& op) {
      for (int i = 0; i < 3; ++i) op();
      CK(cudaStreamSynchronize(s1)); MPI_Barrier(MPI_COMM_WORLD);
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0, s1)); for (int i = 0; i < 10; ++i) op(); CK(cudaEventRecord(e1, s1));
      CK(cudaEventSynchronize(e1)); float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); ms /= 10;
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
      float worst = 0; MPI_Allreduce(&ms, &worst, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      return worst;
    };

    struct V { std::string name; std::function<void()> op; long long diff = -1; std::vector<float> t = {}; };
    std::vector<V> vs{{"pack", pack}, {"dispatch_nccl", dispatchNccl}, {"dispatch_purlin", dispatchPurlin}, {"expert", expert},
                      {"reduce", reduce}, {"combine_nccl", combineNccl}, {"combine_purlin", combinePurlin}, {"unpack", unpack}};
    const auto add = [&](const std::string& n, std::function<void()> dg, std::function<void()> comb) {
      vs.push_back({"dg_" + n, dg});
      const auto l = layer(dg, comb);
      vs.push_back({"layer_" + n, l, n == "nccl" ? -1 : (long long)diffs(l)});
    };
    add("nccl", [&] { dispatchNccl(); expert(); }, combineNccl);
    add("purlin", [&] { dispatchPurlin(); expert(); }, combinePurlin);
    constexpr int REPS = 5;  // interleaved rounds
    for (int r = 0; r < REPS; ++r) for (auto& v : vs) v.t.push_back(timeMs(v.op));
    std::vector<float> med; for (auto& v : vs) { std::sort(v.t.begin(), v.t.end()); med.push_back(v.t[REPS / 2]); }
    const auto m = [&](const std::string& n) { for (size_t i = 0; i < vs.size(); ++i) if (vs[i].name == n) return med[i]; return 0.f; };
    for (size_t i = 0; i < vs.size(); ++i) {
      if (me) break;
      const auto& v = vs[i]; const auto us = v.name.find('_');
      const float ref = us == std::string::npos ? 0.f : m(v.name.substr(0, us + 1) + "nccl");
      std::printf("%s,%s,%.3f,%.3f,%.3f,", cname.c_str(), v.name.c_str(), med[i], v.t.front(), v.t.back());
      if (ref > 0) std::printf("%.3f,", ref / med[i]); else std::printf("-,");
      if (v.diff >= 0) std::printf("%lld\n", v.diff); else std::printf("-\n");
    }
    CK(cudaStreamSynchronize(s1));
    for (auto* p : allocs) CK(cudaFree(p));
    allocs.clear();
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
