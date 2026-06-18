#pragma once

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_matmul.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_select.h"
#include "fss/gpu_truncate.h"
#include "fss/gpu_dpf.h"
#include "fss/gpu_mul.h"

// Piranha softmax (llama / sytorch). The secure scale-24 softmax replaces the
// insecure debug softmax: pirhana_softmax outputs an additive share of
// (softmax(Z)/Bp + opMask); after the label-subtract and reconstruct we obtain
// the masked-public gradient dZ + mask (nothing about Z is revealed).
#include <sytorch/backend/llama_base.h>
#include <sytorch/softmax.h>

extern cudaMemPool_t mempool;
extern bool g_gpuMemPoolEnabled;

namespace standard_gcn_fss {

using T = u64;
static const u64 RELU_P = 0;
static const u64 RELU_Q = 0;

// ---------------------------------------------------------------------------
// The run is split into a single offline KEYGEN pass (the dealer's job; keys
// never cross the party-to-party link and depend only on random masks, so they
// are generated ONCE and reused across epochs) and an ONLINE pass that keeps
// every intermediate activation resident on the GPU and chains masked-public
// values from one FSS gate straight into the next -- no per-gate
// reconstruct -> secret-share -> reconstruct round trip and no CPU ping-pong.
// g_keygen_us / g_keygen_comm_bytes accumulate the offline cost for reporting.
// ---------------------------------------------------------------------------
inline u64 g_keygen_us = 0;
inline u64 g_keygen_comm_bytes = 0;

struct Meta
{
    int N = -1;
    int F = -1;
    int C = -1;
    int H = -1;
    int scale = -1;
    int train_count = -1;
    int test_count = -1;
    int normalized = 1;
};

inline std::string dataRoot()
{
    const char *e = std::getenv("FSS_DATA_ROOT");
    return std::string(e && *e ? e : "datasets/cora_standard_gcn");
}

inline int kvInt(const std::string &l)
{
    auto eq = l.find('=');
    return eq == std::string::npos ? -1 : std::atoi(l.c_str() + eq + 1);
}

inline void loadMeta(const std::string &path, Meta *m)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::fprintf(stderr, "ERROR: missing meta file: %s\n", path.c_str());
        std::abort();
    }
    std::string ln;
    while (std::getline(f, ln))
    {
        if      (ln.rfind("N=", 0) == 0) m->N = kvInt(ln);
        else if (ln.rfind("F=", 0) == 0) m->F = kvInt(ln);
        else if (ln.rfind("C=", 0) == 0) m->C = kvInt(ln);
        else if (ln.rfind("H=", 0) == 0) m->H = kvInt(ln);
        else if (ln.rfind("scale=", 0) == 0) m->scale = kvInt(ln);
        else if (ln.rfind("train_count=", 0) == 0) m->train_count = kvInt(ln);
        else if (ln.rfind("test_count=", 0) == 0) m->test_count = kvInt(ln);
        else if (ln.rfind("normalized=", 0) == 0) m->normalized = kvInt(ln);
    }
    if (m->N <= 0 || m->F <= 0 || m->C <= 0 || m->H <= 0 || m->scale <= 0)
    {
        std::fprintf(stderr, "ERROR: invalid meta in %s\n", path.c_str());
        std::abort();
    }
}

template <typename U>
inline U *readBinT(const std::string &path, size_t elems)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
    {
        std::fprintf(stderr, "ERROR: missing binary file: %s\n", path.c_str());
        std::abort();
    }
    size_t bytes = (size_t)f.tellg();
    if (bytes != elems * sizeof(U))
    {
        std::fprintf(stderr, "ERROR: size mismatch for %s: got %zu bytes, expected %zu\n",
                     path.c_str(), bytes, elems * sizeof(U));
        std::abort();
    }
    f.seekg(0);
    U *buf = new U[elems];
    f.read((char *)buf, bytes);
    return buf;
}

inline T *readBin(const std::string &path, size_t elems)
{
    return readBinT<T>(path, elems);
}

template <typename U>
inline void writeBinT(const std::string &path, const U *buf, size_t elems)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        std::abort();
    }
    f.write((const char *)buf, elems * sizeof(U));
}

inline void writeBin(const std::string &path, const T *buf, size_t elems)
{
    writeBinT<T>(path, buf, elems);
}

inline void ensureDir(const std::string &p)
{
    mkdir(p.c_str(), 0755);
}

inline bool parseIntArg(const char *text, int *out)
{
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || v < INT_MIN || v > INT_MAX)
        return false;
    *out = (int)v;
    return true;
}

inline bool parseDoubleArg(const char *text, double *out)
{
    char *end = nullptr;
    errno = 0;
    double v = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(v))
        return false;
    *out = v;
    return true;
}

inline int asInt(size_t n)
{
    assert(n <= (size_t)INT_MAX && "GPU helper takes int element counts");
    return (int)n;
}

inline void trimGpuPool()
{
    if (!g_gpuMemPoolEnabled)
        return;
    uint64_t threshold = 0;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    cudaDeviceSynchronize();
    cudaMemPoolTrimTo(mempool, 0);
}

inline std::string graphSharePath(const std::string &root, const char *name, int party)
{
    return root + "/graph/" + name + "_share" + std::to_string(party) + ".bin";
}

inline std::string weightSharePath(const std::string &root, const char *name, int party)
{
    return root + "/weights/" + name + "_share" + std::to_string(party) + ".bin";
}

inline T fixedFromDouble(double v, int scale, const char *tag, bool allow_zero = false)
{
    if (!std::isfinite(v))
    {
        std::fprintf(stderr, "ERROR: %s is non-finite\n", tag);
        std::abort();
    }
    if (scale < 0 || scale > 60)
    {
        std::fprintf(stderr, "ERROR: scale=%d out of supported range for %s\n", scale, tag);
        std::abort();
    }
    long double qd = (long double)v * (long double)(1ULL << scale);
    long long q = (long long)std::llround(qd);
    if (!allow_zero && q == 0)
    {
        std::fprintf(stderr, "ERROR: %s=%g quantizes to zero at scale=%d\n", tag, v, scale);
        std::abort();
    }
    return (T)q;
}

// ---------------------------------------------------------------------------
// MatmulParams builders.
//
// stdInit sets shift=scale, so every gpuMatmul truncates its product back to
// `scale` internally (one fused reconstruct + truncate per matmul). For the
// backward pass we never transpose data on the host: cutlass reads an operand
// in transposed layout when rowMaj_A / rowMaj_B is cleared, exactly like the
// orca FC layer's pdW / pdX. mmF: row-major C[M,N] = A[M,K] * B[K,N].
// mmTA (dW style): C[K0,N0] = A[M0,K0]^T * B[M0,N0]; pass forward dims
// (M0,K0,N0); A is the reused forward "left" operand. mmTB (dX style):
// C[M0,K0] = A[M0,N0] * B[K0,N0]^T; pass forward dims (M0,K0,N0); B is the
// reused forward "right" operand.
// ---------------------------------------------------------------------------
inline MatmulParams mmF(int M, int K, int N, int bw, int scale)
{
    MatmulParams p;
    p.batchSz = 1;
    p.M = M; p.K = K; p.N = N;
    stdInit(p, bw, scale);
    return p;
}

inline MatmulParams mmTA(int M0, int K0, int N0, int bw, int scale)
{
    MatmulParams p;
    p.batchSz = 1;
    p.M = K0; p.K = M0; p.N = N0;
    stdInit(p, bw, scale);
    p.rowMaj_A = false;
    return p;
}

inline MatmulParams mmTB(int M0, int K0, int N0, int bw, int scale)
{
    MatmulParams p;
    p.batchSz = 1;
    p.M = M0; p.K = N0; p.N = K0;
    stdInit(p, bw, scale);
    p.rowMaj_B = false;
    return p;
}

// ---------------------------------------------------------------------------
// Masked-public helpers.
// ---------------------------------------------------------------------------

// Open a fresh additive secret share to its masked-public form using the
// per-party mask share stored in the key buffer (one reconstruct).
inline T *openShare(GpuPeer *peer, const T *h_share, T *d_mask_share,
                    size_t elems, int bw, Stats *s)
{
    T *d_open = (T *)moveToGPU((u8 *)h_share, elems * sizeof(T), s);
    gpuLinearComb(bw, asInt(elems), d_open, T(1), d_open, T(1), d_mask_share);
    peer->reconstructInPlace(d_open, bw, elems, s);
    return d_open;
}

// Convert a masked-public value (v + mask) back to an additive secret share of
// v using this party's mask share. Purely local (no communication); used only
// at the few boundaries where a value must leave the masked-public domain:
// secret-bias add, weight/bias gradient updates, softmax input.
inline T *publicMaskedToShare(T *d_public_masked, T *d_mask_share,
                              int party, size_t elems, int bw)
{
    T public_coeff = party == SERVER0 ? T(1) : T(0);
    gpuLinearComb(bw, asInt(elems), d_public_masked,
                  public_coeff, d_public_masked, T(-1), d_mask_share);
    return d_public_masked;
}

inline T *revealShareCpu(GpuPeer *peer, const T *h_share, size_t elems, int bw, Stats *s)
{
    T *d = (T *)moveToGPU((u8 *)h_share, elems * sizeof(T), s);
    peer->reconstructInPlace(d, bw, elems, s);
    T *h_clear = (T *)moveToCPU((u8 *)d, elems * sizeof(T), s);
    gpuFree(d);
    return h_clear;
}

inline T *sumRows(const T *h, int rows, int cols)
{
    T *out = new T[cols]();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[j] += h[(size_t)i * cols + j];
    return out;
}

inline void rowZeroByMaskInPlace(T *h, const uint8_t *mask, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
        if (!mask[i])
            std::memset(&h[(size_t)i * cols], 0, (size_t)cols * sizeof(T));
}

inline void subtractInPlace(T *a, const T *b, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        a[i] -= b[i];
}

inline void applyStepInPlace(T *param, const T *step, size_t elems)
{
    for (size_t i = 0; i < elems; ++i)
        param[i] -= step[i];
}

inline int countMask(const uint8_t *mask, int n)
{
    int c = 0;
    for (int i = 0; i < n; ++i)
        c += mask[i] ? 1 : 0;
    return c;
}

inline double accuracyFromProbs(const T *post_clear, const int64_t *labels,
                                const uint8_t *mask, int N, int C)
{
    int correct = 0, total = 0;
    for (int i = 0; i < N; ++i)
    {
        if (!mask[i]) continue;
        int best = 0;
        T best_v = post_clear[(size_t)i * C];
        for (int c = 1; c < C; ++c)
        {
            T v = post_clear[(size_t)i * C + c];
            if ((i64)v > (i64)best_v) { best_v = v; best = c; }
        }
        correct += (best == (int)labels[i]);
        total++;
    }
    return total == 0 ? 0.0 : (double)correct / (double)total;
}

// ---------------------------------------------------------------------------
// Piranha softmax (secure scale-24, the only softmax path). Mirrors
// GPU-GCN/experiments/GCN
// gcn_train_common.h. Both party processes run keygen with synchronized
// randomness, so the FULL clear opMask is identical in both and writeShares
// splits it into per-party shares.
//
//   gpuGenSoftmaxKey (DEALER): feeds mZ (the clear A.T2 output mask) as the
//     softmax input mask, runs pirhana_softmax in DEALER mode (writes the
//     softmax FSS key into the llama buffer cursor == the GPU key cursor) and
//     returns the clear output mask opMask.
//   piranhaP (SERVER/CLIENT): pirhana_softmax(Z_masked_public) -> additive
//     share of (softmax(Z)/Bp + opMask); reconstruct -> masked-public value
//     softmax(Z)/Bp + opMask (opMask is full-width uniform => Z is fully hidden,
//     nothing about Z leaks); convert back to a share of softmax(Z)/Bp using the
//     opMask share; multiply by Bp (share-safe integer op, cancels pirhana's
//     /Bp) -> an additive SHARE of P = softmax(Z) at `scale`, IDENTICAL in
//     meaning to the debug-softmax posterior share, so the existing backward
//     (dZ = (P - Y) * lr/train_count, train-masked) is reused verbatim.
// ---------------------------------------------------------------------------
inline int piranhaBp(int B)
{
    int Bp = 1;
    while (Bp < B) Bp <<= 1;   // PiranhaSoftmax requires a power-of-two row count
    return Bp;
}

inline T *gpuGenSoftmaxKey(int B, int C, T *d_mZ, LlamaBase<u64> *llama, int scale)
{
    int Bp = piranhaBp(B);
    Tensor4D<u64> inpMask(Bp, C, 1, 1);
    Tensor4D<u64> opMask(Bp, C, 1, 1);
    size_t realSz = (size_t)B * C * sizeof(T);
    std::memset(inpMask.data, 0, (size_t)Bp * C * sizeof(u64));
    moveIntoCPUMem((u8 *)inpMask.data, (u8 *)d_mZ, realSz, nullptr); // first B rows
    pirhana_softmax(inpMask, opMask, (u64)scale);
    return (T *)moveToGPU((u8 *)opMask.data, realSz, nullptr);       // first B rows
}

// Returns the posterior P = softmax(Z) as an additive secret share (CPU), at
// `scale`. `cur` is the live key cursor sitting at the softmax FSS key; after
// pirhana_softmax consumes it (via llama) the opMask shares are read from *cur.
inline T *piranhaP(SigmaPeer *peer, int party, const T *h_Z_masked_public,
                   u8 **cur, int B, int C, int scale, LlamaBase<u64> *llama)
{
    int Bp = piranhaBp(B);
    Tensor4D<u64> inp(Bp, C, 1, 1);
    Tensor4D<u64> op(Bp, C, 1, 1);
    std::memset(inp.data, 0, (size_t)Bp * C * sizeof(u64));
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            inp(b, c, 0, 0) = h_Z_masked_public[(size_t)b * C + c]; // masked-public Z
    pirhana_softmax(inp, op, (u64)scale);            // share of softmax/Bp + opMask
    reconstruct(B * C, op.data, 64);                 // -> masked-public (fully hidden by opMask)
    // opMask shares follow the softmax key in the stream.
    const T *h_opMask_share = (const T *)(*cur);
    *cur += (size_t)B * C * sizeof(T);
    // masked-public -> share of softmax/Bp (party0 keeps public, party1 zeroes),
    // then *Bp to undo pirhana's /Bp -> share of P = softmax(Z) at `scale`.
    T *h_P = new T[(size_t)B * C];
    const T Bp_T = (T)Bp;
    for (size_t i = 0; i < (size_t)B * C; ++i)
    {
        T pub = (party == SERVER0) ? op.data[i] : T(0);
        T share = pub - h_opMask_share[i];           // share of softmax/Bp
        h_P[i] = share * Bp_T;                        // share of softmax = P
    }
    return h_P;
}

// ===========================================================================
// GCN forward/backward in the mask-chained dealer/evaluator style.
//
// Forward:  T1 = X.W1 ; U1 = A.T1 + b1 ; H1 = ReLU(U1) ;
//           T2 = H1.W2 ; Z = A.T2 + b2 ; P = softmax(Z)
// Backward: dZ = (P - Y) * lr/train ; dT2 = A^T.dZ ; dW2 = H1^T.dT2 ;
//           dH1 = dT2.W2^T ; dU1 = DReLU(U1) (.) dH1 ; dT1 = A^T.dU1 ;
//           dW1 = X^T.dT1 ; db1 = sum_rows dU1 ; db2 = sum_rows dZ
//
// The dealer threads each gate's OUTPUT mask in as the next gate's INPUT mask,
// so the evaluator reconstructs each masked-public value exactly once and feeds
// it straight to the next gate. Secret biases (b1,b2) and the softmax input are
// the only points where a value leaves the masked-public domain. Keys depend
// only on the random masks, so one buffer serves every training epoch.
// ===========================================================================

struct GcnDims
{
    int N, F, H, C, bw, scale;
    TruncateType tr = TruncateType::TrWithSlack;
    bool secretMask = false;   // train_mask is a SECRET share (oblivious unlearn); the
                               // gradient is beaver-multiplied by it instead of a public
                               // rowZero (revealing the mask would fingerprint the shard).
};

inline GcnDims makeDims(const Meta &m)
{
    GcnDims d;
    d.N = m.N; d.F = m.F; d.H = m.H; d.C = m.C; d.bw = 64; d.scale = m.scale;
    return d;
}

// Forward state kept resident on the GPU between forward and backward.
struct GcnFwdState
{
    T *d_A = nullptr;   // masked-public adjacency (reused 4x incl. transposes)
    T *d_X = nullptr;   // masked-public features
    T *d_H1 = nullptr;  // masked-public ReLU output
    T *d_U1 = nullptr;  // masked-public ReLU input (for backward DReLU)
    T *d_W2 = nullptr;  // masked-public W2 (for backward dH1 = dT2.W2^T)
    T *h_P = nullptr;   // posterior P = softmax(Z), additive share (CPU)
    T *h_Z = nullptr;   // logits, additive share (CPU; revealed only on eval)
};

inline void freeFwdState(GcnFwdState *st)
{
    if (st->d_A) gpuFree(st->d_A);
    if (st->d_X) gpuFree(st->d_X);
    if (st->d_H1) gpuFree(st->d_H1);
    if (st->d_U1) gpuFree(st->d_U1);
    if (st->d_W2) gpuFree(st->d_W2);
    if (st->h_P) delete[] st->h_P;
    if (st->h_Z) delete[] st->h_Z;
    *st = GcnFwdState();
}

inline u8 *readMaskShareCpu(u8 **cur, size_t elems)
{
    u8 *p = *cur;
    *cur += elems * sizeof(T);
    return p;
}

// -------------------------------- KEYGEN -----------------------------------
// Run by BOTH parties with synchronized randomness; writeShares emits
// complementary per-party shares. Writes all keys into [*kptr ...).
//
// `llama` (DEALER mode, key cursor bound to *kptr) is used at the softmax point
// to generate the piranha-softmax FSS key (the only softmax path) and write the
// opMask shares right after the mZ shares; the backward keygen (random dZ mask)
// is independent because the forward converts the masked-public softmax back to
// a posterior SHARE before the gradient pipeline runs.
inline void gcnKeygen(u8 **kptr, int party, const GcnDims &d,
                      AESGlobalContext *gAES, bool train,
                      LlamaBase<u64> *llama)
{
    assert(llama != nullptr && "piranha softmax requires a DEALER llama");
    const int N = d.N, F = d.F, H = d.H, C = d.C, bw = d.bw, scale = d.scale;
    const TruncateType tr = d.tr;
    const size_t NF = (size_t)N * F, NN = (size_t)N * N, FH = (size_t)F * H,
                 NH = (size_t)N * H, HC = (size_t)H * C, NC = (size_t)N * C;

    auto mX  = randomGEOnGpu<T>(NF, bw);
    auto mA  = randomGEOnGpu<T>(NN, bw);
    auto mW1 = randomGEOnGpu<T>(FH, bw);
    auto mW2 = randomGEOnGpu<T>(HC, bw);

    auto p_T1 = mmF(N, F, H, bw, scale);
    auto mT1 = gpuKeygenMatmul<T>(kptr, party, p_T1, mX, mW1, (T *)nullptr, tr, gAES, true);
    gpuFree(mW1);

    // U1 = A.T1 (no bias): the matmul output mask chains straight into the
    // ReLU input mask -- no re-share, no reopen.
    auto p_U1 = mmF(N, N, H, bw, scale);
    auto mU1 = gpuKeygenMatmul<T>(kptr, party, p_U1, mA, mT1, (T *)nullptr, tr, gAES, true);
    gpuFree(mT1);

    auto mH1 = gpuGenReluKey<T, T, RELU_P, RELU_Q, false>(kptr, party, bw, bw, asInt(NH), mU1, gAES);

    auto p_T2 = mmF(N, H, C, bw, scale);
    auto mT2 = gpuKeygenMatmul<T>(kptr, party, p_T2, mH1, mW2, (T *)nullptr, tr, gAES, true);

    // Z = A.T2 (no bias). mZ is re-shared (writeShares) so the evaluator can
    // reveal Z under --reveal-eval (argmax for accuracy); the default gradient
    // path never reconstructs Z. mZ is ALSO the softmax input mask.
    auto p_Z = mmF(N, N, C, bw, scale);
    auto mZ = gpuKeygenMatmul<T>(kptr, party, p_Z, mA, mT2, (T *)nullptr, tr, gAES, true);
    writeShares<T, T>(kptr, party, asInt(NC), mZ, bw);
    gpuFree(mT2);

    // Piranha softmax FSS key (DEALER): writes into the same key cursor right
    // after the mZ shares. opMask (clear) is re-shared so the evaluator can peel
    // it off the masked-public softmax and recover a posterior SHARE.
    {
        T *opMask = gpuGenSoftmaxKey(N, C, mZ, llama, scale);
        writeShares<T, T>(kptr, party, asInt(NC), opMask, bw);  // opMask share
        gpuFree(opMask);
    }
    gpuFree(mZ);

    if (!train)
    {
        gpuFree(mX); gpuFree(mA); gpuFree(mW2); gpuFree(mU1); gpuFree(mH1);
        return;
    }

    const int coeff_scale = scale + 16;

    // Backward dZ mask: a fresh uniform mask -- the forward yields a posterior
    // SHARE, so the gradient is opened with a plain re-share.
    auto m_dzs = randomGEOnGpu<T>(NC, bw);
    writeShares<T, T>(kptr, party, asInt(NC), m_dzs, bw);    // open dZ_scaled
    auto m_dZ = genGPUTruncateKey<T, T>(kptr, party, tr, bw, bw, coeff_scale, asInt(NC), m_dzs, gAES);
    gpuFree(m_dzs);

    // Secret-mask path: a beaver-mul key to apply the SECRET train_mask to dZ
    // (dZ ⊙ mask_bc, mask broadcast over C). scale=0 (mask is 0/1, no truncation).
    if (d.secretMask)
    {
        auto mDz = randomGEOnGpu<T>(NC, bw);
        auto mMk = randomGEOnGpu<T>(NC, bw);
        auto mC = gpuKeygenMul<T>(kptr, party, bw, 0, asInt(NC), mDz, mMk, TruncateType::None, gAES);
        writeShares<T, T>(kptr, party, asInt(NC), mC, bw);   // [maskMulC] output-mask share
        gpuFree(mDz); gpuFree(mMk); gpuFree(mC);
    }

    auto p_dT2 = mmTA(N, N, C, bw, scale);
    auto m_dT2 = gpuKeygenMatmul<T>(kptr, party, p_dT2, mA, m_dZ, (T *)nullptr, tr, gAES, true);
    gpuFree(m_dZ);

    auto p_dW2 = mmTA(N, H, C, bw, scale);
    auto m_dW2 = gpuKeygenMatmul<T>(kptr, party, p_dW2, mH1, m_dT2, (T *)nullptr, tr, gAES, true);
    writeShares<T, T>(kptr, party, asInt(HC), m_dW2, bw);    // W2 update reshare
    gpuFree(m_dW2);
    gpuFree(mH1);

    auto p_dH1 = mmTB(N, H, C, bw, scale);
    auto m_dH1 = gpuKeygenMatmul<T>(kptr, party, p_dH1, m_dT2, mW2, (T *)nullptr, tr, gAES, true);
    gpuFree(m_dT2);
    gpuFree(mW2);

    auto m_drelu = gpuKeyGenDRelu<T>(kptr, party, bw, asInt(NH), mU1, gAES);
    auto m_dU1 = gpuKeyGenSelect<T, T, T>(kptr, party, asInt(NH), m_dH1, m_drelu, bw);
    gpuFree(m_dH1);
    gpuFree(m_drelu);
    gpuFree(mU1);

    auto p_dT1 = mmTA(N, N, H, bw, scale);
    auto m_dT1 = gpuKeygenMatmul<T>(kptr, party, p_dT1, mA, m_dU1, (T *)nullptr, tr, gAES, true);
    gpuFree(m_dU1);
    gpuFree(mA);

    auto p_dW1 = mmTA(N, F, H, bw, scale);
    auto m_dW1 = gpuKeygenMatmul<T>(kptr, party, p_dW1, mX, m_dT1, (T *)nullptr, tr, gAES, true);
    writeShares<T, T>(kptr, party, asInt(FH), m_dW1, bw);    // W1 update reshare
    gpuFree(m_dW1);
    gpuFree(m_dT1);
    gpuFree(mX);
}

// ===========================================================================
// ONLINE -- split into a key-reading step and a run step (wing style).
//
//   gcnReadForwardKeys / gcnReadBackwardKeys : pure pointer setup. They only
//       carve the offline key material out of the key buffer and advance the
//       cursor. No GPU work, no communication. (cf. wing layer->readForwardKey)
//
//   gcnForwardRun / gcnBackwardRun : the actual 2-party protocol. Every line
//       here is real online work -- openShare() reconstructs a masked-public
//       value (communication), gpuMatmul / gpuRelu / gpuSelect / gpuTruncate
//       are the FSS gates. Activations stay resident on the GPU and each
//       masked-public value feeds straight into the next gate.
// ===========================================================================

// Forward key material (pointers into the key buffer; no data is copied).
struct GcnFwdKeys
{
    MatmulParams pT1, pU1, pT2, pZ;
    GPUMatmulKey<T> kT1, kU1, kT2, kZ;
    GPUReluKey<T> kRelu;
    T *mZ = nullptr;       // mask share of Z = A.T2 (Z reveal under --reveal-eval)
};

// After the mZ shares the cursor sits at the softmax FSS key (read inside
// gcnForwardRun via the llama cursor); the opMask shares follow the softmax key
// and are read by gcnForwardRun AFTER the softmax (write order is mZ,
// softmax-key, opMask -- matching gpuGenSoftmaxKey's single dealer call).
inline void gcnReadForwardKeys(u8 **cur, const GcnDims &d, GcnFwdKeys *k)
{
    const int N = d.N, F = d.F, H = d.H, C = d.C, bw = d.bw, scale = d.scale;
    const TruncateType tr = d.tr;
    const size_t NH = (size_t)N * H, NC = (size_t)N * C;

    k->pT1 = mmF(N, F, H, bw, scale);
    k->kT1 = readGPUMatmulKey<T>(k->pT1, tr, cur);

    k->pU1 = mmF(N, N, H, bw, scale);
    k->kU1 = readGPUMatmulKey<T>(k->pU1, tr, cur);

    k->kRelu = readReluKey<T>(cur);

    k->pT2 = mmF(N, H, C, bw, scale);
    k->kT2 = readGPUMatmulKey<T>(k->pT2, tr, cur);

    k->pZ = mmF(N, N, C, bw, scale);
    k->kZ = readGPUMatmulKey<T>(k->pZ, tr, cur);
    k->mZ = (T *)readMaskShareCpu(cur, NC);
    // NOTE: the softmax FSS key and the opMask shares are consumed by
    // gcnForwardRun (the softmax key via llama, then opMask via *cur).
}

// Secure piranha softmax -> posterior share st.h_P. st.h_P is an additive share
// of P = softmax(Z), and st.h_Z (a plain additive share, no leak) is produced
// for --reveal-eval / inference output. Z itself is never reconstructed.
// `cur` is the live key cursor positioned at the softmax FSS key so the softmax
// can consume it (via llama) and then read the opMask shares.
inline GcnFwdState gcnForwardRun(GpuPeer *peer, int party, AESGlobalContext *gAES,
                                 const GcnDims &d, GcnFwdKeys &k,
                                 const T *h_X, const T *h_A,
                                 const T *h_W1, const T *h_W2,
                                 Stats *s,
                                 LlamaBase<u64> *llama, u8 **cur)
{
    assert(llama != nullptr && cur != nullptr && "piranha softmax requires llama + key cursor");
    const int N = d.N, H = d.H, C = d.C, bw = d.bw, scale = d.scale;
    const TruncateType tr = d.tr;
    const size_t NF = (size_t)N * d.F, NN = (size_t)N * N, FH = (size_t)d.F * H,
                 NH = (size_t)N * H, HC = (size_t)H * C, NC = (size_t)N * C;
    GcnFwdState st;
    T *d_T1 = nullptr, *d_T2 = nullptr;

    // (1) T1 = X . W1 : open both secret operands once, then one matmul.
    {
        auto d_mW1 = (T *)moveToGPU((u8 *)k.kT1.B, FH * sizeof(T), s);
        auto d_W1 = openShare(peer, h_W1, d_mW1, FH, bw, s);
        gpuFree(d_mW1);
        auto d_mX = (T *)moveToGPU((u8 *)k.kT1.A, NF * sizeof(T), s);
        st.d_X = openShare(peer, h_X, d_mX, NF, bw, s);     // kept for backward dW1
        gpuFree(d_mX);
        d_T1 = gpuMatmul<T>(peer, party, k.pT1, k.kT1, st.d_X, d_W1,
                                   (T *)nullptr, tr, gAES, s, true);
        gpuFree(d_W1);
    }

    // (2) U1 = A . T1 (no bias): the matmul output is already masked-public and
    //     feeds straight into ReLU -- no re-share, no reopen.
    {
        auto d_mA = (T *)moveToGPU((u8 *)k.kU1.A, NN * sizeof(T), s);
        st.d_A = openShare(peer, h_A, d_mA, NN, bw, s);     // opened once, reused 4x
        gpuFree(d_mA);
        st.d_U1 = gpuMatmul<T>(peer, party, k.pU1, k.kU1, st.d_A, d_T1,
                               (T *)nullptr, tr, gAES, s, true);  // kept for backward DReLU
        gpuFree(d_T1); d_T1 = nullptr;
    }

    // (3) H1 = ReLU(U1) : one DReLU + one select, reusing the masked U1.
    st.d_H1 = gpuRelu<T, T, RELU_P, RELU_Q, false>(peer, party, k.kRelu, st.d_U1, gAES, s);

    // (4) T2 = H1 . W2
    {
        auto d_mW2 = (T *)moveToGPU((u8 *)k.kT2.B, HC * sizeof(T), s);
        st.d_W2 = openShare(peer, h_W2, d_mW2, HC, bw, s);   // kept for backward dH1
        gpuFree(d_mW2);
        d_T2 = gpuMatmul<T>(peer, party, k.pT2, k.kT2, st.d_H1, st.d_W2,
                                   (T *)nullptr, tr, gAES, s, true);
    }

    // (5) Z = A . T2 (no bias). Z is masked-public (mask mZ). We keep the
    //     masked-public Z on the CPU (piranha softmax input) AND re-share it to
    //     an additive share st.h_Z (purely local, no leak) for eval/inference.
    T *h_Z_mp = nullptr;
    {
        auto d_Z = gpuMatmul<T>(peer, party, k.pZ, k.kZ, st.d_A, d_T2,
                                (T *)nullptr, tr, gAES, s, true);
        gpuFree(d_T2); d_T2 = nullptr;
        // snapshot masked-public Z before converting to a share (softmax input)
        h_Z_mp = new T[NC];
        T *tmp_mp = (T *)moveToCPU((u8 *)d_Z, NC * sizeof(T), s);
        std::memcpy(h_Z_mp, tmp_mp, NC * sizeof(T));
        cpuFree(tmp_mp);
        auto d_mZ = (T *)moveToGPU((u8 *)k.mZ, NC * sizeof(T), s);
        publicMaskedToShare(d_Z, d_mZ, party, NC, bw);
        gpuFree(d_mZ);
        st.h_Z = new T[NC];
        T *tmp = (T *)moveToCPU((u8 *)d_Z, NC * sizeof(T), s);
        std::memcpy(st.h_Z, tmp, NC * sizeof(T));
        cpuFree(tmp);
        gpuFree(d_Z);
    }

    // (6) Softmax -> posterior share st.h_P. Piranha softmax on the masked-public
    //     Z (never reconstructs Z), then peel opMask and undo the /Bp to recover
    //     the posterior share. pirhana_softmax consumes the softmax FSS key from
    //     *cur (via llama); the opMask shares sit right after it in the key stream.
    st.h_P = piranhaP(peer, party, h_Z_mp, cur, N, C, scale, llama);
    delete[] h_Z_mp;
    return st;
}

// Backward key material (pointers into the key buffer).
struct GcnBwdKeys
{
    MatmulParams p_dT2, p_dW2, p_dH1, p_dT1, p_dW1;
    T *m_dzs = nullptr;            // mask share to open scaled dZ
    GPUTruncateKey<T> kTr;         // truncate dZ_scaled by coeff_scale
    GPUMulKey<T> kMaskMul;         // beaver key for secret-mask: dZ ⊙ mask_bc (secretMask only)
    T *m_maskC = nullptr;          // its output-mask share
    GPUMatmulKey<T> k_dT2, k_dW2, k_dH1, k_dT1, k_dW1;
    T *m_dW2 = nullptr;            // dW2 mask share (W2 update re-share)
    GPUDReluKey kdr;
    GPUSelectKey<T> ksel;
    T *m_dW1 = nullptr;            // dW1 mask share (W1 update re-share)
};

inline void gcnReadBackwardKeys(u8 **cur, const GcnDims &d, GcnBwdKeys *k)
{
    const int N = d.N, F = d.F, H = d.H, C = d.C, bw = d.bw, scale = d.scale;
    const TruncateType tr = d.tr;
    const size_t FH = (size_t)F * H, NH = (size_t)N * H, HC = (size_t)H * C,
                 NC = (size_t)N * C;

    k->m_dzs = (T *)readMaskShareCpu(cur, NC);
    k->kTr = readGPUTruncateKey<T>(tr, cur);
    if (d.secretMask)
    {
        k->kMaskMul = readGPUMulKey<T>(cur, NC, NC, NC, TruncateType::None);
        k->m_maskC = (T *)readMaskShareCpu(cur, NC);
    }

    k->p_dT2 = mmTA(N, N, C, bw, scale);
    k->k_dT2 = readGPUMatmulKey<T>(k->p_dT2, tr, cur);

    k->p_dW2 = mmTA(N, H, C, bw, scale);
    k->k_dW2 = readGPUMatmulKey<T>(k->p_dW2, tr, cur);
    k->m_dW2 = (T *)readMaskShareCpu(cur, HC);

    k->p_dH1 = mmTB(N, H, C, bw, scale);
    k->k_dH1 = readGPUMatmulKey<T>(k->p_dH1, tr, cur);

    k->kdr = readGPUDReluKey(cur);
    k->ksel = readGPUSelectKey<T>(cur, asInt(NH));

    k->p_dT1 = mmTA(N, N, H, bw, scale);
    k->k_dT1 = readGPUMatmulKey<T>(k->p_dT1, tr, cur);

    k->p_dW1 = mmTA(N, F, H, bw, scale);
    k->k_dW1 = readGPUMatmulKey<T>(k->p_dW1, tr, cur);
    k->m_dW1 = (T *)readMaskShareCpu(cur, FH);
}

inline void gcnBackwardRun(GpuPeer *peer, int party, AESGlobalContext *gAES,
                           const GcnDims &d, GcnBwdKeys &k, GcnFwdState *st,
                           const T *h_Y, const uint8_t *train_mask,
                           const T *h_tmask_share, T coeff_fp,
                           T *h_W1, T *h_W2, Stats *s)
{
    const int N = d.N, F = d.F, H = d.H, C = d.C, bw = d.bw, scale = d.scale;
    const TruncateType tr = d.tr;
    const int coeff_scale = scale + 16;
    const size_t FH = (size_t)F * H, NH = (size_t)N * H, HC = (size_t)H * C,
                 NC = (size_t)N * C;

    // (B1) dZ = (P - Y), masked to train rows, scaled by coeff_fp.
    // st->h_P is an additive share of P from either softmax path.
    T *h_dZ = new T[NC];
    std::memcpy(h_dZ, st->h_P, NC * sizeof(T));
    subtractInPlace(h_dZ, h_Y, NC);                              // [P - Y]
    if (d.secretMask)
    {
        // SECRET mask: beaver-multiply each gradient row by mask[i] (broadcast over C).
        // Revealing the mask would fingerprint the shard, so it stays a share.
        std::vector<T> h_mbc(NC);
        for (int i = 0; i < N; ++i)
            for (int cc = 0; cc < C; ++cc) h_mbc[(size_t)i * C + cc] = h_tmask_share[i];
        T *d_a = (T *)moveToGPU((u8 *)k.kMaskMul.a, NC * sizeof(T), s);   // [mask_dZ]
        T *d_dZ = (T *)moveToGPU((u8 *)h_dZ, NC * sizeof(T), s);
        gpuLinearComb(bw, asInt(NC), d_dZ, T(1), d_dZ, T(1), d_a);
        peer->reconstructInPlace(d_dZ, bw, NC, s);                       // dZ + mask_dZ (public)
        gpuFree(d_a);
        T *d_b = (T *)moveToGPU((u8 *)k.kMaskMul.b, NC * sizeof(T), s);   // [mask_mbc]
        T *d_mbc = (T *)moveToGPU((u8 *)h_mbc.data(), NC * sizeof(T), s);
        gpuLinearComb(bw, asInt(NC), d_mbc, T(1), d_mbc, T(1), d_b);
        peer->reconstructInPlace(d_mbc, bw, NC, s);                      // mask + mask_mbc (public)
        gpuFree(d_b);
        auto d_dZm = gpuMul<T>(peer, party, bw, 0, asInt(NC), k.kMaskMul, d_dZ, d_mbc,
                               TruncateType::None, gAES, s);             // masked-public dZ⊙mask + mC
        T *d_mc = (T *)moveToGPU((u8 *)k.m_maskC, NC * sizeof(T), s);
        gpuLinearComb(bw, asInt(NC), d_dZm, (party == SERVER0) ? T(1) : T(0), d_dZm, T(-1), d_mc);
        moveIntoCPUMem((u8 *)h_dZ, (u8 *)d_dZm, NC * sizeof(T), s);      // h_dZ = [dZ ⊙ mask]
        gpuFree(d_dZ); gpuFree(d_mbc); gpuFree(d_dZm); gpuFree(d_mc);
    }
    else
    {
        rowZeroByMaskInPlace(h_dZ, train_mask, N, C);            // public mask
    }
    for (size_t i = 0; i < NC; ++i) h_dZ[i] *= coeff_fp;          // -> scale + coeff_scale
    auto d_mdzs = (T *)moveToGPU((u8 *)k.m_dzs, NC * sizeof(T), s);
    auto d_dZsc = openShare(peer, h_dZ, d_mdzs, NC, bw, s);
    gpuFree(d_mdzs);
    delete[] h_dZ;

    // (B2) truncate by coeff_scale -> dZ_step (masked-public, scale).
    auto d_dZstep = gpuTruncate<T, T>(bw, bw, tr, k.kTr, coeff_scale, peer, party,
                                      asInt(NC), d_dZsc, gAES, s);
    if (d_dZstep != d_dZsc) gpuFree(d_dZsc);

    // (B3) dT2 = A^T . dZ_step  (A reused, read transposed via rowMaj_A=false)
    auto d_dT2 = gpuMatmul<T>(peer, party, k.p_dT2, k.k_dT2, st->d_A, d_dZstep,
                              (T *)nullptr, tr, gAES, s, true);
    gpuFree(d_dZstep);

    // (B4) dW2 = H1^T . dT2  -> W2 update
    auto d_dW2 = gpuMatmul<T>(peer, party, k.p_dW2, k.k_dW2, st->d_H1, d_dT2,
                              (T *)nullptr, tr, gAES, s, true);
    gpuFree(st->d_H1); st->d_H1 = nullptr;
    auto d_mdW2 = (T *)moveToGPU((u8 *)k.m_dW2, HC * sizeof(T), s);
    publicMaskedToShare(d_dW2, d_mdW2, party, HC, bw);
    gpuFree(d_mdW2);
    T *dW2 = (T *)moveToCPU((u8 *)d_dW2, HC * sizeof(T), s);
    gpuFree(d_dW2);

    // (B5) dH1 = dT2 . W2^T  (W2 reused, read transposed via rowMaj_B=false)
    auto d_dH1 = gpuMatmul<T>(peer, party, k.p_dH1, k.k_dH1, d_dT2, st->d_W2,
                              (T *)nullptr, tr, gAES, s, true);
    gpuFree(d_dT2);
    gpuFree(st->d_W2); st->d_W2 = nullptr;

    // (B6) dU1 = DReLU(U1) (.) dH1 : reuse masked-public U1 (no reopen) and the
    //      reconstructed grad (no second reconstruct of x), one select.
    std::vector<u32 *> hmask({k.kdr.mask});
    auto d_bit = gpuDcf<T, 1, dReluPrologue<0>, dReluEpilogue<0, false>>(
        k.kdr.dpfKey, party, st->d_U1, gAES, s, &hmask);
    peer->reconstructInPlace(d_bit, 1, NH, s);
    gpuFree(st->d_U1); st->d_U1 = nullptr;
    auto d_dU1 = gpuSelect<T, T, RELU_P, RELU_Q>(peer, party, bw, k.ksel, d_bit, d_dH1, s);
    gpuFree(d_bit);
    gpuFree(d_dH1);

    // (B7) dT1 = A^T . dU1
    auto d_dT1 = gpuMatmul<T>(peer, party, k.p_dT1, k.k_dT1, st->d_A, d_dU1,
                              (T *)nullptr, tr, gAES, s, true);
    gpuFree(d_dU1);
    gpuFree(st->d_A); st->d_A = nullptr;

    // (B8) dW1 = X^T . dT1  -> W1 update
    auto d_dW1 = gpuMatmul<T>(peer, party, k.p_dW1, k.k_dW1, st->d_X, d_dT1,
                              (T *)nullptr, tr, gAES, s, true);
    gpuFree(d_dT1);
    gpuFree(st->d_X); st->d_X = nullptr;
    auto d_mdW1 = (T *)moveToGPU((u8 *)k.m_dW1, FH * sizeof(T), s);
    publicMaskedToShare(d_dW1, d_mdW1, party, FH, bw);
    gpuFree(d_mdW1);
    T *dW1 = (T *)moveToCPU((u8 *)d_dW1, FH * sizeof(T), s);
    gpuFree(d_dW1);

    // SGD update on the CPU weight shares (no bias).
    applyStepInPlace(h_W1, dW1, FH);
    applyStepInPlace(h_W2, dW2, HC);
    cpuFree(dW1); cpuFree(dW2);
}

} // namespace standard_gcn_fss
