
#include <cassert>
#include <algorithm>
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

#include <sytorch/tensor.h>
#include <sytorch/backend/cleartext.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_matmul.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_softmax.h"

extern cudaMemPool_t mempool;

using T = u64;

static const u64 RELU_P = 0;
static const u64 RELU_Q = 0;

static const std::string DATA_ROOT = []()
{
    const char *e = std::getenv("FSS_DATA_ROOT");
    return std::string(e && *e ? e : "datasets/cora_shards");
}();

struct Meta
{
    int N = -1, F = -1, C = -1, k = -1, scale = -1, Ns = -1, num_test = -1, normalized = 0;
};

static int kvInt(const std::string &l)
{
    auto eq = l.find('=');
    return eq == std::string::npos ? -1 : std::atoi(l.c_str() + eq + 1);
}
static void loadMeta(const std::string &path, Meta *m)
{
    std::ifstream f(path);
    assert(f.is_open() && "missing meta file");
    std::string ln;
    while (std::getline(f, ln))
    {
        if (ln.rfind("N=", 0) == 0)
            m->N = kvInt(ln);
        else if (ln.rfind("F=", 0) == 0)
            m->F = kvInt(ln);
        else if (ln.rfind("C=", 0) == 0)
            m->C = kvInt(ln);
        else if (ln.rfind("k=", 0) == 0)
            m->k = kvInt(ln);
        else if (ln.rfind("scale=", 0) == 0)
            m->scale = kvInt(ln);
        else if (ln.rfind("Ns=", 0) == 0)
            m->Ns = kvInt(ln);
        else if (ln.rfind("num_test=", 0) == 0)
            m->num_test = kvInt(ln);
        else if (ln.rfind("normalized=", 0) == 0)
            m->normalized = kvInt(ln);
    }
}

static T *readBin(const std::string &path, size_t elems)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f.is_open());
    size_t bytes = (size_t)f.tellg();
    assert(bytes == elems * sizeof(T) && "bin size mismatch");
    f.seekg(0);
    T *b = new T[elems];
    f.read((char *)b, bytes);
    return b;
}
static void writeBin(const std::string &path, const T *buf, size_t elems)
{
    std::ofstream f(path, std::ios::binary);
    assert(f.is_open());
    f.write((const char *)buf, elems * sizeof(T));
}
static void ensureDir(const std::string &p) { mkdir(p.c_str(), 0755); }

static std::string sharePath(const std::string &prefix, const char *name, int party)
{
    return prefix + "_" + name + "_share" + std::to_string(party) + ".bin";
}

static bool parseIntArg(const char *text, int *out)
{
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || v < INT_MIN || v > INT_MAX)
        return false;
    *out = (int)v;
    return true;
}

static int asInt(size_t n)
{
    assert(n <= (size_t)INT_MAX && "GPU helper takes int element counts");
    return (int)n;
}

static T *expandBias(const T *b, int M, int out_dim)
{
    T *e = new T[(size_t)M * out_dim];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < out_dim; ++j)
            e[(size_t)i * out_dim + j] = b[j];
    return e;
}

static void cpuMatmulTrunc(const T *A, const T *B, T *C,
                           int M, int K, int N, int bw, int scale)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            T sum = 0;
            for (int k = 0; k < K; ++k)
                sum += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = cpuArs(sum, bw, scale);
            cpuMod(C[(size_t)i * N + j], bw);
        }
}
static void cpuReluInPlace(T *A, size_t n, int bw)
{
    for (size_t i = 0; i < n; ++i)
    {
        T x = A[i];
        T sign = x >> (bw - 1);
        A[i] = x * (T(1) - sign);
        cpuMod(A[i], bw);
    }
}

static int paddedClassCount(int C)
{
    int C_pad = 1;
    while (C_pad < C)
        C_pad <<= 1;
    return C_pad < 2 ? 2 : C_pad;
}

static i64 signExtendRingToI64(T x, int bw)
{
    if (bw >= 64)
        return (i64)x;
    T mask = (T(1) << bw) - 1;
    x &= mask;
    T sign = T(1) << (bw - 1);
    if (x & sign)
        x |= ~mask;
    return (i64)x;
}

static T *cpuSoftmaxBC(const T *h_logits, int B, int C, int scale)
{
    T *h_out = new T[(size_t)B * C];
    if (C <= 1)
    {
        T one_fp = T(1) << scale;
        for (size_t i = 0; i < (size_t)B * C; ++i)
            h_out[i] = one_fp;
        return h_out;
    }

    const int sm_bw = 50;
    const int C_pad = paddedClassCount(C);
    const i64 sentinel = -(i64(1) << 35); // Matches p.bin=38 sentinel in FSS softmax.
    std::vector<i64> padded((size_t)B * C_pad);
    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
            padded[(size_t)b * C_pad + c] =
                signExtendRingToI64(h_logits[(size_t)b * C + c], sm_bw);
        for (int c = C; c < C_pad; ++c)
            padded[(size_t)b * C_pad + c] = sentinel;
    }

    Tensor<i64> in(padded.data(), {(u64)B, (u64)C_pad});
    Tensor<i64> out({(u64)B, (u64)C_pad});
    auto ct = new ClearText<i64>();
    ct->bw = sm_bw;
    ct->softmax(in, out, scale, 0);
    delete ct;

    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            h_out[(size_t)b * C + c] = (T)out.data[(size_t)b * C_pad + c];
    return h_out;
}

// =========================================================================
// Share/mask helpers
// =========================================================================
static T *splitMaskForParty(T *d_full_mask, int party, size_t elems, int bw)
{
    T *d_share = randomGEOnGpu<T>(elems, bw); // mask_share_0
    if (party == SERVER1)
        gpuLinearComb(bw, asInt(elems), d_share, T(1), d_full_mask, T(-1), d_share);
    return d_share;
}

static T *openMaskedInputFromCpuShare(GpuPeer *peer, const T *h_value_share,
                                      T *d_mask_share, size_t elems, int bw,
                                      Stats *s)
{
    T *d_open = (T *)moveToGPU((u8 *)h_value_share, elems * sizeof(T), s);
    gpuLinearComb(bw, asInt(elems), d_open, T(1), d_open, T(1), d_mask_share);
    peer->reconstructInPlace(d_open, bw, elems, s);
    return d_open;
}

static T *openMaskedInputFromGpuShare(GpuPeer *peer, T *d_value_share,
                                      T *d_mask_share, size_t elems, int bw,
                                      Stats *s)
{
    gpuLinearComb(bw, asInt(elems), d_value_share, T(1), d_value_share, T(1), d_mask_share);
    peer->reconstructInPlace(d_value_share, bw, elems, s);
    return d_value_share;
}

// Convert a public masked FSS output (Y + mask_Y) into additive output shares.
// Party 0 carries the public term and both parties subtract only their own
// mask share, so the two resulting shares sum to Y.
static T *publicMaskedOutputToShare(T *d_public_masked, T *d_mask_share,
                                    int party, size_t elems, int bw)
{
    T public_coeff = party == SERVER0 ? T(1) : T(0);
    gpuLinearComb(bw, asInt(elems), d_public_masked,
                  public_coeff, d_public_masked, T(-1), d_mask_share);
    return d_public_masked;
}

static void addBiasShareInPlace(T *d_value_share, const T *h_bias_share,
                                int rows, int cols, int bw, Stats *s)
{
    T *h_expanded = expandBias(h_bias_share, rows, cols);
    T *d_bias = (T *)moveToGPU((u8 *)h_expanded, (size_t)rows * cols * sizeof(T), s);
    gpuLinearComb(bw, rows * cols, d_value_share, T(1), d_value_share, T(1), d_bias);
    gpuFree(d_bias);
    delete[] h_expanded;
}

static T *reconstructShareOnCpu(GpuPeer *peer, const T *h_share, size_t elems, int bw)
{
    T *d = (T *)moveToGPU((u8 *)h_share, elems * sizeof(T), nullptr);
    peer->reconstructInPlace(d, bw, elems, nullptr);
    T *h_clear = (T *)moveToCPU((u8 *)d, elems * sizeof(T), nullptr);
    gpuFree(d);
    return h_clear;
}

static T *fssSoftmaxShareBC(GpuPeer *peer, int party, AESGlobalContext *gAES,
                            u8 **curPtrRef, u8 *startPtr,
                            const T *h_logits_share, int B, int C, int bw, int scale,
                            T *d_nExpMsbTab, T *d_nExpLsbTab, T *d_invTab,
                            Stats *stats)
{
    if (C <= 1)
    {
        T *h_O = new T[(size_t)B * C];
        T one_fp = (T)1 << scale;
        for (size_t i = 0; i < (size_t)B * C; ++i)
            h_O[i] = party == SERVER0 ? one_fp : T(0);
        return h_O;
    }

    const int C_pad = paddedClassCount(C);
    const int sm_bw = 50;
    MaxpoolParams p;
    p.bw = sm_bw;
    p.bin = 38;
    p.scale = scale;
    p.scaleDiv = 0;
    p.bwBackprop = 0;
    p.N = B;
    p.imgH = 1;
    p.imgW = C_pad;
    p.C = 1;
    p.FH = 1;
    p.FW = C_pad;
    p.strideH = 1;
    p.strideW = C_pad;
    p.zPadHLeft = 0;
    p.zPadHRight = 0;
    p.zPadWLeft = 0;
    p.zPadWRight = 0;
    p.H = 1;
    p.W = 1;
    p.isLowerTriangular = false;

    const int sz_pad = B * C_pad;
    const T sentinel = -(T(1) << (p.bin - 3));
    const u64 sm_mask = (sm_bw >= 64) ? ~T(0) : ((T(1) << sm_bw) - 1);
    const T sm_mod = (T(1) << sm_bw);

    auto d_mask_I = randomGEOnGpu<T>(sz_pad, sm_bw);
    auto d_mask_Is = splitMaskForParty(d_mask_I, party, sz_pad, sm_bw);

    auto d_mask_O = gpuKeygenSoftmax(curPtrRef, party, p, d_mask_I, gAES);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, (size_t)sz_pad * sizeof(T), nullptr);

    u8 *readPtr = startPtr;
    auto k = readGPUSoftMaxKey<T>(p, &readPtr);

    T *h_padded_share = new T[sz_pad];
    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
            h_padded_share[(size_t)b * C_pad + c] =
                h_logits_share[(size_t)b * C + c] & sm_mask;
        for (int c = C; c < C_pad; ++c)
            h_padded_share[(size_t)b * C_pad + c] =
                (party == SERVER0 ? sentinel : T(0)) & sm_mask;
    }

    auto d_masked = openMaskedInputFromCpuShare(peer, h_padded_share,
                                                d_mask_Is, sz_pad, sm_bw, stats);
    gpuFree(d_mask_Is);

    auto d_O = gpuSoftmax(peer, party, p, k, d_masked,
                          d_nExpMsbTab, d_nExpLsbTab, d_invTab, gAES, stats);
    auto h_O_masked = (T *)moveToCPU((u8 *)d_O, (size_t)sz_pad * sizeof(T), stats);
    gpuFree(d_masked);
    gpuFree(d_O);

    T *h_O_share = new T[(size_t)B * C];
    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
        {
            size_t in = (size_t)b * C_pad + c;
            T masked = h_O_masked[in] & sm_mask;
            T mask = h_mask_O[in] & sm_mask;
            h_O_share[(size_t)b * C + c] =
                (party == SERVER0) ? masked : (T((masked < mask) ? sm_mod : 0) - mask);
        }
    }

    delete[] h_padded_share;
    cpuFree(h_mask_O, true);
    cpuFree(h_O_masked, true);
    gpuFree(d_mask_I);
    gpuFree(d_mask_O);
    return h_O_share;
}

// =========================================================================
// One shard
// =========================================================================
struct LayerSizes
{
    MatmulParams p11, p12, p21, p22;
};
static LayerSizes makeLS(int Ns, int F, int H, int C, int bw, int scale)
{
    LayerSizes ls;
    auto fill = [&](MatmulParams &p, int M, int K, int N)
    {
        p.batchSz = 1;
        p.M = M;
        p.K = K;
        p.N = N;
        stdInit(p, bw, scale);
    };
    fill(ls.p11, Ns, F, H);
    fill(ls.p12, Ns, Ns, H);
    fill(ls.p21, Ns, H, C);
    fill(ls.p22, Ns, Ns, C);
    return ls;
}

static u64 runShard(int shard, int party, GpuPeer *peer, AESGlobalContext *gAES,
                    const Meta &gm, int hidden_dim,
                    const std::string &shardPrefix, const std::string &outPath,
                    u8 *keyBufStart, size_t keyBufSize,
                    T *d_nExpMsbTab, T *d_nExpLsbTab, T *d_invTab)
{
    Meta sm;
    loadMeta(shardPrefix + "_meta.txt", &sm);
    const int Ns = sm.Ns, F = sm.F, C = gm.C, H = hidden_dim;
    const int bw = 64, scale = gm.scale;
    const TruncateType tr = TruncateType::TrWithSlack;
    LayerSizes ls = makeLS(Ns, F, H, C, bw, scale);

    printf("[shard %d] Ns=%d F=%d H=%d C=%d  bw=%d scale=%d\n",
           shard, Ns, F, H, C, bw, scale);
    fflush(stdout);

    // ----- only this party's private input/model shares are read from disk -----
    T *h_X_share = readBin(sharePath(shardPrefix, "feat", party), (size_t)Ns * F);
    T *h_A_share = readBin(sharePath(shardPrefix, "adj", party), (size_t)Ns * Ns);
    const std::string wd = DATA_ROOT + "/weights/shard_" + std::to_string(shard);
    T *h_W1_share = readBin(sharePath(wd, "W1", party), (size_t)F * H);
    T *h_b1_share = readBin(sharePath(wd, "b1", party), H);
    T *h_W2_share = readBin(sharePath(wd, "W2", party), (size_t)H * C);
    T *h_b2_share = readBin(sharePath(wd, "b2", party), C);
    if (shard == 0)
        printf("[shard 0] model/input shares loaded for party %d\n", party);

    // ===== OFFLINE: full masks, per-party mask shares, and FSS keygen =====
    u8 *startPtr = keyBufStart;
    u8 *curPtr = keyBufStart;
    (void)keyBufSize;

    auto d_mask_X = randomGEOnGpu<T>((size_t)Ns * F, bw);
    auto d_mask_Xs = splitMaskForParty(d_mask_X, party, (size_t)Ns * F, bw);
    auto d_mask_A = randomGEOnGpu<T>((size_t)Ns * Ns, bw);
    auto d_mask_As = splitMaskForParty(d_mask_A, party, (size_t)Ns * Ns, bw);
    auto d_mask_W1 = randomGEOnGpu<T>((size_t)F * H, bw);
    auto d_mask_W1s = splitMaskForParty(d_mask_W1, party, (size_t)F * H, bw);
    auto d_mask_W2 = randomGEOnGpu<T>((size_t)H * C, bw);
    auto d_mask_W2s = splitMaskForParty(d_mask_W2, party, (size_t)H * C, bw);

    // 1.1 T1 = X · W1
    auto d_mask_T1 = gpuKeygenMatmul<T>(&curPtr, party, ls.p11,
                                        d_mask_X, d_mask_W1, (T *)nullptr,
                                        tr, gAES, true);
    auto d_mask_T1s = splitMaskForParty(d_mask_T1, party, (size_t)Ns * H, bw);

    // 1.2 U1 = A · T1
    auto h_mask_T1 = (T *)moveToCPU((u8 *)d_mask_T1, (size_t)Ns * H * sizeof(T), nullptr);
    auto d_mask_U1 = gpuKeygenMatmul<T>(&curPtr, party, ls.p12,
                                        d_mask_A, h_mask_T1, (T *)nullptr, tr, gAES);
    auto d_mask_U1s = splitMaskForParty(d_mask_U1, party, (size_t)Ns * H, bw);
    cpuFree(h_mask_T1);
    gpuFree(d_mask_T1);

    // ReLU on U1 + b1.  The bias is added to the value shares, so the input
    // mask for ReLU is still d_mask_U1.
    auto d_mask_H1 = gpuGenReluKey<T, T, RELU_P, RELU_Q, false>(
        &curPtr, party, bw, bw, ls.p12.size_C, d_mask_U1, gAES);
    auto d_mask_H1s = splitMaskForParty(d_mask_H1, party, (size_t)Ns * H, bw);
    gpuFree(d_mask_U1);

    // 2.1 T2 = H1 · W2
    auto d_mask_T2 = gpuKeygenMatmul<T>(&curPtr, party, ls.p21,
                                        d_mask_H1, d_mask_W2, (T *)nullptr,
                                        tr, gAES, true);
    auto d_mask_T2s = splitMaskForParty(d_mask_T2, party, (size_t)Ns * C, bw);
    gpuFree(d_mask_H1);

    // 2.2 Z = A · T2
    auto h_mask_T2 = (T *)moveToCPU((u8 *)d_mask_T2, (size_t)Ns * C * sizeof(T), nullptr);
    auto d_mask_Z = gpuKeygenMatmul<T>(&curPtr, party, ls.p22,
                                       d_mask_A, h_mask_T2, (T *)nullptr, tr, gAES);
    auto d_mask_Zs = splitMaskForParty(d_mask_Z, party, (size_t)Ns * C, bw);
    cpuFree(h_mask_T2);
    gpuFree(d_mask_T2);

    gpuFree(d_mask_X);
    gpuFree(d_mask_A);
    gpuFree(d_mask_W1);
    gpuFree(d_mask_W2);
    gpuFree(d_mask_Z);

    printf("[shard %d] key size = %lu MB\n", shard, (curPtr - startPtr) / (1024 * 1024));
    fflush(stdout);

    // deserialize keys for online
    auto k11 = readGPUMatmulKey<T>(ls.p11, tr, &startPtr);
    auto k12 = readGPUMatmulKey<T>(ls.p12, tr, &startPtr);
    auto krelu = readReluKey<T>(&startPtr);
    auto k21 = readGPUMatmulKey<T>(ls.p21, tr, &startPtr);
    auto k22 = readGPUMatmulKey<T>(ls.p22, tr, &startPtr);

    // ===== ONLINE =====
    Stats s;
    memset(&s, 0, sizeof(s));

    peer->sync();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Initial private shares -> public masked inputs consumed by the FSS API.
    auto d_masked_X = openMaskedInputFromCpuShare(peer, h_X_share, d_mask_Xs, (size_t)Ns * F, bw, &s);
    auto d_masked_A = openMaskedInputFromCpuShare(peer, h_A_share, d_mask_As, (size_t)Ns * Ns, bw, &s);
    auto d_masked_W1 = openMaskedInputFromCpuShare(peer, h_W1_share, d_mask_W1s, (size_t)F * H, bw, &s);
    auto d_masked_W2 = openMaskedInputFromCpuShare(peer, h_W2_share, d_mask_W2s, (size_t)H * C, bw, &s);
    gpuFree(d_mask_Xs);
    gpuFree(d_mask_As);
    gpuFree(d_mask_W1s);
    gpuFree(d_mask_W2s);

    auto d_T1 = gpuMatmul<T>(peer, party, ls.p11, k11,
                             d_masked_X, d_masked_W1, (T *)nullptr,
                             tr, gAES, &s, true);
    gpuFree(d_masked_X);
    gpuFree(d_masked_W1);
    publicMaskedOutputToShare(d_T1, d_mask_T1s, party, (size_t)Ns * H, bw);

    auto d_masked_T1 = openMaskedInputFromGpuShare(peer, d_T1, d_mask_T1s,
                                                   (size_t)Ns * H, bw, &s);
    gpuFree(d_mask_T1s);
    auto d_U1 = gpuMatmul<T>(peer, party, ls.p12, k12,
                             d_masked_A, d_masked_T1, (T *)nullptr,
                             tr, gAES, &s, true);
    gpuFree(d_masked_T1);
    publicMaskedOutputToShare(d_U1, d_mask_U1s, party, (size_t)Ns * H, bw);
    addBiasShareInPlace(d_U1, h_b1_share, Ns, H, bw, &s);

    auto d_masked_U1 = openMaskedInputFromGpuShare(peer, d_U1, d_mask_U1s,
                                                   (size_t)Ns * H, bw, &s);
    gpuFree(d_mask_U1s);
    auto d_H1 = gpuRelu<T, T, RELU_P, RELU_Q, false>(
        peer, party, krelu, d_masked_U1, gAES, &s);
    gpuFree(d_masked_U1);
    publicMaskedOutputToShare(d_H1, d_mask_H1s, party, (size_t)Ns * H, bw);

    auto d_masked_H1 = openMaskedInputFromGpuShare(peer, d_H1, d_mask_H1s,
                                                   (size_t)Ns * H, bw, &s);
    gpuFree(d_mask_H1s);
    auto d_T2 = gpuMatmul<T>(peer, party, ls.p21, k21,
                             d_masked_H1, d_masked_W2, (T *)nullptr,
                             tr, gAES, &s, true);
    gpuFree(d_masked_H1);
    gpuFree(d_masked_W2);
    publicMaskedOutputToShare(d_T2, d_mask_T2s, party, (size_t)Ns * C, bw);

    auto d_masked_T2 = openMaskedInputFromGpuShare(peer, d_T2, d_mask_T2s,
                                                   (size_t)Ns * C, bw, &s);
    gpuFree(d_mask_T2s);
    auto d_Z = gpuMatmul<T>(peer, party, ls.p22, k22,
                            d_masked_A, d_masked_T2, (T *)nullptr,
                            tr, gAES, &s, true);
    gpuFree(d_masked_A);
    gpuFree(d_masked_T2);
    publicMaskedOutputToShare(d_Z, d_mask_Zs, party, (size_t)Ns * C, bw);
    gpuFree(d_mask_Zs);
    addBiasShareInPlace(d_Z, h_b2_share, Ns, C, bw, &s);

    auto h_logits_share = (T *)moveToCPU((u8 *)d_Z, (size_t)Ns * C * sizeof(T), &s);
    gpuFree(d_Z);
    u8 *smStart = keyBufStart;
    u8 *smCur = keyBufStart;
    T *h_post_share = fssSoftmaxShareBC(peer, party, gAES, &smCur, smStart,
                                        h_logits_share, Ns, C, bw, scale,
                                        d_nExpMsbTab, d_nExpLsbTab, d_invTab,
                                        &s);

    auto t1 = std::chrono::high_resolution_clock::now();
    u64 comm = peer->bytesSent() + peer->bytesReceived();
    printf("[shard %d] online = %lu us, comm = %lu B\n", shard,
           std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(), comm);
    fflush(stdout);

    // ----- write party's additive posterior share -----
    writeBin(outPath + "_share" + std::to_string(party) + ".bin", h_post_share, (size_t)Ns * C);

    // ----- regression output: reconstruct posterior cleartext via peer -----
    auto h_post_clear = reconstructShareOnCpu(peer, h_post_share, (size_t)Ns * C, bw);
    writeBin(outPath + "_clear.bin", h_post_clear, (size_t)Ns * C);

    // CPU reference uses peer reconstruction of the already-loaded shares.
    // These clear values are not fed back into the FSS online path.
    T *h_X = reconstructShareOnCpu(peer, h_X_share, (size_t)Ns * F, bw);
    T *h_A = reconstructShareOnCpu(peer, h_A_share, (size_t)Ns * Ns, bw);
    T *h_W1 = reconstructShareOnCpu(peer, h_W1_share, (size_t)F * H, bw);
    T *h_b1 = reconstructShareOnCpu(peer, h_b1_share, H, bw);
    T *h_W2 = reconstructShareOnCpu(peer, h_W2_share, (size_t)H * C, bw);
    T *h_b2 = reconstructShareOnCpu(peer, h_b2_share, C, bw);

    // CPU reference: T1 = X·W1, U1 = A·T1 (+b1), H1 = ReLU(U1), T2 = H1·W2, softmax(A·T2 + b2)
    T *cpu_T1 = new T[(size_t)Ns * H];
    cpuMatmulTrunc(h_X, h_W1, cpu_T1, Ns, F, H, bw, scale);
    T *cpu_U1 = new T[(size_t)Ns * H];
    cpuMatmulTrunc(h_A, cpu_T1, cpu_U1, Ns, Ns, H, bw, scale);
    for (int i = 0; i < Ns; ++i)
        for (int j = 0; j < H; ++j)
        {
            cpu_U1[(size_t)i * H + j] += h_b1[j];
            cpuMod(cpu_U1[(size_t)i * H + j], bw);
        }
    cpuReluInPlace(cpu_U1, (size_t)Ns * H, bw);
    T *cpu_T2 = new T[(size_t)Ns * C];
    cpuMatmulTrunc(cpu_U1, h_W2, cpu_T2, Ns, H, C, bw, scale);
    T *cpu_Z = new T[(size_t)Ns * C];
    cpuMatmulTrunc(h_A, cpu_T2, cpu_Z, Ns, Ns, C, bw, scale);
    for (int i = 0; i < Ns; ++i)
        for (int j = 0; j < C; ++j)
        {
            cpu_Z[(size_t)i * C + j] += h_b2[j];
            cpuMod(cpu_Z[(size_t)i * C + j], bw);
        }
    T *cpu_post = cpuSoftmaxBC(cpu_Z, Ns, C, scale);

    int errs = 0;
    for (size_t i = 0; i < (size_t)Ns * C; ++i)
        if (h_post_clear[i] != cpu_post[i])
            ++errs;
    printf("[shard %d] FSS vs CPU mismatches = %d / %lu\n", shard, errs, (size_t)Ns * C);
    for (int i = 0; i < 5 && i < Ns * C; ++i)
        printf("  P[%d]  cpu=%10.4f  fss=%10.4f\n", i,
               asFloat(cpu_post[i], bw, scale),
               asFloat(h_post_clear[i], bw, scale));

    // cleanup
    cpuFree(h_logits_share);
    delete[] h_post_share;
    cpuFree(h_post_clear);
    cpuFree(h_X);
    cpuFree(h_A);
    cpuFree(h_W1);
    cpuFree(h_b1);
    cpuFree(h_W2);
    cpuFree(h_b2);
    delete[] h_X_share;
    delete[] h_A_share;
    delete[] h_W1_share;
    delete[] h_b1_share;
    delete[] h_W2_share;
    delete[] h_b2_share;
    delete[] cpu_T1;
    delete[] cpu_U1;
    delete[] cpu_T2;
    delete[] cpu_Z;
    delete[] cpu_post;
    return comm;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr,
                "Usage: %s <party> <peer_ip> <hidden_dim> [shard_id] "
                "[--port <int>] [--comm-buf-mb <int>]\n",
                argv[0]);
        return 1;
    }
    int party = atoi(argv[1]);
    const char *ip = argv[2];
    int hidden_dim = atoi(argv[3]);
    int single_shard = -1;
    int port = 42003;
    int comm_buf_mb = 5 * 1024;
    bool saw_shard = false;

    for (int i = 4; i < argc; ++i)
    {
        if (strcmp(argv[i], "--port") == 0)
        {
            if (++i >= argc || !parseIntArg(argv[i], &port) || port <= 0 || port > 65535)
            {
                fprintf(stderr, "ERROR: --port requires an integer in [1, 65535]\n");
                return 1;
            }
        }
        else if (strcmp(argv[i], "--comm-buf-mb") == 0)
        {
            if (++i >= argc || !parseIntArg(argv[i], &comm_buf_mb) || comm_buf_mb <= 0)
            {
                fprintf(stderr, "ERROR: --comm-buf-mb requires a positive integer\n");
                return 1;
            }
        }
        else if (strncmp(argv[i], "--", 2) == 0)
        {
            fprintf(stderr, "ERROR: unknown option: %s\n", argv[i]);
            return 1;
        }
        else if (!saw_shard)
        {
            if (!parseIntArg(argv[i], &single_shard))
            {
                fprintf(stderr, "ERROR: shard_id must be an integer\n");
                return 1;
            }
            saw_shard = true;
        }
        else
        {
            fprintf(stderr, "ERROR: unexpected argument: %s\n", argv[i]);
            return 1;
        }
    }
    assert((party == SERVER0 || party == SERVER1) && "party must be 0 or 1");
    const size_t comm_buf_bytes = (size_t)comm_buf_mb * 1024 * 1024;

    initGPUMemPool();
    // initGPUMemPool primes the pool with a 2 GB alloc-then-free and sets an
    // infinite release threshold.  Trim it back so L1 has room on smaller GPUs.
    {
        uint64_t threshold = 0;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(mempool, 0);
    }

    AESGlobalContext g;
    initAESContext(&g);

    auto peer = new GpuPeer(true);
    peer->connect(party, ip, port);
    printf("[comm] port=%d comm_buf=%d MB\n", port, comm_buf_mb);
    fflush(stdout);

    initGPURandomness();

    Meta gm;
    loadMeta(DATA_ROOT + "/meta.txt", &gm);
    printf("Global: N=%d F=%d C=%d k=%d scale=%d\n",
           gm.N, gm.F, gm.C, gm.k, gm.scale);
    fflush(stdout);
    if (gm.scale != 12)
    {
        fprintf(stderr,
                "ERROR: L1 FSS softmax requires scale=12 (meta.txt has scale=%d).\n",
                gm.scale);
        return 1;
    }

    ensureDir(DATA_ROOT + "/posteriors");

    const size_t KEY_BUF = (size_t)2 * 1024 * 1024 * 1024;
    u8 *keyBufStart = nullptr;
    u8 *keyBufCur = nullptr;
    getKeyBuf(&keyBufStart, &keyBufCur, KEY_BUF);
    (void)keyBufCur;

    const int C_pad = paddedClassCount(gm.C);
    const int inv_bin = std::max(8, int(std::ceil(std::log2((double)C_pad))) + gm.scale);
    auto d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, gm.scale);
    auto d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, gm.scale);
    auto d_invTab = genLUT<T, inv<T>>(inv_bin, 6, gm.scale);
    printf("[lut] C_pad=%d nExp(8,4,%d) + nExp(8,12,%d) + inv(%d,6,%d) ready\n",
           C_pad, gm.scale, gm.scale, inv_bin, gm.scale);
    fflush(stdout);

    u64 totComm = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    int lo = single_shard >= 0 ? single_shard : 0;
    int hi = single_shard >= 0 ? single_shard + 1 : gm.k;
    for (int s = lo; s < hi; ++s)
    {
        std::string sp = DATA_ROOT + "/shards/shard_" + std::to_string(s);
        std::string op = DATA_ROOT + "/posteriors/shard_" + std::to_string(s) + "_post";
        totComm += runShard(s, party, peer, &g, gm, hidden_dim, sp, op,
                            keyBufStart, KEY_BUF,
                            d_nExpMsbTab, d_nExpLsbTab, d_invTab);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\n=== L1 done. wall=%ld ms, total comm=%lu MB ===\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
           totComm / (1024 * 1024));

    destroyGPURandomness();
    // Skip `delete peer` - GpuPeer dtor has a joinable thread issue in this
    // process layout. Process exit cleans up sockets/threads.
    fflush(stdout);
    _exit(0);
}
