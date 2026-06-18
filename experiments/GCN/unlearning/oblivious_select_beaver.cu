// oblivious_select_beaver.cu — BEAVER-MATMUL VARIANT of the oblivious select (for ABLATION).
//
// User's design: DPF per-node one-hot -> per-shard SUM -> secret-shared shard-level
// one-hot (which shard X is in stays HIDDEN) -> one-hot x per-shard adjacency = an
// oblivious GATHER (a Beaver matmul). NOTHING about X or the routed shard is revealed.
//
//   isX[k,j]      = 1{nodeid_k[j] == X}   via DPF-LUT(eqTab, nodeid - X)  (secret share)
//   shard_oh[k]   = sum_j isX[k,j]        (LOCAL sum of shares; one-hot over shards)
//   A_sel[Ns^2]   = shard_oh[1xK] . A_stacked[K x Ns^2]   (Beaver matmul; gather)
//   isX_sel[j]    = sum_k isX[k,j]        (LOCAL; X's row inside the selected shard)
//   keep[j]       = 1 - isX_sel[j]
//
// ORCA-STYLE DEALER / EVALUATOR SEPARATION (no interleaving):
//   PHASE 1 (DEALER): generate EVERY key (DPF-LUT + all gather matmuls) into one key
//     buffer. Keys depend only on random masks, never on data.
//   PHASE 2 (EVALUATOR): first READ all the keys, THEN execute (open operands +
//     reconstruct). No "read-a-key-then-run" loop.
//
// MASK DISCIPLINE (library pattern: gpu_gelu.cu / gcn_agg_layer.h; the previous
// student's splitMaskForParty/openMasked re-drew randomGEOnGpu after keygen -> a
// share inconsistent with the one keygen wrote -> garbage; removed):
//   * keygen writeShares() the mask -> the share lives in the key buffer.
//   * eval moveToGPU() that share, add to the value share, reconstructInPlace() to open.
//   * gpuDpfLUT(opMasked=true) -> masked-public one-hot; subtract the writeShares'd
//     [maskOut] -> share. LBW=64 so the per-shard SUM of one-hot shares is exact mod
//     2^64 (a 16-bit LBW leaks 2^16 carries from zero entries into shard_oh).
//   * gpuMatmulPlaintext(keygen)+gpuMatmulBeaver(eval) exactly as AggLayer does.
//
// Data root = cora_shards_canon (canonical sorted(comm[s] U test) order; X=270 -> shard
// 0 row 74). Verify: reconstruct(A_sel) == canonical shard(X) adjacency; keep[74]=0.
//
// Usage: ./oblivious_select <party> <peer_ip> [--port N] [--reveal-asel]
//   <root>/select/qnode_share<p>.bin must hold this party's share of X.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_dpf.h"
#include "fss/gpu_lut.h"
#include "fss/gpu_matmul.h"
#include "fss/gpu_mul.h"
#include "fss/gpu_select.h"

#include <cuda_runtime.h>
extern cudaMemPool_t mempool;   // global from utils/gpu_mem.cu (matmul cutlass alloc path)
using T = u64;
extern unsigned long long g_commRounds;   // comm-round counter (utils/sigma_comms.cpp)

static std::string ROOT()
{
    const char *e = std::getenv("FSS_DATA_ROOT");
    return std::string(e && *e ? e : "datasets/cora_shards_canon");
}
static int parseMeta(const std::string &path, const char *key)
{
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "ERROR: missing %s\n", path.c_str()); std::abort(); }
    std::string ln, k = std::string(key) + "=";
    while (std::getline(f, ln))
        if (ln.rfind(k, 0) == 0) return std::atoi(ln.c_str() + k.size());
    fprintf(stderr, "ERROR: key %s not in %s\n", key, path.c_str()); std::abort();
}
template <typename U>
static U *readBinT(const std::string &path, size_t elems)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { fprintf(stderr, "ERROR: missing %s\n", path.c_str()); std::abort(); }
    size_t bytes = (size_t)f.tellg();
    if (bytes != elems * sizeof(U))
    { fprintf(stderr, "ERROR: %s has %zu bytes want %zu\n", path.c_str(), bytes, elems * sizeof(U)); std::abort(); }
    f.seekg(0); U *buf = new U[elems]; f.read((char *)buf, bytes); return buf;
}
static void writeBinT(const std::string &path, const void *p, size_t bytes)
{ std::ofstream f(path, std::ios::binary); f.write((const char *)p, bytes); }

static MatmulParams mmF(int M, int K, int N, int bw, int scale)
{ MatmulParams p; p.batchSz = 1; p.M = M; p.K = K; p.N = N; stdInit(p, bw, scale); return p; }

// ----------------------------------------------------------------------------- //
// Oblivious GATHER  sel[1,W] = oh[1,K] . stack[K,W]  for a SECRET stack.
// Split dealer (key gen) and evaluator (read-all-then-run) per the Orca style.
// W is tiled by CS so each cutlass GEMM stays in CUDA grid bounds.
// ----------------------------------------------------------------------------- //
// One mask per WHOLE gather (NOT per tile), so the evaluator opens each operand
// with a SINGLE reconstruct. Tiling exists only for the cutlass grid limit and is
// pure LOCAL compute (gpuMatmulBeaver does no communication; the per-tile output
// is the share directly -- no output mask, no per-tile reconstruct).
struct GatherKeys {
    size_t W = 0, CS = 0;
    std::vector<size_t> cs;          // per-tile width
    std::vector<u8 *> mmStart;       // per-tile matmul key (A=[mask_oh], B=[mask_A_tile], C)
    std::vector<MatmulParams> p;     // per-tile params
    u8 *fullMaskA = nullptr;         // [K,W] mask_A share, written ONCE (open A_stack once)
};

// DEALER: ONE mask_A[K,W] (+ the shared mask_oh) -> per-tile beaver C = mask_oh.mask_A_tile.
// No output mask: gpuMatmulBeaver then yields the gather SHARE directly.
static void dealerGatherKeys(u8 **kCur, int party, int K, size_t W, size_t CS,
                             T *d_mask_oh, AESGlobalContext *gAES, GatherKeys &gk)
{
    gk.W = W; gk.CS = CS;
    auto d_mask_A = randomGEOnGpu<T>((size_t)K * W, 64);              // [K,W], one mask
    gk.fullMaskA = *kCur;
    writeShares<T, T>(kCur, party, (size_t)K * W, d_mask_A, 64);      // for the single open
    for (size_t c = 0; c < W; c += CS) {
        size_t cs = (W - c < CS) ? (W - c) : CS;
        MatmulParams p = mmF(1, K, (int)cs, 64, 0);                   // [1,K] x [K,cs] -> [1,cs]
        T *d_mAt = (T *)gpuMalloc((size_t)K * cs * sizeof(T));        // mask_A column-tile [K,cs]
        cudaMemcpy2D(d_mAt, cs * sizeof(T), d_mask_A + c, W * sizeof(T),
                     cs * sizeof(T), K, cudaMemcpyDeviceToDevice);
        auto d_C = gpuMatmulPlaintext(p, d_mask_oh, d_mAt, (T *)nullptr, false);  // mask_oh.mask_A_tile
        u8 *mmStart = *kCur;
        writeShares<T, T>(kCur, party, p.size_A, d_mask_oh, p.bw);    // k.A = [mask_oh]
        writeShares<T, T>(kCur, party, p.size_B, d_mAt, p.bw);        // k.B = [mask_A_tile]
        writeShares<T, T>(kCur, party, p.size_C, d_C, p.bw);          // k.C (no output mask)
        gk.cs.push_back(cs); gk.mmStart.push_back(mmStart); gk.p.push_back(p);
        gpuFree(d_mAt); gpuFree(d_C);
    }
    gpuFree(d_mask_A);
}

// EVALUATOR: open A_stack ONCE (single reconstruct), read all tile keys, then run the
// per-tile beaver matmul LOCALLY (no comm). d_oh_pub = oh + mask_oh, opened once by the
// caller (shared across all gathers). Output is the gather SHARE (no reconstruct).
static void evalGather(int party, int K, const std::vector<T> &h_stack,
                       T *d_oh_pub, GpuPeer *peer, const GatherKeys &gk, std::vector<T> &h_sel)
{
    const size_t W = gk.W, nC = gk.cs.size(), KW = (size_t)K * W;
    h_sel.assign(W, 0);
    // open A_stack once: A_pub = A_stack + mask_A  (the ONE communication of this gather)
    T *d_maskA = (T *)moveToGPU((u8 *)gk.fullMaskA, KW * sizeof(T), nullptr);
    T *d_A = (T *)moveToGPU((u8 *)h_stack.data(), KW * sizeof(T), nullptr);
    gpuLinearComb(64, (int)KW, d_A, T(1), d_A, T(1), d_maskA);
    peer->reconstructInPlace(d_A, 64, KW, nullptr);
    gpuFree(d_maskA);
    // read all tile keys, then execute locally
    std::vector<GPUMatmulKey<T>> mmK(nC);
    for (size_t i = 0; i < nC; ++i) { u8 *kR = gk.mmStart[i]; mmK[i] = readGPUMatmulKey<T>(gk.p[i], TruncateType::None, &kR); }
    size_t c = 0;
    for (size_t i = 0; i < nC; ++i) {
        const size_t cs = gk.cs[i]; const MatmulParams &p = gk.p[i];
        T *d_At = (T *)gpuMalloc((size_t)K * cs * sizeof(T));        // A_pub column-tile [K,cs]
        cudaMemcpy2D(d_At, cs * sizeof(T), d_A + c, W * sizeof(T),
                     cs * sizeof(T), K, cudaMemcpyDeviceToDevice);
        T *d_mOhS = (T *)moveToGPU((u8 *)mmK[i].A, p.size_A * sizeof(T), nullptr);
        T *d_mAtS = (T *)moveToGPU((u8 *)mmK[i].B, p.size_B * sizeof(T), nullptr);
        auto d_Z = gpuMatmulBeaver<T>(p, mmK[i], party, d_oh_pub, d_At, d_mOhS, d_mAtS, (T *)nullptr, nullptr);
        moveIntoCPUMem((u8 *)&h_sel[c], (u8 *)d_Z, cs * sizeof(T), nullptr);   // share, no reconstruct
        gpuFree(d_At); gpuFree(d_mOhS); gpuFree(d_mAtS); gpuFree(d_Z);
        c += cs;
    }
    gpuFree(d_A);
}

// ----------------------------------------------------------------------------- //
// Elementwise secret*secret multiply  out[N] = X .* Y  (Beaver, gpuMul), dealer/eval
// split. scale=0 / TruncateType::None: keep is 0/1 (unscaled) so A.*keep stays scale-12.
// Used to APPLY keep (X removal): A_masked = A_sel .* keep_col (zero X's adj column),
// train_mask_eff = train_mask_sel .* keep (drop X from the loss).
// ----------------------------------------------------------------------------- //
struct MulKeys { u8 *kStart = nullptr; u8 *mCShare = nullptr; size_t N = 0; };

static void dealerMulKeys(u8 **kCur, int party, size_t N, AESGlobalContext *gAES, MulKeys &mk)
{
    auto d_mA = randomGEOnGpu<T>(N, 64);
    auto d_mB = randomGEOnGpu<T>(N, 64);
    mk.kStart = *kCur;
    auto d_mC = gpuKeygenMul<T>(kCur, party, 64, 0, (int)N, d_mA, d_mB, TruncateType::None, gAES);
    mk.mCShare = *kCur;
    writeShares<T, T>(kCur, party, N, d_mC, 64);   // [mC] for masked-public -> share
    mk.N = N;
    gpuFree(d_mA); gpuFree(d_mB); gpuFree(d_mC);
}

static void evalMul(int party, const std::vector<T> &hX, const std::vector<T> &hY,
                    GpuPeer *peer, const MulKeys &mk, AESGlobalContext *gAES, std::vector<T> &hOut)
{
    const size_t N = mk.N; hOut.assign(N, 0);
    u8 *kR = mk.kStart;
    auto k = readGPUMulKey<T>(&kR, N, N, N, TruncateType::None);
    // open operands to masked-public using the triple's input-mask shares (k.a, k.b).
    T *d_mAs = (T *)moveToGPU((u8 *)k.a, N * sizeof(T), nullptr);
    T *d_X = (T *)moveToGPU((u8 *)hX.data(), N * sizeof(T), nullptr);
    gpuLinearComb(64, (int)N, d_X, T(1), d_X, T(1), d_mAs);
    peer->reconstructInPlace(d_X, 64, N, nullptr);
    gpuFree(d_mAs);
    T *d_mBs = (T *)moveToGPU((u8 *)k.b, N * sizeof(T), nullptr);
    T *d_Y = (T *)moveToGPU((u8 *)hY.data(), N * sizeof(T), nullptr);
    gpuLinearComb(64, (int)N, d_Y, T(1), d_Y, T(1), d_mBs);
    peer->reconstructInPlace(d_Y, 64, N, nullptr);
    gpuFree(d_mBs);
    auto d_Z = gpuMul<T>(peer, party, 64, 0, (int)N, k, d_X, d_Y, TruncateType::None, gAES, nullptr);
    // gpuMul returns masked-public (X.*Y + mC); subtract [mC] -> share.
    T *d_mCs = (T *)moveToGPU((u8 *)mk.mCShare, N * sizeof(T), nullptr);
    gpuLinearComb(64, (int)N, d_Z, (party == 0) ? T(1) : T(0), d_Z, T(-1), d_mCs);
    moveIntoCPUMem((u8 *)hOut.data(), (u8 *)d_Z, N * sizeof(T), nullptr);
    gpuFree(d_X); gpuFree(d_Y); gpuFree(d_Z); gpuFree(d_mCs);
}

int main(int argc, char *argv[])
{
    if (argc < 3) { fprintf(stderr, "Usage: %s <party> <ip> [--port N] [--reveal-asel]\n", argv[0]); return 1; }
    int party = atoi(argv[1]); const char *ip = argv[2];
    int port = 42004; bool reveal_asel = false;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reveal-asel")) reveal_asel = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (party != 0 && party != 1) { fprintf(stderr, "party 0/1\n"); return 1; }

    const std::string root = ROOT(), sh = root + "/shards", sd = root + "/select";
    const int K = parseMeta(root + "/meta.txt", "k");
    const int Ns = parseMeta(root + "/meta.txt", "Ns_max");
    const int F = parseMeta(root + "/meta.txt", "F");
    const int C = parseMeta(root + "/meta.txt", "C");
    const int bin = parseMeta(root + "/meta.txt", "bin");
    const int dom = 1 << bin;
    const size_t Q = (size_t)K * Ns;          // all shard rows
    const size_t N2 = (size_t)Ns * Ns;        // flattened adjacency per shard
    const size_t NF = (size_t)Ns * F;         // flattened features per shard
    const size_t NC = (size_t)Ns * C;         // flattened y_onehot per shard
    const int LBW = 64;                       // DPF-LUT output bitwidth (64: exact per-shard sum)
    const size_t CS = 4096;                   // gather N tile

    initGPUMemPool();
    {   // mirror grapheraser_fss_l1: trim the primed pool (matmul alloc path)
        uint64_t threshold = 0;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(mempool, 0);
    }
    AESGlobalContext gAES; initAESContext(&gAES);
    setenv("WING_PORT", std::to_string(port).c_str(), 1);
    auto peer = new GpuPeer(true);
    peer->connect(party, (char *)ip);
    initGPURandomness();

    // ---- load: public node-id tables + masks, secret-shared adj/feat/y_onehot, secret X ----
    std::vector<T> h_nodeids(Q);
    std::vector<T> h_Astack((size_t)K * N2, 0);   // this party's share of each shard adjacency
    std::vector<T> h_Fstack((size_t)K * NF, 0);   // ... features
    std::vector<T> h_Yohstack((size_t)K * NC, 0); // ... y_onehot
    std::vector<T> h_TMstack((size_t)K * Ns, 0);  // train_mask (PUBLIC, per shard)
    std::vector<T> h_TEstack((size_t)K * Ns, 0);  // test_mask  (PUBLIC, per shard)
    for (int k = 0; k < K; ++k) {
        const std::string pk = sh + "/shard_" + std::to_string(k);
        T *nid = readBinT<T>(pk + "_nodeids.bin", Ns);
        std::memcpy(&h_nodeids[(size_t)k * Ns], nid, Ns * sizeof(T)); delete[] nid;
        T *adj = readBinT<T>(pk + "_adj_share" + std::to_string(party) + ".bin", N2);
        std::memcpy(&h_Astack[(size_t)k * N2], adj, N2 * sizeof(T)); delete[] adj;
        T *ft = readBinT<T>(pk + "_feat_share" + std::to_string(party) + ".bin", NF);
        std::memcpy(&h_Fstack[(size_t)k * NF], ft, NF * sizeof(T)); delete[] ft;
        T *yo = readBinT<T>(pk + "_y_onehot_share" + std::to_string(party) + ".bin", NC);
        std::memcpy(&h_Yohstack[(size_t)k * NC], yo, NC * sizeof(T)); delete[] yo;
        T *tm = readBinT<T>(pk + "_train_mask.bin", Ns);
        std::memcpy(&h_TMstack[(size_t)k * Ns], tm, Ns * sizeof(T)); delete[] tm;
        T *te = readBinT<T>(pk + "_test_mask.bin", Ns);
        std::memcpy(&h_TEstack[(size_t)k * Ns], te, Ns * sizeof(T)); delete[] te;
    }
    T *h_q = readBinT<T>(sd + "/qnode_share" + std::to_string(party) + ".bin", 1);

    const size_t KEY_BUF = (size_t)8ULL * 1024 * 1024 * 1024;
    u8 *kStart = nullptr, *kCur = nullptr; getKeyBuf(&kStart, &kCur, KEY_BUF);

    // share of (nodeid_q - X) over all K*Ns rows.
    std::vector<T> h_diff(Q);
    T xs = h_q[0];
    for (size_t q = 0; q < Q; ++q)
        h_diff[q] = (party == 0) ? (h_nodeids[q] - xs) : (T(0) - xs);

    // =====================================================================
    // PHASE 1 — DEALER: generate every key (DPF-LUT for isX + adjacency gather).
    // =====================================================================
    auto d_rq = randomGEOnGpu<T>(Q, bin);                 // isX input mask r
    u8 *kRqShare = kCur;
    writeShares<T, T>(&kCur, party, Q, d_rq, bin);        // [r]
    u8 *kLutStart = kCur;
    auto d_maskOut = gpuKeyGenLUT<T, T>(&kCur, party, bin, LBW, (int)Q, d_rq, &gAES);
    u8 *kMaskOut = kCur;
    writeShares<T, T>(&kCur, party, Q, d_maskOut, LBW);   // [maskOut]
    gpuFree(d_maskOut); gpuFree(d_rq);
    auto d_mask_oh = randomGEOnGpu<T>(K, 64);              // shared shard-one-hot mask (open once)
    u8 *kOhMask = kCur;
    writeShares<T, T>(&kCur, party, K, d_mask_oh, 64);
    GatherKeys gkAdj, gkFeat, gkYoh;
    dealerGatherKeys(&kCur, party, K, N2, CS, d_mask_oh, &gAES, gkAdj);   // adjacency  [K, Ns*Ns]
    dealerGatherKeys(&kCur, party, K, NF, CS, d_mask_oh, &gAES, gkFeat);  // features   [K, Ns*F]
    dealerGatherKeys(&kCur, party, K, NC, CS, d_mask_oh, &gAES, gkYoh);   // y_onehot   [K, Ns*C]
    gpuFree(d_mask_oh);
    MulKeys mkAdjKeep, mkTMKeep;
    dealerMulKeys(&kCur, party, N2, &gAES, mkAdjKeep);         // A_sel .* keep_col
    dealerMulKeys(&kCur, party, Ns, &gAES, mkTMKeep);          // train_mask_sel .* keep
    fprintf(stderr, "[P%d] dealer(BEAVER): keys done (LUT isX + open-once beaver gather + 2 keep-muls)\n",
            party); fflush(stderr);

    // =====================================================================
    // PHASE 2 — EVALUATOR: read all keys, then execute.
    // =====================================================================
    // --- read phase ---
    u8 *kLutRead = kLutStart; auto lutKey = readGPULUTKey<T>(&kLutRead);
    T *d_rqs = (T *)moveToGPU((u8 *)kRqShare, Q * sizeof(T), nullptr);          // [r]
    T *d_moS = (T *)moveToGPU((u8 *)kMaskOut, Q * sizeof(T), nullptr);          // [maskOut]

    auto commBytes = [&]() { return peer->bytesSent() + peer->bytesReceived(); };
    u64 cb0 = commBytes();
    unsigned long long r0 = g_commRounds; auto t0 = std::chrono::high_resolution_clock::now();

    // --- isX: ARITHMETIC one-hot via DPF-LUT (LBW=64 so the per-shard SUM is exact mod 2^64) ---
    std::vector<T> h_eqTab(dom, 0); h_eqTab[0] = 1;
    T *d_eqTab = (T *)moveToGPU((u8 *)h_eqTab.data(), (size_t)dom * sizeof(T), nullptr);
    T *d_X = (T *)moveToGPU((u8 *)h_diff.data(), Q * sizeof(T), nullptr);
    gpuLinearComb(bin, (int)Q, d_X, T(1), d_X, T(1), d_rqs);
    peer->reconstructInPlace(d_X, bin, Q, nullptr);                            // (nodeid-X)+r masked-public
    auto d_isX = gpuDpfLUT<T, T>(lutKey, peer, party, d_X, d_eqTab, &gAES, nullptr, true);
    gpuLinearComb(LBW, (int)Q, d_isX, (party == 0) ? T(1) : T(0), d_isX, T(-1), d_moS); // -> [isX]
    T *h_isX = (T *)moveToCPU((u8 *)d_isX, Q * sizeof(T), nullptr);
    gpuFree(d_X); gpuFree(d_isX); gpuFree(d_rqs); gpuFree(d_moS); gpuFree(d_eqTab);
    fprintf(stderr, "[P%d] eval(BEAVER): isX done\n", party); fflush(stderr);

    // shard_oh[k] = sum_j isX ; isX_sel[j] = sum_k isX ; keep = 1 - isX_sel  (arithmetic)
    std::vector<T> h_oh(K, 0), h_isXsel(Ns, 0), h_keep(Ns);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < Ns; ++j) { T v = h_isX[(size_t)k * Ns + j]; h_oh[k] += v; h_isXsel[j] += v; }
    for (int j = 0; j < Ns; ++j) h_keep[j] = ((party == 0) ? T(1) : T(0)) - h_isXsel[j];

    // ============================ BEAVER GATHER (open-once matmul) ============================
    // open shard one-hot ONCE (shared across gathers)
    T *d_oh_pub = (T *)moveToGPU((u8 *)h_oh.data(), K * sizeof(T), nullptr);
    T *d_mOhg = (T *)moveToGPU((u8 *)kOhMask, K * sizeof(T), nullptr);
    gpuLinearComb(64, K, d_oh_pub, T(1), d_oh_pub, T(1), d_mOhg);
    peer->reconstructInPlace(d_oh_pub, 64, K, nullptr);
    gpuFree(d_mOhg);
    // each gather = ONE reconstruct (open the stacked operand); gpuMatmulBeaver is local
    std::vector<T> h_Asel, h_Fsel, h_Yohsel;
    evalGather(party, K, h_Astack, d_oh_pub, peer, gkAdj, h_Asel);     // A_sel  [Ns*Ns]
    evalGather(party, K, h_Fstack, d_oh_pub, peer, gkFeat, h_Fsel);    // feat   [Ns*F]
    evalGather(party, K, h_Yohstack, d_oh_pub, peer, gkYoh, h_Yohsel); // y_onehot [Ns*C]
    gpuFree(d_oh_pub);
    // masks PUBLIC per shard -> LOCAL secret-oh x public-mask gather
    std::vector<T> h_TMsel(Ns, 0), h_TEsel(Ns, 0);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < Ns; ++j) {
            h_TMsel[j] += h_oh[k] * h_TMstack[(size_t)k * Ns + j];
            h_TEsel[j] += h_oh[k] * h_TEstack[(size_t)k * Ns + j];
        }
    u64 cb_gather = commBytes();
    fprintf(stderr, "[P%d] eval(BEAVER): gathers done (adj/feat/yoh/masks)\n", party); fflush(stderr);

    // ============================ keep (X removal) via gpuMul ============================
    std::vector<T> h_keepcol(N2);
    for (int i = 0; i < Ns; ++i)
        for (int j = 0; j < Ns; ++j) h_keepcol[(size_t)i * Ns + j] = h_keep[j];
    std::vector<T> h_Amask, h_TMeff;
    evalMul(party, h_Asel, h_keepcol, peer, mkAdjKeep, &gAES, h_Amask);   // A_sel .* keep_col
    evalMul(party, h_TMsel, h_keep, peer, mkTMKeep, &gAES, h_TMeff);      // train_mask_sel .* keep
    u64 cb_end = commBytes();
    fprintf(stderr, "[P%d] eval(BEAVER): keep applied (adj col zeroed + train_mask drops X)\n", party); fflush(stderr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (party == 0)
        fprintf(stderr, "[RESULT beaver] bytes: isX+gather=%.2f MB keep=%.2f MB TOTAL=%.2f MB | rounds=%llu | online_time=%.1f ms\n",
                (cb_gather - cb0) / 1e6, (cb_end - cb_gather) / 1e6, (cb_end - cb0) / 1e6, g_commRounds - r0, ms);

    // =====================================================================
    // outputs (secret shares of the X-removed selected subgraph; shard hidden)
    // =====================================================================
    makeDir(sd);
    writeBinT(sd + "/Asel_share" + std::to_string(party) + ".bin", h_Amask.data(), N2 * sizeof(T)); // X-removed adj
    writeBinT(sd + "/featsel_share" + std::to_string(party) + ".bin", h_Fsel.data(), NF * sizeof(T));
    writeBinT(sd + "/yohsel_share" + std::to_string(party) + ".bin", h_Yohsel.data(), NC * sizeof(T));
    writeBinT(sd + "/trainmask_share" + std::to_string(party) + ".bin", h_TMeff.data(), Ns * sizeof(T)); // X dropped
    writeBinT(sd + "/testmask_share" + std::to_string(party) + ".bin", h_TEsel.data(), Ns * sizeof(T));
    writeBinT(sd + "/keep_share" + std::to_string(party) + ".bin", h_keep.data(), Ns * sizeof(T));
    writeBinT(sd + "/isxsel_share" + std::to_string(party) + ".bin", h_isXsel.data(), Ns * sizeof(T));

    if (party == 0) {
        printf("[obliv-select] isX over %zu rows (K=%d Ns=%d) via DPF-LUT (arithmetic), X never revealed\n", Q, K, Ns);
        printf("[obliv-select] gather = open-once BEAVER matmul (oh . stacked); keep gpuMul\n");
        printf("[obliv-select] wrote %s/{Asel_share*,keep_share*,isxsel_share*}.bin\n", sd.c_str());
    }
    // optional: reveal the gathered subgraph for the numpy oracle (obliv_verify.py).
    // (debug only; the real pipeline keeps everything secret-shared.)
    if (reveal_asel) {
        auto reveal = [&](std::vector<T> &h, size_t n, const std::string &name) {
            auto d = (T *)moveToGPU((u8 *)h.data(), n * sizeof(T), nullptr);
            peer->reconstructInPlace(d, 64, n, nullptr);
            T *hc = (T *)moveToCPU((u8 *)d, n * sizeof(T), nullptr);
            if (party == 0) writeBinT(sd + "/" + name + "_clear.bin", hc, n * sizeof(T));
            gpuFree(d);
        };
        reveal(h_Amask, N2, "Amask");      // keep-applied adjacency (X column zeroed)
        reveal(h_Fsel, NF, "featsel");
        reveal(h_Yohsel, NC, "yohsel");
        reveal(h_TMeff, Ns, "tmeff");      // keep-applied train_mask (X dropped)
        reveal(h_TEsel, Ns, "testmask");
        reveal(h_keep, Ns, "keep");
        if (party == 0) printf("[obliv-select] (reveal) wrote {Amask,featsel,yohsel,tmeff,testmask,keep}_clear.bin\n");
    }
    peer->sync();
    return 0;
}
