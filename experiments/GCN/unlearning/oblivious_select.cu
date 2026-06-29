// oblivious_select.cu — NO-REVEAL privacy-preserving SELECT for single-node unlearning.
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


// (the keep step now uses gpuSelect directly — no Beaver elementwise multiply needed.)

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
    const size_t Wtot = N2 + NF + NC + 2 * Ns;            // concat [adj|feat|y_onehot|train_mask|test_mask]
    const size_t Ksel = (size_t)K * Wtot;                 // select over all K shards' cat
    auto d_rq = randomGEOnGpu<T>(Q, bin);                 // isX input mask r (full; synced RNG)
    u8 *kRqShare = kCur;
    writeShares<T, T>(&kCur, party, Q, d_rq, bin);        // [r]
    u8 *kDpfStart = kCur;
    gpuKeyGenDPF<T>(&kCur, party, bin, (int)Q, d_rq, &gAES, /*evalAll=*/false); // 1-bit equality DPF
    gpuFree(d_rq);

    // ONE gpuSelect over the concatenated cat_stack: maskB = selector (bit) mask,
    // maskX = value mask. gpuKeyGenSelect writes k.a=[maskB], k.b=[maskX], k.c/d1/d2.
    auto d_maskX = randomGEOnGpu<T>(Ksel, 64);
    // selector mask is PER-SHARD (broadcast across the Wtot columns of a shard) so the
    // evaluator opens only K selector bits, not K*Wtot -- the one-hot is the same for all
    // w of a shard, and the mask is reused. (gpuKeyGenSelect still wants a [Ksel] maskB.)
    auto d_maskBk = randomGEOnGpu<T>(K, 1);
    T *h_mbk = (T *)moveToCPU((u8 *)d_maskBk, K * sizeof(T), nullptr);
    std::vector<T> h_maskB(Ksel);
    for (int k = 0; k < K; ++k)
        for (size_t w = 0; w < Wtot; ++w) h_maskB[(size_t)k * Wtot + w] = h_mbk[k];
    auto d_maskB = (T *)moveToGPU((u8 *)h_maskB.data(), Ksel * sizeof(T), nullptr);
    u8 *kSelStart = kCur;
    gpuKeyGenSelect<T, T>(&kCur, party, (int)Ksel, d_maskX, d_maskB, 64, /*opMasked=*/false);
    gpuFree(d_maskX); gpuFree(d_maskBk); gpuFree(d_maskB);

    // keep (X removal) via DIRECT gpuSelect (keep_bit ? value : 0), NOT a Beaver multiply:
    // a select opens one value + a 1-bit selector; the multiply opened BOTH operands. A over N^2.
    auto d_mXadj = randomGEOnGpu<T>(N2, 64); auto d_mBadj = randomGEOnGpu<T>(N2, 1);
    u8 *kAdjKeepS = kCur; gpuKeyGenSelect<T, T>(&kCur, party, (int)N2, d_mXadj, d_mBadj, 64, /*opMasked=*/false);
    gpuFree(d_mXadj); gpuFree(d_mBadj);
    auto d_mXtm = randomGEOnGpu<T>(Ns, 64); auto d_mBtm = randomGEOnGpu<T>(Ns, 1);
    u8 *kTMKeepS = kCur; gpuKeyGenSelect<T, T>(&kCur, party, Ns, d_mXtm, d_mBtm, 64, /*opMasked=*/false);
    gpuFree(d_mXtm); gpuFree(d_mBtm);
    fprintf(stderr, "[P%d] dealer: keys done (eq-DPF + gather select Wtot=%zu + 2 keep selects)\n",
            party, Wtot); fflush(stderr);

    // =====================================================================
    // PHASE 2 — EVALUATOR: read all keys, then execute.
    // =====================================================================
    // --- read phase ---
    u8 *kd = kDpfStart; auto dpfKey = readGPUDPFKey(&kd);                       // equality DPF
    u8 *ks = kSelStart; auto selKey = readGPUSelectKey<T>(&ks, (int)Ksel);      // gather select; k.a=[maskB], k.b=[maskX]
    u8 *kak = kAdjKeepS; auto adjKeepKey = readGPUSelectKey<T>(&kak, (int)N2);  // keep-col select on A_sel
    u8 *ktk = kTMKeepS;  auto tmKeepKey  = readGPUSelectKey<T>(&ktk, Ns);       // keep select on train_mask
    T *d_rqs = (T *)moveToGPU((u8 *)kRqShare, Q * sizeof(T), nullptr);          // [r]

    auto commBytes = [&]() { return peer->bytesSent() + peer->bytesReceived(); };
    u64 cb0 = commBytes();
    unsigned long long r0 = g_commRounds; auto t0 = std::chrono::high_resolution_clock::now();
    auto getbit = [](const u32 *p, size_t i) { return (p[i >> 5] >> (i & 31)) & 1u; };

    // --- isX: 1-bit boolean equality (gpuDpf) ---
    T *d_X = (T *)moveToGPU((u8 *)h_diff.data(), Q * sizeof(T), nullptr);
    gpuLinearComb(bin, (int)Q, d_X, T(1), d_X, T(1), d_rqs);
    peer->reconstructInPlace(d_X, bin, Q, nullptr);                            // (nodeid-X)+r masked-public
    u32 *d_isX = gpuDpf<T>(dpfKey, party, d_X, &gAES, nullptr);                // raw boolean share (packed)
    gpuFree(d_X); gpuFree(d_rqs);
    size_t isXw = (Q + 31) / 32;
    u32 *h_isXb = (u32 *)moveToCPU((u8 *)d_isX, isXw * sizeof(u32), nullptr);
    gpuFree(d_isX);
    // shard_oh[k] = XOR_j isX[k,j] ; isX_sel[j] = XOR_k isX[k,j]  (boolean, LOCAL)
    std::vector<u32> b_oh(K, 0), b_isXsel(Ns, 0);
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < Ns; ++j) { u32 b = getbit(h_isXb, (size_t)k * Ns + j); b_oh[k] ^= b; b_isXsel[j] ^= b; }
    // (h_isXb left to process exit)
    fprintf(stderr, "[P%d] eval: isX (1-bit equality DPF) done\n", party); fflush(stderr);

    // ============================ GATHER via ONE gpuSelect ============================
    // selector[k,w] = shard_oh[k] XOR maskB[k,w]  (reconstruct bw=1) ; value = cat + maskX.
    const T *maskB = selKey.a, *maskX = selKey.b;   // CPU pointers into the key buffer
    size_t selw = (Ksel + 31) / 32;
    std::vector<u32> h_selb(selw, 0);
    std::vector<T> h_cat(Ksel);
    const size_t OFF_TM = N2 + NF + NC, OFF_TE = N2 + NF + NC + Ns;
    // selector is PER-SHARD (broadcast): open only K bits (shard_oh ^ maskB_perShard),
    // then replicate across Wtot LOCALLY. maskB[k*Wtot] is the per-shard mask.
    std::vector<u32> h_selKb((K + 31) / 32, 0);
    for (int k = 0; k < K; ++k)
        if (b_oh[k] ^ (u32)(maskB[(size_t)k * Wtot] & 1)) h_selKb[k >> 5] |= (1u << (k & 31));
    u32 *d_selK = (u32 *)moveToGPU((u8 *)h_selKb.data(), ((K + 31) / 32) * sizeof(u32), nullptr);
    peer->reconstructInPlace(d_selK, 1, K, nullptr);                           // open ONLY K selector bits
    u32 *h_selKpub = (u32 *)moveToCPU((u8 *)d_selK, ((K + 31) / 32) * sizeof(u32), nullptr);
    gpuFree(d_selK);
    for (int k = 0; k < K; ++k) {
        u32 sb = (h_selKpub[k >> 5] >> (k & 31)) & 1;             // opened per-shard selector bit
        if (sb) for (size_t w = 0; w < Wtot; ++w) { size_t i = (size_t)k * Wtot + w; h_selb[i >> 5] |= (1u << (i & 31)); }
        T *dst = &h_cat[(size_t)k * Wtot];
        std::memcpy(dst, &h_Astack[(size_t)k * N2], N2 * sizeof(T));
        std::memcpy(dst + N2, &h_Fstack[(size_t)k * NF], NF * sizeof(T));
        std::memcpy(dst + N2 + NF, &h_Yohstack[(size_t)k * NC], NC * sizeof(T));
        for (int j = 0; j < Ns; ++j) {                            // masks PUBLIC -> party0=value, party1=0
            dst[OFF_TM + j] = (party == 0) ? h_TMstack[(size_t)k * Ns + j] : 0;
            dst[OFF_TE + j] = (party == 0) ? h_TEstack[(size_t)k * Ns + j] : 0;
        }
    }
    u32 *d_sel = (u32 *)moveToGPU((u8 *)h_selb.data(), selw * sizeof(u32), nullptr);
    // d_sel already masked-public (opened at K level + broadcast); no K*Wtot reconstruct.
    T *d_val = (T *)moveToGPU((u8 *)h_cat.data(), Ksel * sizeof(T), nullptr);
    T *d_mX = (T *)moveToGPU((u8 *)maskX, Ksel * sizeof(T), nullptr);
    gpuLinearComb(64, (int)Ksel, d_val, T(1), d_val, T(1), d_mX);
    peer->reconstructInPlace(d_val, 64, Ksel, nullptr);                        // masked-public value
    gpuFree(d_mX);
    T *d_csel = gpuSelect<T, T, 0, 0>(peer, party, 64, selKey, d_sel, d_val, nullptr, /*opMasked=*/false);
    gpuFree(d_sel); gpuFree(d_val);
    T *h_out = (T *)moveToCPU((u8 *)d_csel, Ksel * sizeof(T), nullptr);
    gpuFree(d_csel);
    // sum over K (one shard is selected, rest are 0) -> cat_sel[Wtot] ; split
    std::vector<T> h_cs(Wtot, 0);
    for (int k = 0; k < K; ++k)
        for (size_t w = 0; w < Wtot; ++w) h_cs[w] += h_out[(size_t)k * Wtot + w];
    // (h_out left to process exit)
    std::vector<T> h_Asel(h_cs.begin(), h_cs.begin() + N2);
    std::vector<T> h_Fsel(h_cs.begin() + N2, h_cs.begin() + N2 + NF);
    std::vector<T> h_Yohsel(h_cs.begin() + N2 + NF, h_cs.begin() + N2 + NF + NC);
    std::vector<T> h_TMsel(h_cs.begin() + OFF_TM, h_cs.begin() + OFF_TM + Ns);
    std::vector<T> h_TEsel(h_cs.begin() + OFF_TE, h_cs.begin() + OFF_TE + Ns);
    u64 cb_gather = commBytes();
    fprintf(stderr, "[P%d] eval: gather select done (adj/feat/yoh/masks in one pass)\n", party); fflush(stderr);

    // ============================ keep (X removal) ============================

    // A_masked = keep_bit ? A_sel : 0  (zero X's adj COLUMN) ; train_mask_eff = keep_bit ? tm : 0.
    // DIRECT gpuSelect with keep_bit = NOT isX_sel (boolean): selector ^ maskB opened at bw=1
    // (cheap) + ONE value-open; replaces the Beaver multiply that opened both operands.
    std::vector<u32> b_keep(Ns);
    for (int j = 0; j < Ns; ++j) b_keep[j] = b_isXsel[j] ^ (u32)(party == 0 ? 1u : 0u);
    std::vector<T> h_Amask(N2), h_TMeff(Ns);
    {   // zero X's adjacency column: keep_bit[j] broadcast across rows i
        const T *mB = adjKeepKey.a, *mX = adjKeepKey.b;
        size_t aw = (N2 + 31) / 32; std::vector<u32> sb(aw, 0);
        for (int i = 0; i < Ns; ++i)
            for (int j = 0; j < Ns; ++j) { size_t e = (size_t)i * Ns + j;
                if (b_keep[j] ^ (u32)(mB[e] & 1)) sb[e >> 5] |= (1u << (e & 31)); }
        u32 *d_sb = (u32 *)moveToGPU((u8 *)sb.data(), aw * sizeof(u32), nullptr);
        peer->reconstructInPlace(d_sb, 1, N2, nullptr);                       // open N^2 selector bits (1-bit, cheap)
        T *d_v = (T *)moveToGPU((u8 *)h_Asel.data(), N2 * sizeof(T), nullptr);
        T *d_mx = (T *)moveToGPU((u8 *)mX, N2 * sizeof(T), nullptr);
        gpuLinearComb(64, (int)N2, d_v, T(1), d_v, T(1), d_mx);
        peer->reconstructInPlace(d_v, 64, N2, nullptr); gpuFree(d_mx);        // masked-public value (A_sel)
        T *d_o = gpuSelect<T, T, 0, 0>(peer, party, 64, adjKeepKey, d_sb, d_v, nullptr, /*opMasked=*/false);
        gpuFree(d_sb); gpuFree(d_v);
        moveIntoCPUMem((u8 *)h_Amask.data(), (u8 *)d_o, N2 * sizeof(T), nullptr); gpuFree(d_o);
    }
    {   // drop X from the train mask
        const T *mB = tmKeepKey.a, *mX = tmKeepKey.b;
        size_t tw = (Ns + 31) / 32; std::vector<u32> sb(tw, 0);
        for (int j = 0; j < Ns; ++j) if (b_keep[j] ^ (u32)(mB[j] & 1)) sb[j >> 5] |= (1u << (j & 31));
        u32 *d_sb = (u32 *)moveToGPU((u8 *)sb.data(), tw * sizeof(u32), nullptr);
        peer->reconstructInPlace(d_sb, 1, Ns, nullptr);
        T *d_v = (T *)moveToGPU((u8 *)h_TMsel.data(), Ns * sizeof(T), nullptr);
        T *d_mx = (T *)moveToGPU((u8 *)mX, Ns * sizeof(T), nullptr);
        gpuLinearComb(64, Ns, d_v, T(1), d_v, T(1), d_mx);
        peer->reconstructInPlace(d_v, 64, Ns, nullptr); gpuFree(d_mx);
        T *d_o = gpuSelect<T, T, 0, 0>(peer, party, 64, tmKeepKey, d_sb, d_v, nullptr, /*opMasked=*/false);
        gpuFree(d_sb); gpuFree(d_v);
        moveIntoCPUMem((u8 *)h_TMeff.data(), (u8 *)d_o, Ns * sizeof(T), nullptr); gpuFree(d_o);
    }
    u64 cb_end = commBytes();
    fprintf(stderr, "[P%d] eval: keep applied (adj col zeroed + train_mask drops X)\n", party); fflush(stderr);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (party == 0)
        fprintf(stderr, "[RESULT gpuSelect] bytes: isX+gather=%.2f MB keep=%.2f MB TOTAL=%.2f MB | rounds=%llu | online_time=%.1f ms\n",
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

    if (party == 0) {
        printf("[obliv-select] isX over %zu rows (K=%d Ns=%d) via 1-bit equality DPF, X never revealed\n", Q, K, Ns);
        printf("[obliv-select] gather = ONE gpuSelect(shard one-hot, concatenated cat) over Wtot=%zu; keep = 1-bit select\n", Wtot);
        printf("[obliv-select] wrote %s/{Asel_share*,featsel_share*,yohsel_share*,trainmask_share*,testmask_share*}.bin\n", sd.c_str());
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
        if (party == 0) printf("[obliv-select] (reveal) wrote {Amask,featsel,yohsel,tmeff,testmask}_clear.bin\n");
    }
    peer->sync();
    return 0;
}
