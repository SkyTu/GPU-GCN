// Standard two-layer GCN inference under GPU-MPC FSS.
//
// Forward:
//   T1 = X  · W1
//   U1 = A  · T1 + b1
//   H1 = ReLU(U1)
//   T2 = H1 · W2
//   Z  = A  · T2 + b2
//   P  = softmax(Z)
//
// Structure: a single offline keygen pass fills the key buffer (the dealer's
// job; data independent), then the online pass keeps every activation on the
// GPU and chains masked-public values gate to gate -- no per-gate re-sharing
// and no host round trips.
//
#include <unistd.h>
#include <cstdlib>

#include "tests/GNN/standard_gcn_fss_common.h"

#include <sytorch/backend/llama_base.h>
#include <sytorch/utils.h>

namespace sgf = standard_gcn_fss;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::fprintf(stderr,
                     "Usage: %s <party> <peer_ip> [--port N] [--comm-buf-mb MB] [--key-buf-gb GB]\n",
                     argv[0]);
        return 1;
    }

    int party = std::atoi(argv[1]);
    const char *ip = argv[2];
    int port = 42101;
    int comm_buf_mb = 5 * 1024;
    int key_buf_gb = 4;

    for (int i = 3; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--port"))
        {
            if (++i >= argc || !sgf::parseIntArg(argv[i], &port) || port <= 0 || port > 65535)
            { std::fprintf(stderr, "ERROR: --port requires an integer in [1,65535]\n"); return 1; }
        }
        else if (!std::strcmp(argv[i], "--comm-buf-mb"))
        {
            if (++i >= argc || !sgf::parseIntArg(argv[i], &comm_buf_mb) || comm_buf_mb <= 0)
            { std::fprintf(stderr, "ERROR: --comm-buf-mb requires a positive integer\n"); return 1; }
        }
        else if (!std::strcmp(argv[i], "--key-buf-gb"))
        {
            if (++i >= argc || !sgf::parseIntArg(argv[i], &key_buf_gb) || key_buf_gb <= 0)
            { std::fprintf(stderr, "ERROR: --key-buf-gb requires a positive integer\n"); return 1; }
        }
        else { std::fprintf(stderr, "ERROR: unknown argument: %s\n", argv[i]); return 1; }
    }

    if (party != SERVER0 && party != SERVER1)
    {
        std::fprintf(stderr, "ERROR: party must be 0 or 1\n");
        return 1;
    }

    const std::string root = sgf::dataRoot();
    sgf::Meta meta;
    sgf::loadMeta(root + "/meta.txt", &meta);
    sgf::GcnDims dims = sgf::makeDims(meta);

    // Secure piranha softmax is the only softmax path.

    initGPUMemPool();
    sgf::trimGpuPool();
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    sytorch_init();

    setenv("WING_PORT", std::to_string(port).c_str(), 1);
    auto peer = new GpuPeer(true);

    u8 *key_buf_start = nullptr, *key_buf_cur = nullptr;
    getKeyBuf(&key_buf_start, &key_buf_cur, (size_t)key_buf_gb * 1024 * 1024 * 1024);
    // Pinned headroom after the key buffer. The forward key readers parse a few
    // bytes past the tail of the consumed key region, so a pinned slack buffer
    // must follow the key buffer (this is the old softmax scratch region, now
    // unused for data but still required to back-stop the key reads).
    u8 *key_buf_headroom = nullptr, *key_buf_headroom_cur = nullptr;
    getKeyBuf(&key_buf_headroom, &key_buf_headroom_cur, (size_t)512 * 1024 * 1024);
    u8 *sm_dealer_scratch = (u8 *)std::malloc((size_t)256 * 1024 * 1024);

    std::printf("[infer] root=%s party=%d port=%d N=%d F=%d H=%d C=%d scale=%d softmax=piranha\n",
                root.c_str(), party, port, meta.N, meta.F, meta.H, meta.C, meta.scale);
    std::fflush(stdout);

    sgf::T *h_X = sgf::readBin(sgf::graphSharePath(root, "feat", party), (size_t)meta.N * meta.F);
    sgf::T *h_A = sgf::readBin(sgf::graphSharePath(root, "adj", party), (size_t)meta.N * meta.N);
    sgf::T *h_W1 = sgf::readBin(sgf::weightSharePath(root, "W1", party), (size_t)meta.F * meta.H);
    sgf::T *h_b1 = sgf::readBin(sgf::weightSharePath(root, "b1", party), meta.H);
    sgf::T *h_W2 = sgf::readBin(sgf::weightSharePath(root, "W2", party), (size_t)meta.H * meta.C);
    sgf::T *h_b2 = sgf::readBin(sgf::weightSharePath(root, "b2", party), meta.C);

    sgf::ensureDir(root + "/outputs");

    Stats stats;
    std::memset(&stats, 0, sizeof(stats));

    // ---- offline keygen (dealer phase) ----
    u8 *kcur = key_buf_start;
    LlamaConfig::party = DEALER;
    LlamaBase<u64> *llama = new LlamaBase<u64>();
    {
        u8 *thr = sm_dealer_scratch;
        bool isServer = (party + 2 == SERVER);
        llama->initDealer((char **)(isServer ? &kcur : &thr),
                          (char **)(isServer ? &thr : &kcur));
    }
    auto kg0 = std::chrono::high_resolution_clock::now();
    sgf::gcnKeygen(&kcur, party, dims, &g, /*train=*/false, llama);
    checkCudaErrors(cudaDeviceSynchronize());
    sgf::g_keygen_us += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - kg0).count();
    std::printf("[infer] softmax=piranha keygen done: %zu key bytes\n",
                (size_t)(kcur - key_buf_start));
    std::fflush(stdout);

    // ---- bring up the online link (after offline keygen) ----
    u8 *cur = key_buf_start;
    delete llama;
    LlamaConfig::party = party + 2;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&cur);
    else
        llama->initClient(ip, (char **)&cur);
    peer->peer = LlamaConfig::peer;

    // ---- online forward ----
    peer->sync();
    u64 online_comm0 = peer->bytesSent() + peer->bytesReceived();
    auto t0 = std::chrono::high_resolution_clock::now();
    cur = key_buf_start;
    sgf::GcnFwdKeys fk;
    sgf::gcnReadForwardKeys(&cur, dims, &fk);           // load offline key material
    sgf::GcnFwdState st = sgf::gcnForwardRun(peer, party, &g, dims, fk,
                                             h_X, h_A, h_W1, h_W2, &stats, llama, &cur);
    auto t1 = std::chrono::high_resolution_clock::now();

    const size_t logits_elems = (size_t)meta.N * meta.C;
    sgf::writeBin(root + "/outputs/logits_share" + std::to_string(party) + ".bin", st.h_Z, logits_elems);
    sgf::writeBin(root + "/outputs/post_share" + std::to_string(party) + ".bin", st.h_P, logits_elems);

    sgf::T *logits_clear = sgf::revealShareCpu(peer, st.h_Z, logits_elems, 64, &stats);
    sgf::T *post_clear = sgf::revealShareCpu(peer, st.h_P, logits_elems, 64, &stats);
    sgf::writeBin(root + "/outputs/logits_clear.bin", logits_clear, logits_elems);
    sgf::writeBin(root + "/outputs/post_clear.bin", post_clear, logits_elems);

    int64_t *labels = sgf::readBinT<int64_t>(root + "/graph/labels.bin", meta.N);
    uint8_t *train_mask = sgf::readBinT<uint8_t>(root + "/graph/train_mask.bin", meta.N);
    uint8_t *test_mask = sgf::readBinT<uint8_t>(root + "/graph/test_mask.bin", meta.N);
    double train_acc = sgf::accuracyFromProbs(post_clear, labels, train_mask, meta.N, meta.C);
    double test_acc = sgf::accuracyFromProbs(post_clear, labels, test_mask, meta.N, meta.C);

    u64 online_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    u64 online_comm = peer->bytesSent() + peer->bytesReceived() - online_comm0;
    std::printf("[infer] keygen=%lu us online=%lu us online_comm=%lu B keygen_comm=%lu B "
                "linear_comm=%lu B train_acc=%.4f test_acc=%.4f\n",
                sgf::g_keygen_us, online_us, online_comm, sgf::g_keygen_comm_bytes,
                stats.linear_comm_bytes, train_acc, test_acc);
    std::fflush(stdout);

    sgf::freeFwdState(&st);
    cpuFree(logits_clear);
    cpuFree(post_clear);
    delete[] labels; delete[] train_mask; delete[] test_mask;
    delete[] h_X; delete[] h_A; delete[] h_W1; delete[] h_b1; delete[] h_W2; delete[] h_b2;
    destroyGPURandomness();
    _exit(0);
}
