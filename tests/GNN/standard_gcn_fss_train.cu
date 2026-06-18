// Standard two-layer GCN full-batch training under GPU-MPC FSS.
//
// Objective:
//   CE(softmax(A · (ReLU(A · (X · W1) + b1) · W2) + b2), y)
//
// The graph, features, labels, and model weights are 2-out-of-2 additive
// shares. The train/test masks are public (standard transductive GCN). Keys are
// data independent, so a single offline keygen pass is generated once and the
// same buffer is replayed every epoch. Each epoch's online pass keeps all
// activations on the GPU and chains masked-public values through forward and
// backward, re-sharing only at the secret-bias / weight-update boundaries.
//
// Usage:
//   ./standard_gcn_fss_train <party> <peer_ip>
//       [--epochs E] [--lr LR] [--port N]
//       [--comm-buf-mb MB] [--key-buf-gb GB] [--reveal-eval]

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
            "Usage: %s <party> <peer_ip> [--epochs E] [--lr LR] [--port N] "
            "[--comm-buf-mb MB] [--key-buf-gb GB] [--reveal-eval]\n", argv[0]);
        return 1;
    }

    int party = std::atoi(argv[1]);
    const char *ip = argv[2];
    int epochs = 1;
    double lr = 0.01;
    int port = 42111;
    int comm_buf_mb = 5 * 1024;
    int key_buf_gb = 4;
    bool reveal_eval = false;
    bool secret_mask = false;
    int train_count_fixed = 0;

    for (int i = 3; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--epochs"))
        {
            if (++i >= argc || !sgf::parseIntArg(argv[i], &epochs) || epochs <= 0)
            { std::fprintf(stderr, "ERROR: --epochs requires a positive integer\n"); return 1; }
        }
        else if (!std::strcmp(argv[i], "--lr"))
        {
            if (++i >= argc || !sgf::parseDoubleArg(argv[i], &lr) || lr <= 0.0)
            { std::fprintf(stderr, "ERROR: --lr requires a positive number\n"); return 1; }
        }
        else if (!std::strcmp(argv[i], "--port"))
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
        else if (!std::strcmp(argv[i], "--reveal-eval")) { reveal_eval = true; }
        else if (!std::strcmp(argv[i], "--secret-mask")) { secret_mask = true; }
        else if (!std::strcmp(argv[i], "--train-count"))
        {
            if (++i >= argc || !sgf::parseIntArg(argv[i], &train_count_fixed) || train_count_fixed <= 0)
            { std::fprintf(stderr, "ERROR: --train-count requires a positive integer\n"); return 1; }
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
    dims.secretMask = secret_mask;   // gradient beaver-multiplied by the SECRET train_mask

    // Secure piranha softmax is the only softmax path.

    initGPUMemPool();
    sgf::trimGpuPool();
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    sytorch_init();

    // The llama party-to-party socket carries the online piranha-softmax traffic
    // and is reused by the GpuPeer (peer->peer = LlamaConfig::peer), so it is
    // bound to --port via WING_PORT. The GpuPeer does NOT open its own socket.
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

    // Throwaway buffer for the OTHER party's softmax-key stream during DEALER
    // keygen (this party keeps only its own stream, written to the key buffer).
    u8 *sm_dealer_scratch = (u8 *)std::malloc((size_t)256 * 1024 * 1024);

    sgf::T *h_X = sgf::readBin(sgf::graphSharePath(root, "feat", party), (size_t)meta.N * meta.F);
    sgf::T *h_A = sgf::readBin(sgf::graphSharePath(root, "adj", party), (size_t)meta.N * meta.N);
    sgf::T *h_Y = sgf::readBin(sgf::graphSharePath(root, "y_onehot", party), (size_t)meta.N * meta.C);
    // secret-mask: the train_mask is a SECRET share used in the gradient beaver-mul.
    // The public train/test masks + labels are still read for the (debug) reveal-eval.
    sgf::T *h_tmask_share = nullptr;
    if (secret_mask)
        h_tmask_share = sgf::readBin(root + "/graph/train_mask_share" + std::to_string(party) + ".bin", meta.N);
    uint8_t *train_mask = sgf::readBinT<uint8_t>(root + "/graph/train_mask.bin", meta.N);
    uint8_t *test_mask = sgf::readBinT<uint8_t>(root + "/graph/test_mask.bin", meta.N);
    int64_t *labels = sgf::readBinT<int64_t>(root + "/graph/labels.bin", meta.N);

    sgf::T *h_W1 = sgf::readBin(sgf::weightSharePath(root, "W1", party), (size_t)meta.F * meta.H);
    sgf::T *h_b1 = sgf::readBin(sgf::weightSharePath(root, "b1", party), meta.H);
    sgf::T *h_W2 = sgf::readBin(sgf::weightSharePath(root, "W2", party), (size_t)meta.H * meta.C);
    sgf::T *h_b2 = sgf::readBin(sgf::weightSharePath(root, "b2", party), meta.C);

    // train_count normalizer: a fixed public constant in secret-mask mode (the true
    // count would leak the shard); --train-count overrides, else fall back to the count.
    const int train_count = (secret_mask && train_count_fixed > 0)
                                ? train_count_fixed : sgf::countMask(train_mask, meta.N);
    if (train_count <= 0) { std::fprintf(stderr, "ERROR: empty train mask\n"); return 1; }

    const int coeff_scale = meta.scale + 16;
    sgf::T step_coeff_fp = sgf::fixedFromDouble(lr / (double)train_count, coeff_scale, "lr/train_count");

    std::printf("[train] root=%s party=%d port=%d N=%d F=%d H=%d C=%d scale=%d "
                "epochs=%d lr=%.6g train=%d coeff_scale=%d coeff_fp=%llu\n",
                root.c_str(), party, port, meta.N, meta.F, meta.H, meta.C, meta.scale,
                epochs, lr, train_count, coeff_scale, (unsigned long long)step_coeff_fp);
    std::fflush(stdout);

    sgf::ensureDir(root + "/weights");
    sgf::ensureDir(root + "/outputs");

    // ---- offline keygen (once; reused every epoch) ----
    // The piranha-softmax FSS key is generated by a DEALER llama whose key cursor
    // is THIS party's stream of the same key buffer the GPU keygen advances, so
    // the softmax key lands contiguously between the forward and backward keys.
    // (Mirrors GPU-GCN/experiments/GCN/gcn_dealer.cu.) Keygen is local/offline.
    u8 *kcur = key_buf_start;
    LlamaConfig::party = DEALER;
    LlamaBase<u64> *llama = new LlamaBase<u64>();
    {
        u8 *thr = sm_dealer_scratch;
        bool isServer = (party + 2 == SERVER);   // party 0 -> SERVER stream
        llama->initDealer((char **)(isServer ? &kcur : &thr),
                          (char **)(isServer ? &thr : &kcur));
    }

    auto kg0 = std::chrono::high_resolution_clock::now();
    sgf::gcnKeygen(&kcur, party, dims, &g, /*train=*/true, llama);
    checkCudaErrors(cudaDeviceSynchronize());
    sgf::g_keygen_us += std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - kg0).count();
    std::printf("[train] softmax=piranha keygen done: %zu key bytes, keygen=%lu us\n",
                (size_t)(kcur - key_buf_start), sgf::g_keygen_us);
    std::fflush(stdout);

    // ---- bring up the online 2-party link (after the offline keygen) ----
    // The online llama is bound ONCE to the stable cursor `cur`; each epoch resets
    // cur = key_buf_start so the softmax key (read inside gcnForwardRun via this
    // cursor) is consumed in lockstep with the GPU key reads. (Mirrors gcn_evaluator.)
    u8 *cur = key_buf_start;
    delete llama;                                // drop the dealer (buffer-only)
    LlamaConfig::party = party + 2;              // SERVER0->SERVER, SERVER1->CLIENT
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&cur);
    else
        llama->initClient(ip, (char **)&cur);
    peer->peer = LlamaConfig::peer;              // GpuPeer rides the llama socket
    peer->sync();

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        Stats stats;
        std::memset(&stats, 0, sizeof(stats));
        peer->sync();
        u64 comm_at_start = peer->bytesSent() + peer->bytesReceived();
        auto t0 = std::chrono::high_resolution_clock::now();

        cur = key_buf_start;                                 // reset shared cursor
        sgf::GcnFwdKeys fk;
        sgf::gcnReadForwardKeys(&cur, dims, &fk);            // load forward key material
        sgf::GcnFwdState st = sgf::gcnForwardRun(peer, party, &g, dims, fk,
                                                 h_X, h_A, h_W1, h_W2, &stats, llama, &cur);

        double train_acc = 0, test_acc = 0;
        sgf::T *post_clear = nullptr;
        if (reveal_eval)
        {
            post_clear = sgf::revealShareCpu(peer, st.h_P, (size_t)meta.N * meta.C, 64, &stats);
            train_acc = sgf::accuracyFromProbs(post_clear, labels, train_mask, meta.N, meta.C);
            test_acc = sgf::accuracyFromProbs(post_clear, labels, test_mask, meta.N, meta.C);
        }

        sgf::GcnBwdKeys bk;
        sgf::gcnReadBackwardKeys(&cur, dims, &bk);           // load backward key material
        sgf::gcnBackwardRun(peer, party, &g, dims, bk, &st, h_Y, train_mask,
                            h_tmask_share, step_coeff_fp, h_W1, h_W2, &stats);

        auto t1 = std::chrono::high_resolution_clock::now();
        u64 total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        u64 transfer_us = stats.transfer_time;
        u64 epoch_comm = peer->bytesSent() + peer->bytesReceived() - comm_at_start;
        std::printf("[train] epoch=%d online=%lu us h2d_d2h=%lu us comm=%lu B linear_comm=%lu B",
                    epoch, total_us, transfer_us, epoch_comm, stats.linear_comm_bytes);
        if (reveal_eval)
            std::printf(" train_acc=%.4f test_acc=%.4f", train_acc, test_acc);
        std::printf("\n");
        std::fflush(stdout);

        if (post_clear) cpuFree(post_clear);
        sgf::freeFwdState(&st);
    }

    sgf::writeBin(sgf::weightSharePath(root, "W1", party), h_W1, (size_t)meta.F * meta.H);
    sgf::writeBin(sgf::weightSharePath(root, "b1", party), h_b1, meta.H);
    sgf::writeBin(sgf::weightSharePath(root, "W2", party), h_W2, (size_t)meta.H * meta.C);
    sgf::writeBin(sgf::weightSharePath(root, "b2", party), h_b2, meta.C);
    std::printf("[train] updated weight shares written under %s/weights\n", root.c_str());
    std::fflush(stdout);

    delete[] h_X; delete[] h_A; delete[] h_Y;
    delete[] train_mask; delete[] test_mask; delete[] labels;
    delete[] h_W1; delete[] h_b1; delete[] h_W2; delete[] h_b2;
    destroyGPURandomness();
    _exit(0);
}
