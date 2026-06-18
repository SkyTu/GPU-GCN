// Clean FSS GCN -- EVALUATOR (online 2PC) for per-shard training.
//   ./gcn_evaluator <party> <ip> <keyDir> <dataRoot> <shard> <epochs> <weightsFile> <outWeights>
// Reads the dealer's per-epoch keys, runs forward + piranha-softmax + backward
// each epoch (peer = the other party), then party 0 dumps the trained (public)
// weights to <outWeights> for numpy accuracy eval.
#include <cassert>
#include <cstdint>
#include <chrono>
#include <unistd.h>
#include <string>
#include <fstream>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"
#include "utils/helper_cuda.h"

#include "experiments/GCN/gcn_train_common.h"
#include <sytorch/backend/llama_base.h>

int global_device = 0;
double wan_time = 0;
uint64_t fss_rounds = 0;

int main(int argc, char *argv[])
{
    assert(argc >= 9 && "usage: gcn_evaluator <party> <ip> <keyDir> <dataRoot> <shard> <epochs> <weightsFile> <outWeights>");
    int party = atoi(argv[1]);
    std::string ip = argv[2];
    std::string keyDir = argv[3];
    std::string dataRoot = argv[4];
    int shard = atoi(argv[5]);
    int epochs = atoi(argv[6]);
    std::string weightsFile = argv[7];
    std::string outWeights = argv[8];

    initGPUMemPool();   // must precede any gpuMalloc (sets vGPU cudaMalloc fallback)
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    sytorch_init();

    gcn::ShardMeta sm;
    gcn::loadShardMeta(gcn::shardDir(dataRoot, shard) + "/meta.txt", &sm);
    const int Ns = sm.Ns, F = sm.F, H = sm.H, C = sm.C;
    printf("[gcn_eval] shard=%d Ns=%d F=%d H=%d C=%d epochs=%d party=%d\n", shard, Ns, F, H, C, epochs, party);

    auto gcnm = dcf::orca::getGCNModel<u64>(Ns, F, H, C);
    gcnm.m->setTrain(false);
    gcnm.m->initWeights(weightsFile, true /*floatWeights*/);

    // shard data (this party's shares)
    std::string d = gcn::shardDir(dataRoot, shard);
    u64 *h_X = gcn::readBinT<u64>(gcn::sharePath(d, "feat", party), (size_t)Ns * F);
    u64 *h_A = gcn::readBinT<u64>(gcn::sharePath(d, "adj", party), (size_t)Ns * Ns);
    u64 *h_Y = gcn::readBinT<u64>(gcn::sharePath(d, "y_onehot", party), (size_t)Ns * C);
    gcnm.agg1->setAShare(h_A);
    gcnm.agg2->setAShare(h_A);

    // key buffer
    std::string keySzDir = keyDir + "/keysize/";
    std::string modelName = "gcn_shard" + std::to_string(shard);
    u64 keySz = gcn::getKeySzHelper(keySzDir, modelName);
    u8 *keyBuf;
    getAlignedBuf(&keyBuf, keySz);
    u8 *curKeyBuf = keyBuf;

    // 2-party connect (crypto party = party + 2; SERVER listens, CLIENT connects)
    SigmaPeer *peer = new GpuPeer(false);
    LlamaConfig::party = party + 2;
    auto llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    std::string keyFile = keyDir + "/" + modelName + "_key" + std::to_string(party) + ".dat";
    int fd = openForReading(keyFile);

    for (int e = 0; e < epochs; e++)
    {
        u64 keyReadTime = 0;
        readKey(fd, keySz, keyBuf, &keyReadTime); // load this epoch's key blob
        curKeyBuf = keyBuf;                        // reset shared cursor (GPU keys + llama softmax keys)
        peer->sync();
        auto comm0 = peer->bytesSent() + peer->bytesReceived();
        auto t0 = std::chrono::high_resolution_clock::now();
        gcn::trainStep(gcnm.m, &curKeyBuf, party, peer, h_X, h_Y, &g, (LlamaBase<u64> *)llama, e);
        checkCudaErrors(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        u64 comm = peer->bytesSent() + peer->bytesReceived() - comm0;
        u64 us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (e == 0 || (e + 1) % 10 == 0 || e == epochs - 1)
            printf("[gcn_eval] epoch=%d time=%lu us comm=%lu B\n", e + 1, us, comm);
        fflush(stdout);
    }
    close(fd);

    // dump trained (public) weights for numpy accuracy eval
    if (party == SERVER0)
    {
        gcnm.m->dumpWeights(outWeights);
        printf("[gcn_eval] trained weights dumped -> %s\n", outWeights.c_str());
    }
    LlamaConfig::peer->close();
    destroyGPURandomness();
    printf("[gcn_eval] done shard=%d\n", shard);
    _exit(0);
}
