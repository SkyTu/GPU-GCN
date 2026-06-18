// Clean FSS GCN -- DEALER (offline keygen) for per-shard training.
//   ./gcn_dealer <party> <keyDir> <dataRoot> <shard> <epochs>
// Writes all <epochs> per-iteration training keys (forward + piranha-softmax +
// backward) to one file. Keygen is local + weight-independent (orca FCLayer
// mask_W=0), so each party's dealer runs independently and finishes before its
// evaluator reads the file. No dealer<->evaluator FIFO needed.
#include <cassert>
#include <cstdint>
#include <unistd.h>
#include <string>
#include <cstring>

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
    assert(argc >= 6 && "usage: gcn_dealer <party> <keyDir> <dataRoot> <shard> <epochs>");
    int party = atoi(argv[1]);
    std::string keyDir = argv[2];
    std::string dataRoot = argv[3];
    int shard = atoi(argv[4]);
    int epochs = atoi(argv[5]);

    initGPUMemPool();   // must precede any gpuMalloc (sets vGPU cudaMalloc fallback)
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    sytorch_init();

    gcn::ShardMeta sm;
    gcn::loadShardMeta(gcn::shardDir(dataRoot, shard) + "/meta.txt", &sm);
    printf("[gcn_dealer] shard=%d Ns=%d F=%d H=%d C=%d epochs=%d\n", shard, sm.Ns, sm.F, sm.H, sm.C, epochs);

    auto gcnm = dcf::orca::getGCNModel<u64>(sm.Ns, sm.F, sm.H, sm.C);
    gcnm.m->setTrain(false); // plain SGD, no momentum

    // llama dealer (piranha softmax preprocessing)
    LlamaConfig::party = DEALER;
    auto llama = new LlamaBase<u64>();
    u8 *startPtr, *curPtr, *tmpPtr1, *tmpPtr2;
    size_t bufSize = 4ULL * OneGB;
    getAlignedBuf(&startPtr, bufSize);
    tmpPtr1 = (u8 *)malloc(OneGB);
    bool isServer = party + 2 == SERVER;
    llama->initDealer((char **)(isServer ? &curPtr : &tmpPtr2), (char **)(isServer ? &tmpPtr2 : &curPtr));

    makeDir(keyDir);
    std::string keySzDir = keyDir + "/keysize/";
    std::string modelName = "gcn_shard" + std::to_string(shard);
    std::string keyFile = keyDir + "/" + modelName + "_key" + std::to_string(party) + ".dat";

    char *zeros = nullptr;
    size_t padding = 0;
    int fd = openForWriting(keyFile);
    for (int e = 0; e < epochs; e++)
    {
        curPtr = startPtr;
        tmpPtr2 = tmpPtr1;
        gcn::genModelKey(gcnm.m, &curPtr, party, &g, (LlamaBase<u64> *)llama, e);
        if (e == 0)
        {
            size_t keySz = curPtr - startPtr;
            padding = (4096 - (keySz % 4096)) % 4096;
            zeros = new char[padding ? padding : 1];
            memset(zeros, 0, padding);
            gcn::writeKeySzHelper(keySzDir, modelName, keySz + padding);
            printf("[gcn_dealer] per-epoch key = %lu MB (padded %lu B)\n",
                   (keySz + padding) / (1024 * 1024), padding);
            fflush(stdout);
        }
        if (padding) { memcpy(curPtr, zeros, padding); curPtr += padding; }
        writeKeyBuf(fd, curPtr - startPtr, startPtr); // append this epoch's key
    }
    assert(0 == fsync(fd));
    close(fd);
    delete[] zeros;
    destroyGPURandomness();
    printf("[gcn_dealer] done shard=%d epochs=%d -> %s\n", shard, epochs, keyFile.c_str());
    _exit(0);
}
