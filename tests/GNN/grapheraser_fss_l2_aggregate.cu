
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_mul.h"

#include <cuda_runtime.h>

extern cudaMemPool_t mempool;

using T = u64;

static const std::string DATA_ROOT = []()
{
    const char *e = std::getenv("FSS_DATA_ROOT");
    return std::string(e && *e ? e : "datasets/cora_shards");
}();

struct GMeta
{
    int N, F, C, k, scale;
};

static int parseIntKV(const std::string &l)
{
    auto eq = l.find('=');
    return eq == std::string::npos ? -1 : std::atoi(l.c_str() + eq + 1);
}
static int parseIntKV2(const std::string &path, const std::string &key)
{
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l))
        if (l.rfind(key + "=", 0) == 0)
            return parseIntKV(l);
    return -1;
}
static GMeta loadGlobalMeta()
{
    return {
        parseIntKV2(DATA_ROOT + "/meta.txt", "N"),
        parseIntKV2(DATA_ROOT + "/meta.txt", "F"),
        parseIntKV2(DATA_ROOT + "/meta.txt", "C"),
        parseIntKV2(DATA_ROOT + "/meta.txt", "k"),
        parseIntKV2(DATA_ROOT + "/meta.txt", "scale"),
    };
}
static int loadShardNs(int s)
{
    return parseIntKV2(DATA_ROOT + "/shards/shard_" + std::to_string(s) + "_meta.txt", "Ns");
}
static T *readBin(const std::string &path, size_t elems)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    assert(f.is_open());
    size_t bytes = (size_t)f.tellg();
    assert(bytes == elems * sizeof(T));
    f.seekg(0);
    T *b = new T[elems];
    f.read((char *)b, bytes);
    return b;
}
static bool fileExists(const std::string &p)
{
    std::ifstream f(p);
    return f.good();
}

static void writeBin(const std::string &path, const T *b, size_t elems)
{
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f)
        {
            fprintf(stderr, "ERROR: cannot open %s for write\n", tmp.c_str());
            _exit(1);
        }
        f.write((const char *)b, elems * sizeof(T));
        if (!f)
        {
            fprintf(stderr, "ERROR: write failed for %s\n", tmp.c_str());
            _exit(1);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
    {
        fprintf(stderr, "ERROR: rename %s -> %s failed: %s\n",
                tmp.c_str(), path.c_str(), std::strerror(errno));
        _exit(1);
    }
}

static T *extractTestRows(const T *post, int Ns, int C,
                          const uint8_t *test_mask, int &out_rows)
{
    out_rows = 0;
    for (int i = 0; i < Ns; ++i)
        if (test_mask[i])
            ++out_rows;
    T *o = new T[(size_t)out_rows * C];
    int j = 0;
    for (int i = 0; i < Ns; ++i)
        if (test_mask[i])
        {
            std::memcpy(&o[(size_t)j * C], &post[(size_t)i * C], C * sizeof(T));
            ++j;
        }
    return o;
}

static int asInt(size_t n)
{
    assert(n <= (size_t)INT_MAX && "GPU helper takes int element counts");
    return (int)n;
}

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

static T *publicMaskedOutputToShare(T *d_public_masked, T *d_mask_share,
                                    int party, size_t elems, int bw)
{
    T public_coeff = party == SERVER0 ? T(1) : T(0);
    gpuLinearComb(bw, asInt(elems), d_public_masked,
                  public_coeff, d_public_masked, T(-1), d_mask_share);
    return d_public_masked;
}

static T *fssMulPublicAlphaByPostShare(GpuPeer *peer, int party,
                                       AESGlobalContext *gAES,
                                       u8 **curPtrRef, u8 *startPtr,
                                       const T *h_alpha_share,
                                       const T *h_post_share,
                                       size_t elems, int bw, int scale,
                                       Stats *stats)
{
    const TruncateType tr = TruncateType::TrWithSlack;
    const int N = asInt(elems);

    auto d_mask_alpha = randomGEOnGpu<T>(elems, bw);
    auto d_mask_alphas = splitMaskForParty(d_mask_alpha, party, elems, bw);
    auto d_mask_post = randomGEOnGpu<T>(elems, bw);
    auto d_mask_posts = splitMaskForParty(d_mask_post, party, elems, bw);

    auto d_mask_product = gpuKeygenMul<T>(curPtrRef, party, bw, scale, N,
                                          d_mask_alpha, d_mask_post, tr, gAES);
    auto d_mask_products = splitMaskForParty(d_mask_product, party, elems, bw);

    u8 *readPtr = startPtr;
    auto k = readGPUMulKey<T>(&readPtr, N, N, N, tr);

    gpuFree(d_mask_alpha);
    gpuFree(d_mask_post);
    gpuFree(d_mask_product);

    auto d_masked_alpha = openMaskedInputFromCpuShare(peer, h_alpha_share,
                                                      d_mask_alphas, elems,
                                                      bw, stats);
    auto d_masked_post = openMaskedInputFromCpuShare(peer, h_post_share,
                                                     d_mask_posts, elems,
                                                     bw, stats);
    gpuFree(d_mask_alphas);
    gpuFree(d_mask_posts);

    auto d_product = gpuMul<T>(peer, party, bw, scale, N, k,
                               d_masked_alpha, d_masked_post, tr, gAES,
                               stats);
    gpuFree(d_masked_alpha);
    gpuFree(d_masked_post);

    publicMaskedOutputToShare(d_product, d_mask_products, party, elems, bw);
    gpuFree(d_mask_products);
    return d_product;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <party> <peer_ip> mean | weighted <alpha.bin>\n", argv[0]);
        return 1;
    }
    int party = atoi(argv[1]);
    const char *ip = argv[2];
    std::string mode = argv[3];
    std::string alphaPath = (mode == "weighted" && argc >= 5) ? argv[4] : "";
    assert((party == SERVER0 || party == SERVER1) && "party must be 0 or 1");

    initGPUMemPool();
    {

        uint64_t threshold = 0;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(mempool, 0);
    }
    AESGlobalContext gAES;
    initAESContext(&gAES);
    initGPURandomness();

    auto peer = new GpuPeer(true);
    peer->connect(party, (char *)ip);

    GMeta gm = loadGlobalMeta();
    const int k = gm.k, C = gm.C, scale = gm.scale, bw = 64;
    if (k <= 0 || C <= 0 || scale <= 0)
    {
        fprintf(stderr, "ERROR: bad meta (k=%d C=%d scale=%d). Did prepare_shards.py run?\n",
                k, C, scale);
        return 1;
    }

    T *alpha = nullptr;
    if (mode == "weighted")
    {
        assert(!alphaPath.empty() && "alpha.bin path required for weighted mode");
        std::string dir;
        size_t slash = alphaPath.find_last_of('/');
        dir = (slash == std::string::npos) ? "" : alphaPath.substr(0, slash + 1);
        std::string shareAlphaPath = dir + "alpha_share" + std::to_string(party) + ".bin";
        if (fileExists(shareAlphaPath))
        {
            alpha = readBin(shareAlphaPath, k);
            printf("[L2] mode=weighted, using α share %s\n", shareAlphaPath.c_str());
        }
        else
        {
            T *alpha_public = readBin(alphaPath, k);
            alpha = new T[k];
            for (int s = 0; s < k; ++s)
                alpha[s] = party == SERVER0 ? alpha_public[s] : T(0);
            printf("[L2] mode=weighted, using legacy public α (fp scale=%d):", scale);
            for (int s = 0; s < k; ++s)
                printf(" %.4f", (double)(int64_t)alpha_public[s] / (double)(1LL << scale));
            printf("\n");
            delete[] alpha_public;
        }
    }
    else if (mode == "mean")
    {
        alpha = new T[k];
        T one_over_k = (T)((1LL << scale) / k);
        for (int s = 0; s < k; ++s)
            alpha[s] = party == SERVER0 ? one_over_k : T(0);
        printf("[L2] mode=mean, α[s]=1/k=%.4f\n",
               (double)(int64_t)one_over_k / (double)(1LL << scale));
    }
    else
    {
        fprintf(stderr, "ERROR: unknown mode '%s' (expected mean|weighted)\n", mode.c_str());
        return 1;
    }

    int num_test = parseIntKV2(DATA_ROOT + "/shards/shard_0_meta.txt", "num_test");
    if (num_test <= 0)
    {
        fprintf(stderr, "ERROR: num_test=%d\n", num_test);
        return 1;
    }
    printf("[L2] num_test=%d  C=%d  k=%d  scale=%d  bw=%d\n", num_test, C, k, scale, bw);

    const size_t KEY_BUF = (size_t)256 * 1024 * 1024;
    u8 *kStart = nullptr, *kCur = nullptr;
    getKeyBuf(&kStart, &kCur, KEY_BUF);

    const size_t aggElems = (size_t)num_test * C;
    std::vector<T> zero_share(aggElems, T(0));
    T *d_agg_share = (T *)moveToGPU((u8 *)zero_share.data(),
                                    aggElems * sizeof(T), nullptr);
    Stats stats;
    std::memset(&stats, 0, sizeof(stats));

    auto t0 = std::chrono::high_resolution_clock::now();
    u64 totalComm0 = peer->bytesSent() + peer->bytesReceived();

    for (int s = 0; s < k; ++s)
    {
        int Ns = loadShardNs(s);
        T *post_share = readBin(
            DATA_ROOT + "/posteriors/shard_" + std::to_string(s) +
                "_post_share" + std::to_string(party) + ".bin",
            (size_t)Ns * C);

        std::ifstream mf(DATA_ROOT + "/shards/shard_" + std::to_string(s) + "_test_mask.bin",
                         std::ios::binary);
        if (!mf.is_open())
        {
            fprintf(stderr, "ERROR: missing test_mask for shard %d\n", s);
            return 1;
        }
        std::vector<uint8_t> test_mask(Ns);
        mf.read((char *)test_mask.data(), Ns);

        int rows = 0;
        T *test_rows_share = extractTestRows(post_share, Ns, C, test_mask.data(), rows);
        if (rows != num_test)
        {
            fprintf(stderr, "ERROR: shard %d has %d test rows (expected %d)\n",
                    s, rows, num_test);
            return 1;
        }

        std::vector<T> alpha_share(aggElems, alpha[s]);

        kCur = kStart;
        T *d_prod_share = fssMulPublicAlphaByPostShare(peer, party, &gAES,
                                                       &kCur, kStart,
                                                       alpha_share.data(),
                                                       test_rows_share,
                                                       aggElems, bw, scale,
                                                       &stats);
        gpuLinearComb(bw, asInt(aggElems), d_agg_share,
                      T(1), d_agg_share, T(1), d_prod_share);

        gpuFree(d_prod_share);
        delete[] post_share;
        delete[] test_rows_share;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    u64 totalComm = (peer->bytesSent() + peer->bytesReceived()) - totalComm0;

    printf("[L2] aggregation done in %ld ms, FSS comm = %.3f MB (%lu bytes)\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
           (double)totalComm / (1024.0 * 1024.0), totalComm);

    std::string share_path = DATA_ROOT + "/posteriors/aggregate_post_share" +
                             std::to_string(party) + "_" + mode + ".bin";
    auto h_agg_share = (T *)moveToCPU((u8 *)d_agg_share, aggElems * sizeof(T), nullptr);
    writeBin(share_path, h_agg_share, aggElems);
    printf("[L2] wrote %s\n", share_path.c_str());

    peer->reconstructInPlace(d_agg_share, bw, aggElems, &stats);
    std::string clear_path = DATA_ROOT + "/posteriors/aggregate_post_clear_" + mode + ".bin";
    if (party == SERVER0)
    {
        auto h_agg_clear = (T *)moveToCPU((u8 *)d_agg_share, aggElems * sizeof(T), nullptr);
        writeBin(clear_path, h_agg_clear, aggElems);
        printf("[L2] wrote %s\n", clear_path.c_str());
        cpuFree(h_agg_clear);
    }
    else
    {
        printf("[L2] party 1 participated in final reveal; party 0 publishes %s\n",
               clear_path.c_str());
    }

    gpuFree(d_agg_share);
    cpuFree(h_agg_share);
    delete[] alpha;
    fflush(stdout);
    _exit(0);
}
