#pragma once
#include "hnswlib.h"
#include <mutex>
#include <algorithm>
#include <fstream>
using hnsw = hnswlib::HierarchicalNSW<float>;
extern std::unique_lock<std::mutex>* glock;

class maria
{
    private:
    std::string index_file;

    public:
    int N;
    int dim;
    int S;
    int L;
    int K;

    std::string alg_name = "maria";
    Partition parti;
    Preprocess* prep = nullptr;
    IpSpace* ips = nullptr;
    hnsw** apgs = nullptr;
    public:
    maria(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) :parti(part_) {
        N = param_.N;
        dim = param_.dim + 1;
        L = param_.L;
        K = param_.K;
        S = param_.S;
        prep = &prep_;
        randomXT();
        buildIndex();
    }

    void buildIndex() {
        int M = 24;
        int ef = 40;
        ips = new IpSpace(dim);
        apgs = new hnsw * [parti.numChunks];
        size_t report_every = N / 20;
        if(report_every > 1e5) report_every = 1e5;

        int j1 = 0;
        for(int i = 0; i < parti.numChunks; ++i) {
            apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
            auto& appr_alg = apgs[i];
            auto id = parti.EachParti[i][0];
            auto data = prep->data.val[id];
            appr_alg->addPoint((void*)(data), (size_t)id);
            std::mutex inlock;

            auto vecsize = parti.nums[i];
            lsh::timer timer;
#pragma omp parallel for
            for(int k = 1; k < vecsize; k++) {
                size_t j2 = 0;
#pragma omp critical
                {
                    j1++;
                    j2 = j1;
                    if(j1 % report_every == 0) {
                        std::cout << (int)round(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
                        timer.restart();
                    }
                }
                j2 = parti.EachParti[i][k];
                float* data = prep->data.val[j2];
                appr_alg->addPoint((void*)(data), (size_t)j2);
            }
        }
    }

    void randomXT() {
        std::mt19937 rng(int(std::time(0)));
        std::uniform_real_distribution<float> ur(-1, 1);
        int count = 0;
        for(int j = 0; j < N; j++)
        {
            assert(parti.MaxLen[parti.chunks[j]] >= prep->norms[j]);
            prep->data.val[j][dim - 1] = sqrt(parti.MaxLen[parti.chunks[j]] - prep->norms[j]);
            if(ur(rng) > 0) {
                prep->data.val[j][dim - 1] *= -1;
                ++count;
            }
        }
    }

    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();

        for(int i = parti.numChunks - 1; i >= 0; --i) {
            if((!q->resHeap.empty()) && (1.0f - q->resHeap.top().dist) >
                q->norm * (parti.MaxLen[i])) break;


            //apgs[i] = new hnsw(ips, parti.nums[i], M, ef);
            auto& appr_alg = apgs[i];
            auto id = parti.EachParti[i][0];
            auto data = prep->data.val[id];
            //appr_alg->addPoint((void*)(data), (size_t)id);
            //std::mutex inlock;
            appr_alg->setEf(q->k + 100);
            auto res = appr_alg->searchKnn(q->queryPoint, q->k);
            while(!res.empty()) {
                auto top = res.top();
                res.pop();
                q->resHeap.emplace(top.second, top.first);
                while(q->resHeap.size() > q->k) q->resHeap.pop();
            }

        }

        while(!q->resHeap.empty()) {
            auto top = q->resHeap.top();
            q->resHeap.pop();
            q->res.emplace_back(top.id, 1.0 - top.dist);
        }

        std::reverse(q->res.begin(), q->res.end());

        q->time_total = timer.elapsed();
    }

    //void GetTables(Preprocess& prep);
    //bool IsBuilt(const std::string& file);
    ~maria() {
        for(int i = 0; i < parti.numChunks; ++i) {
            delete apgs[i];
        }
        delete[] apgs;
    }
};

class ipNSW {
    private:
    std::string index_file;
    IpSpace* ips = nullptr;
    hnsw* apg = nullptr;
    //Preprocess* prep = nullptr;
    Data data;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels
    float indexing_time = 0;
    public:
    int N;
    int dim;
    int M;
    // // Number of hash functions
    // int S;
    // //#L Tables; 
    // int L;
    // // Dimension of the hash table
    // int K;
    //std::string index_file;
    std::string alg_name = "hnsw";
    float delta_thred = 0.001;

    ipNSW(Preprocess& prep_, Parameter& param_, const std::string& file, int M_ = 24) {
        M = M_;
        reset(prep_.data, param_, file);
    }

    ipNSW(Data& data_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) {
        reset(data_, param_, file);
    }

    inline bool exists_test(const std::string& name) {
        //return false;
        std::ifstream f(name.c_str());
        return f.good();
    }

    void reset(Data& data_, Parameter& param_, const std::string& file, bool isbuilt = 1) {
        N = param_.N;
        dim = param_.dim;
        data = data_;
        index_file = file;
        if(isbuilt && exists_test(index_file)) {
            std::cout << "Loading index from " << index_file << ":\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            ips = new IpSpace(dim);
            apg = new hnsw(ips, index_file, false);
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }
        else {
            buildIndex();
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/ipnsw_info.txt", "a");
            if(fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }
    }

    void setEf(size_t ef) {
        apg->setEf(ef);
    }

    int getM() {
        return apg->maxM0_;
    }

    void buildMap() {
        hnsw_maps.resize(N, -1);
        for(int i = 0; i < N; ++i) {
            size_t uid = (apg->getExternalLabel(i));
            hnsw_maps[uid] = i;
        }
    }

    void getEdgeSet(int pid, int* ptr) {
        int id = hnsw_maps[pid];

        int* dptr = (int*)(apg->get_linklist0(id));
        size_t size = apg->getListCount((unsigned int*)dptr);

        ptr[0] = size;
        for(size_t j = 1; j <= size; j++) {
            ptr[j] = apg->getExternalLabel(*(dptr + j));
        }
    }

    void buildIndex() {
        //int M = 24;
        int efC = 80;
        ips = new IpSpace(dim);
        //apg = new hnsw[parti.numChunks];
        size_t report_every = N / 20;
        if(report_every > 1e5) report_every = N / 100;
        lsh::timer timer, timer_total;
        int j1 = 0;
        apg = new hnsw(ips, N, M, efC);
        auto id = 0;
        auto data0 = data.val[id];
        apg->addPoint((void*)(data0), (size_t)id);
        std::mutex inlock;

        auto vecsize = N;

#pragma omp parallel for schedule(dynamic,256)
        for(int k = 1; k < vecsize; k++) {
            size_t j2 = 0;
#pragma omp critical
            {
                j1++;
                j2 = j1;
                if(j1 % report_every == 0) {
                    std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
                    timer.restart();
                }
            }
            j2 = k;
            float* data0 = data.val[j2];
            apg->addPoint((void*)(data0), (size_t)j2);
        }

        std::cout << " Finish building ip-NSW\n";

        indexing_time = timer_total.elapsed();
        apg->saveIndex(index_file);
    }


    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();
        int ef = apg->ef_;
        //ef = 200;
        auto& appr_alg = apg;
        auto id = 0;
        auto res = appr_alg->searchKnn(q->queryPoint, q->k + ef);

        while(!res.empty()) {
            auto top = res.top();
            res.pop();
            q->resHeap.emplace(top.second, top.first);
            while(q->resHeap.size() > q->k) q->resHeap.pop();
        }

        while(!q->resHeap.empty()) {
            auto top = q->resHeap.top();
            q->resHeap.pop();
            q->res.emplace_back(top.id, 1.0 - top.dist);
        }
        std::reverse(q->res.begin(), q->res.end());
        q->time_total = timer.elapsed();
    }

    ~ipNSW() {
        delete apg;
    }
};

//HNSW for ANNS
class HNSW {
    private:
    std::string index_file;
    L2Space* ips = nullptr;
    hnsw* apg = nullptr;
    //Preprocess* prep = nullptr;
    Data data;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels
    float indexing_time = 0;
    public:
    int N;
    int dim;
    // // Number of hash functions
    // int S;
    // //#L Tables; 
    // int L;
    // // Dimension of the hash table
    // int K;
    //std::string index_file;
    std::string alg_name = "hnsw";

    HNSW(Preprocess& prep_, Parameter& param_, const std::string& file, const std::string& funtable) {
        reset(prep_.data, param_, file);
    }

    HNSW(Data& data_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable) {
        reset(data_, param_, file);
    }

    inline bool exists_test(const std::string& name) {
        //return false;
        std::ifstream f(name.c_str());
        return f.good();
    }

    void reset(Data& data_, Parameter& param_, const std::string& file, bool isbuilt = 1) {
        N = param_.N;
        dim = param_.dim;
        data = data_;
        index_file = file;
        if(isbuilt && exists_test(index_file)) {
            std::cout << "Loading index from " << index_file << ":\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            ips = new L2Space(dim);
            apg = new hnsw(ips, index_file, false);
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }
        else {
            buildIndex();
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/ipnsw_info.txt", "a");
            if(fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }
    }

    void setEf(size_t ef) {
        apg->setEf(ef);
    }

    int getM() {
        return apg->maxM0_;
    }

    void buildMap() {
        hnsw_maps.resize(N, -1);
        for(int i = 0; i < N; ++i) {
            size_t uid = (apg->getExternalLabel(i));
            hnsw_maps[uid] = i;
        }
    }

    void getEdgeSet(int pid, int* ptr) {
        int id = hnsw_maps[pid];

        int* dptr = (int*)(apg->get_linklist0(id));
        size_t size = apg->getListCount((unsigned int*)dptr);

        ptr[0] = size;
        for(size_t j = 1; j <= size; j++) {
            ptr[j] = apg->getExternalLabel(*(dptr + j));
        }
    }

    void buildIndex() {
        int M = 24;
        int efC = 80;
        ips = new L2Space(dim);
        //apg = new hnsw[parti.numChunks];
        size_t report_every = N / 20;
        if(report_every > 1e5) report_every = N / 100;
        lsh::timer timer, timer_total;
        int j1 = 0;
        apg = new hnsw(ips, N, M, efC);
        auto id = 0;
        auto data0 = data.val[id];
        apg->addPoint((void*)(data0), (size_t)id);
        std::mutex inlock;

        auto vecsize = N;

#pragma omp parallel for schedule(dynamic,256)
        for(int k = 1; k < vecsize; k++) {
            size_t j2 = 0;
#pragma omp critical
            {
                j1++;
                j2 = j1;
                if(j1 % report_every == 0) {
                    std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
                    timer.restart();
                }
            }
            j2 = k;
            float* data0 = data.val[j2];
            apg->addPoint((void*)(data0), (size_t)j2);
        }

        std::cout << " Finish building ip-NSW\n";

        indexing_time = timer_total.elapsed();
        apg->saveIndex(index_file);
    }


    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();
        int ef = apg->ef_;
        //ef = 200;
        auto& appr_alg = apg;
        auto id = 0;
        auto res = appr_alg->searchKnn(q->queryPoint, q->k + ef);

        while(!res.empty()) {
            auto top = res.top();
            res.pop();
            q->resHeap.emplace(top.second, top.first);
            while(q->resHeap.size() > q->k) q->resHeap.pop();
        }

        while(!q->resHeap.empty()) {
            auto top = q->resHeap.top();
            q->resHeap.pop();
            q->res.emplace_back(top.id, 1.0 - top.dist);
        }
        std::reverse(q->res.begin(), q->res.end());
        q->time_total = timer.elapsed();
    }

    ~HNSW() {
        delete apg;
    }
};