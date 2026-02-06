#pragma once

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h> 
#include <mutex>
#include <algorithm>
#include <fstream>

#include "patch_ubuntu.h"
#include "Preprocess.h"

#include "hnsw_faiss.h"

class hnsw_Faiss {
    private:
    std::string index_file;
    //IpSpace* ips = nullptr;
    //hnsw* apg = nullptr;
    //Preprocess* prep = nullptr;

    faiss::IndexHNSWFlat* apg = nullptr;

    Data data;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels
    float indexing_time = 0;
    public:
    int N;
    int dim;
    int M = 24;
    std::string alg_name = "hf";
    float delta_thred = 0.001;

    hnsw_Faiss(Preprocess& prep_, Parameter& param_, const std::string& file, int M_ = 24) {
        M = M_;
        reset(prep_.data, param_, file);
    }

    hnsw_Faiss(Data& data_, Parameter& param_, const std::string& file, int M_ = 24) {
        M = M_;
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
            apg = faiss::read_index(index_file.c_str());
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }
        else {
            buildIndex();
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/Faiss_info.txt", "a");
            if(fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }
    }

    void setEf(size_t ef) {
        //apg->setEf(ef);
        apg->hnsw.efSearch = ef;
    }


    void buildIndex() {
        //M = 24;
        int efC = 80;


        lsh::timer timer, timer_total;

        apg = new faiss::IndexHNSWFlat(dim, M, faiss::METRIC_INNER_PRODUCT);
        apg->hnsw.efConstruction = efC;

        apg->add(N, data.base);

        std::cout << " Finish building Faiss-HNSW\n";

        indexing_time = timer_total.elapsed();

        //apg->save(index_file.c_str());
        faiss::write_index(apg, index_file.c_str());
    }


    void knn(std::vector<queryN>& qs) {
        auto k = qs[0].k;
        int nq = qs.size();
        std::vector<float> xq(nq * dim);
        lsh::timer timer;
        std::vector<faiss::idx_t> I(k * nq);
        std::vector<float> D(k * nq);
        for(int i = 0;i < nq;++i){
            auto& q = qs[i];
            memcpy(xq.data() + i * dim, q.queryPoint, sizeof(float) * dim);

        }
        apg->search(nq, xq.data(), k, D.data(), I.data());

        // for (int i = nq - 5; i < nq; i++) {
        //     for (int j = 0; j < k; j++) {
        //         printf("%5zd ", I[i * k + j]);
        //     }
        //     printf("\n");
        // }

        // printf("D=\n");
        // for (int i = 0; i < 100; i++) {
        //     for (int j = 0; j < k; j++) {
        //         printf("%5f ", D[i * k + j]);
        //     }
        //     printf("\n");
        // }

        for(int i = 0;i < nq;++i){
            auto& q = qs[i];
            for(int j = 0;j < k;++j){
                q.res.emplace_back(I[i * k + j], D[i * k + j]);
            }
        }

        //qtime = timer.elapsed();
    }

    ~hnsw_Faiss() {
        delete apg;
    }
};
