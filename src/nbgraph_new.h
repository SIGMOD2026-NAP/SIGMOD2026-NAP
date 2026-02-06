#pragma once
#include "hnswlib.h"
#include <mutex>
#include <algorithm>
#include <fstream>
#include "Preprocess.h"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
using hnsw = hnswlib::HierarchicalNSW<float>;
extern std::unique_lock<std::mutex>* glock;


#include "compute_prob_mnk.h"

class nbgraphv1 {
    private:
    std::string index_file;
    IpSpace* ips = nullptr;
    hnsw* apg = nullptr;
    hnsw* apg_head = nullptr;
    //Preprocess* prep = nullptr;
    Data data;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels
    float indexing_time = 0;
    public:
    int N;
    int dim;
    std::string alg_name = "nbgraph_new";
    std::vector<float> rank;
    std::vector<std::pair<float, int>> norm_pairs;
    std::vector<std::vector<float>> blocks; //#block* (block_size * dim)
    std::vector<std::vector<int>> pid_blocks;//#block* (block_size)
    float thred = 0.5;
    float gamma = 0.2;

    int main_body_before = 0;
    int main_body_after = 0;
    int max_range = 0;
    float delta_thred = 0.001;
    const int block_size = 512;

    nbgraphv1(Preprocess& prep_, Parameter& param_, const std::string& file) {
        rank = prep_.rank;
        norm_pairs = prep_.len;
        reset(prep_.data, param_, file);
    }

    void partition(){
        int pt1 = N - block_size;
        int pt2 = pt1;

        while(pt2 > 0){
            pt2 = std::lower_bound(norm_pairs.begin(), norm_pairs.end(), std::make_pair(gamma * norm_pairs[pt1].first, -1)) - norm_pairs.begin();
            if(pt2 > pt1 - block_size){
                break;
            }
            pt1 = pt1 - block_size;
            // std::vector<float> bk(block_size * dim, 0.0f);
            // std::vector<int> bkid(block_size, -1);
            // // for(int i = pt2, j = 0; i < pt1; ++i, ++j){
            // //     memcpy(data.val[norm_pairs[i].second], data.val[norm_pairs[i].second] + dim, bk.data() + j * dim);
            // //     bkid[j] = norm_pairs[i].second;
            // // }

            // // blocks.push_back(bk);
            // // pid_blocks.push_back(bkid);
            // pt1 = pt2;
        }
        pt1 = N - 512;
        pt2 = 100;
        boost::math::normal_distribution<> standard_normal(0.0, 1.0);
        const double pi = 3.14159265358979323846;
        if(pt2 < pt1){
            //double k = 2.0 / pi * atan(norm_pairs[pt2].first / norm_pairs[pt1].first);
            double k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
            if(k < thred){
                main_body_before = 0;
                main_body_after = N - 1;
                max_range = N;
                std::cout << "main_body: [" << main_body_before << ", " << main_body_after << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
                return;
            }
            double x = k * boost::math::quantile(standard_normal, 1.0 - 1.0 / (N - pt1));
            printf("initial k=%f, x=%f\n", k, x);
            while(pt2 < pt1 && pt2 * (1.0 - boost::math::cdf(standard_normal, x)) < 1.0){
                pt2++;
                k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
                x = k * boost::math::quantile(standard_normal, 1.0 - 10.0 / (N - pt1));
            }
            //double ratio1=
            // while((N - pt1) * (1.0 - cdf(standard_normal, p / k)) < 1.0) {
            //     pt1--;
            //     p = quantile(standard_normal, 1.0 - 1.0 / pt2);
            // }
            main_body_before = pt2;
            main_body_after = pt1;
            max_range = pt1 - pt2 + 1;
            std::cout << "main_body: [" << main_body_before << ", " << main_body_after << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
        }
    }

    void partition1(){
        int pt1 = N - block_size;
        int pt2 = pt1;

        while(pt2 > 0){
            pt2 = std::lower_bound(norm_pairs.begin(), norm_pairs.end(), std::make_pair(gamma * norm_pairs[pt1].first, -1)) - norm_pairs.begin();
            if(pt2 > pt1 - block_size){
                break;
            }
            pt1 = pt1 - block_size;
            // std::vector<float> bk(block_size * dim, 0.0f);
            // std::vector<int> bkid(block_size, -1);
            // // for(int i = pt2, j = 0; i < pt1; ++i, ++j){
            // //     memcpy(data.val[norm_pairs[i].second], data.val[norm_pairs[i].second] + dim, bk.data() + j * dim);
            // //     bkid[j] = norm_pairs[i].second;
            // // }

            // // blocks.push_back(bk);
            // // pid_blocks.push_back(bkid);
            // pt1 = pt2;
        }
        pt1 = N - 512;
        pt2 = 100;
        if(pt2 < pt1){
            double k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
            if(k < thred){
                main_body_before = 0;
                main_body_after = N - 1;
                max_range = N;
                std::cout << "main_body: [" << main_body_before << ", " << main_body_after << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
                return;
            }

            while(pt2 < pt1 && compute_integral(N - pt1, pt2, k)>0.99){
                pt2++;
                k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
                //x = k * boost::math::quantile(standard_normal, 1.0 - 10.0 / (N - pt1));
            }

            printf("initial k=%f, m=%d,n=%d,I=%f\n", k, N - pt1, pt2, compute_integral(N - pt1, pt2, k));
            //double ratio1=
            // while((N - pt1) * (1.0 - cdf(standard_normal, p / k)) < 1.0) {
            //     pt1--;
            //     p = quantile(standard_normal, 1.0 - 1.0 / pt2);
            // }
            main_body_before = pt2;
            main_body_after = pt1;
            max_range = pt1 - pt2 + 1;
            std::cout << "main_body: [" << main_body_before << ", " << main_body_after << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
        }
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

        partition1();
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
            fopen_s(&fp, "./indexes/nbg_info.txt", "a");
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
        ips = new IpSpace(dim);
        //apg = new hnsw[parti.numChunks];
        size_t report_every = N / 20;
        if(report_every > 1e5) report_every = N / 100;
        lsh::timer timer, timer_total;
        int j1 = 0;
        apg = new hnsw(ips, N, M, efC);
        apg_head = new hnsw(ips, N, M, efC);
        auto id = 0;
        auto data0 = data.val[id];
        apg->addPoint((void*)(data0), (size_t)id);

        id = norm_pairs[N - 1].second;
        data0 = data.val[id];
        apg_head->addPoint((void*)(data0), (size_t)id);
        std::mutex inlock;

        auto vecsize = N;

#pragma omp parallel for //schedule(dynamic,256)
        for(int k = main_body_before; k < main_body_after; k++) {
            //if(k > 0) continue;
            size_t j2 = norm_pairs[k].second;
            //printf("rank=%f\n", rank[j2]);
            // if(rank[j2] < thred){
            //     //printf("j2=%d\n", j2);
            //     continue;
            // }
#pragma omp critical
            {
                j1++;
                //j2 = j1;
                if(j1 % report_every == 0) {
                    std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
                    timer.restart();
                }
            }
            j2 = norm_pairs[k].second;
            float* data0 = data.val[j2];
            apg->addPoint((void*)(data0), (size_t)j2);
        }

        std::cout << "Finish building NB-Graph for main body\n";
        printf("Inserted %d points\n", j1);

        j1 = 0;
#pragma omp parallel for //schedule(dynamic,256)
        for(int k = main_body_after; k < N; k++) {
            size_t j2 = norm_pairs[k].second;
            //printf("rank=%f\n", rank[j2]);
            // if(rank[j2] < thred){
            //     //printf("j2=%d\n", j2);
            //     continue;
            // }
#pragma omp critical
            {
                j1++;
                //j2 = j1;
                if(j1 % report_every == 0) {
                    std::cout << (int)(j1 / (0.01 * N)) << " %, " << (report_every / (1000.0 * timer.elapsed())) << " kips\n";
                    timer.restart();
                }
            }
            j2 = norm_pairs[k].second;
            float* data0 = data.val[j2];
            apg_head->addPoint((void*)(data0), (size_t)j2);
        }

        std::cout << "Finish building NB-Graph for head\n";
        printf("Inserted %d points\n", j1);

        indexing_time = timer_total.elapsed();
        apg_head->saveIndex(index_file + ".head");
    }


    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();
        int ef = apg->ef_;
        ef = 200;
        auto& appr_alg = apg;
        auto id = 0;

        bool f = 1;

        std::priority_queue<std::pair<float, unsigned int>> qres;
        for(int i = main_body_after;i < N;++i) qres.push(std::make_pair(1.0f - cal_inner_product(q->queryPoint, data[norm_pairs[i].second], data.dim), norm_pairs[i].second));

        while(!qres.empty()) {
            auto top = qres.top();
            qres.pop();
            q->resHeap.emplace(top.second, top.first);
            while(q->resHeap.size() > q->k) q->resHeap.pop();
        }
        //f = 0;
        ef = 0;
        if(f && q->resHeap.size() >= q->k && 1.0 - q->resHeap.top().dist < q->norm * norm_pairs[main_body_after - 1].first) {
            auto res = appr_alg->searchKnn(q->queryPoint, q->k + ef);
            while(!res.empty()) {
                auto top = res.top();
                res.pop();
                q->resHeap.emplace(top.second, top.first);
                while(q->resHeap.size() > q->k) q->resHeap.pop();
            }
        }
        else{
            //printf("skip main body\n");
        }

        while(!q->resHeap.empty()) {
            auto top = q->resHeap.top();
            q->resHeap.pop();
            q->res.emplace_back(top.id, 1.0 - top.dist);
        }
        std::reverse(q->res.begin(), q->res.end());
        q->time_total = timer.elapsed();
    }

    void batch_knn(std::vector<queryN*>& qs) {
        lsh::timer timer;
        timer.restart();


        for(auto& q : qs) {
            knn(q);
        }
    }

    ~nbgraphv1() {
        delete apg;
    }
};
