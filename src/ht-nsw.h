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

#include "srp.h"

#include "compute_prob_mnk.h"

template <bool use_heuristic = 1, bool isbuilt = 1>
class ht_nsw {
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
    std::string alg_name = "ht_nsw";
    std::vector<float> rank;
    std::vector<std::pair<float, int>> norm_pairs;
    std::vector<std::vector<float>> blocks; //#block* (block_size * dim)
    std::vector<std::vector<int>> pid_blocks;//#block* (block_size)
    float thred = 0.5;
    float gamma = 0.2;
    float delta_thred = 0.001;

    int n1 = 0;
    int n2 = 0;
    int max_range = 0;
    //double delta = 0.01;
    int block_size = 2048;

    //Parameters for heuristic NAP
    double L = 10.0;
    double a, b;
    double h;
    float kappa = 1.0f;
    int n_int = 2000;
    int step = 200;
    std::vector<double> xs, w, Fx, Fdx, Fy;


    ht_nsw(Preprocess& prep_, const std::string& file, float delta_thred_ = 0.01) {
        delta_thred = delta_thred_;
        if(prep_.data.N < 1e5) block_size /= 4;
        rank = prep_.rank;
        norm_pairs = prep_.len;
        reset(prep_.data, file);
        alg_name += (use_heuristic ? "_h" : "_g");
    }

    //template <bool use_heuristic = 1, bool isbuilt = 1>
    void reset(Data& data_, const std::string& file) {
        // N = param_.N;
        // dim = param_.dim;
        N = data_.N;
        dim = data_.dim;
        data = data_;
        index_file = file;

        // //greedyNAP();
        // heuristicNAP();

        if(use_heuristic) index_file += ".heuristic";
        else index_file += ".greedy";

        if(isbuilt && exists_test(index_file)) {

            std::cout << "Loading index for " << index_file << ":\n";
            lsh::timer timer;
            //load n1 n2
            //load mips_for_head
            load_nap();

            if(n1 < N){
                Data data_head;
                data_head.dim = data.dim;
                data_head.N = N - n1;
                data_head.val = new float* [data_head.N];
                for(int i = n1; i < N; ++i) data_head.val[i - n1] = data.val[norm_pairs[i].second];
                srp = new lsh::srp(data_head, index_file + ".head", norm_pairs, n1, 5, floor(log2(N - n1)), 0);

            }
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            ips = new IpSpace(dim);
            apg = new hnsw(ips, index_file + ".body", false);

            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
            indexing_time = timer.elapsed();
            std::cout << "Loading time:" << indexing_time << "  seconds.\n";
        }
        else {
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            if(use_heuristic) heuristicNAP();
            else greedyNAP();
            buildIndex();
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "Build time:" << indexing_time << "  seconds.\n";
            // FILE* fp = nullptr;
            // fopen_s(&fp, "./indexes/nbg_info.txt", "a");
            // if(fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            saveIndexInfo(index_file, "./indexes/indexing_info.txt", memf - mem, indexing_time);
        }
    }

    void greedyNAP(){
        int pt1 = N - block_size;
        int pt2 = pt1;

        while(pt2 > 0){
            pt2 = std::lower_bound(norm_pairs.begin(), norm_pairs.end(), std::make_pair(gamma * norm_pairs[pt1].first, -1)) - norm_pairs.begin();
            if(pt2 > pt1 - block_size){
                break;
            }
            pt1 = pt1 - block_size;
        }
        pt1 = N - block_size;
        pt2 = 100;

        double k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
        if(k < thred){
            n2 = 0;
            n1 = N - 1;
            max_range = N;
            std::cout << "main_body: [" << n2 << ", " << n1 << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
            return;
        }

        float old_F = 0.0f;
        while(pt2 < pt1){

            if(compute_integral(N - pt1, pt2, k) < old_F) break;

            while(pt2 < pt1 && compute_integral(N - pt1, pt2, k)>1 - delta_thred){
                pt2++;
                k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
                //x = k * boost::math::quantile(standard_normal, 1.0 - 10.0 / (N - pt1));
            }

            pt1 -= block_size;
            if(pt1 <= pt2) break;
            old_F = compute_integral(N - pt1, pt2, k);
        }

        printf("initial k=%f, m=%d,n=%d,I=%f\n", k, N - pt1, pt2, compute_integral(N - pt1, pt2, k));
        n2 = pt2;
        n1 = pt1;
        max_range = pt1 - pt2 + 1;
        std::cout << "main_body: [" << n2 << ", " << n1 << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
    }

    double compute_F(){
        double sum = 0.0;
        for(int i = 0; i < n_int; ++i) {
            double integrand = (Fdx[i]) * Fy[i];
            sum += w[i] * integrand;
        }
        return sum * (h / 3.0);
    }

    void initialize_F(){
        a = -L, b = +L;
        h = (b - a) / n_int;

        xs.resize(n_int);
        w.resize(n_int);
        Fx.resize(n_int, 1.0);
        Fdx.resize(n_int, 0.0);
        Fy.resize(n_int, 1.0);

        for(int i = 0; i < n_int; ++i) {
            xs[i] = a + i * h;
            w[i] = (i == 0 || i == N) ? 1.0 :
                (i % 2 == 1 ? 4.0 : 2.0);
        }
    }

    void update_Fx(float val){
        val /= kappa;
        // #pragma omp parallel for
        for(int i = 0; i < n_int; ++i) {
            auto x = xs[i];
            auto& F_old = Fx[i];
            auto& Fd_old = Fdx[i];
            double f = Phi(x / (val));
            double fd = 1.0 / val * phi(x / val);
            double F_new = F_old * f;
            double Fd_new = Fd_old * f + F_old * fd;
            F_old = F_new;
            Fd_old = Fd_new;
        }
    }

    void update_Fx(int s, int e){
        //#pragma omp parallel for
        for(int i = 0; i < n_int; ++i) {
            float val = norm_pairs[s].first / kappa;
            int m = e - s;
            auto x = xs[i];
            auto& F_old = Fx[i];
            auto& Fd_old = Fdx[i];
            double f = std::pow(Phi(x / (val)), m);
            double fd = 1.0 / val * phi(x / val) * m * std::pow(Phi(x / (val)), m - 1);
            double F_new = F_old * f;
            double Fd_new = Fd_old * f + F_old * fd;
            F_old = F_new;
            Fd_old = Fd_new;
        }
    }

    void update_Fy(float val){
        //#pragma omp parallel for
        for(int i = 0; i < n_int; ++i) {
            auto x = xs[i];
            auto& Fy_old = Fy[i];
            double f = Phi(x / (val / kappa));
            Fy_old *= f;
        }
    }

    void update_Fy(int s, int e){
        // #pragma omp parallel for
        for(int i = 0; i < n_int; ++i) {
            // for(int j = s; j < e; ++j){
            //     auto val = norm_pairs[j].first;
            //     auto x = xs[i];
            //     auto& Fy_old = Fy[i];
            //     double f = Phi(x / (val));
            //     Fy_old *= f;
            // }

            auto val = norm_pairs[e - 1].first / kappa;
            auto x = xs[i];
            auto& Fy_old = Fy[i];
            double f = Phi(x / (val));
            Fy_old *= std::pow(f, e - s);
            // for(auto& val : vals){
            //     auto x = xs[i];
            //     auto& Fy_old = Fy[i];
            //     double f = Phi(x / (val));
            //     Fy_old *= f;
            // }
        }
    }

    void heuristicNAP(){
        int pt1 = N - 1;
        int pt2 = 0;
        kappa = norm_pairs[N - 1].first;
        int min_body_size = std::max(2048, N / 100);
        //auto good_pt = std::lower_bound(norm_pairs.begin(), norm_pairs.end(), std::make_pair(0.1 * norm_pairs[N - block_size].first, -1)) - norm_pairs.begin();
        //int good_pt = 1000;
        //if (pt2 < (int)good_pt) pt2 = good_pt;

        //int init = 0;
        while(norm_pairs[pt2].first < 0.1 * norm_pairs[N - block_size].first) pt2++;
        pt2 += step;

        initialize_F();
        update_Fx(norm_pairs[pt1].first);
        update_Fy(pt2 - step, pt2);
        double I = compute_F();
        std::cout << "initial I=" << I << "\n";
        std::cout << "initial k=" << (norm_pairs[pt1].first / norm_pairs[pt2].first) << ", m=" << N - pt1 << ", n=" << pt2 << "\n";
        //pt1 = N - 2048;
        //pt2 = 100;
        lsh::progress_display pd(pt1 - pt2);
        lsh::timer timer;
        while(pt2 + step + min_body_size < pt1) {
            double k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
            if(k < thred){
                n2 = 0;
                n1 = N - 1;
                max_range = N;
                std::cout << "main_body: [" << n2 << ", " << n1 << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
                return;
            }

            while(pt2 + step + min_body_size < pt1 && compute_F()> 1.0 - delta_thred){
                //pt2++;
                pt2 += step;
                pd += step;
                update_Fy(pt2 - step, pt2);
            }

            while(pt2 + step + min_body_size < pt1 && compute_F() <= 1.0 - delta_thred){
                if(N - pt1 < block_size){
                    pt1--;
                    ++pd;
                    update_Fx(norm_pairs[pt1].first);
                }
                else{
                    pt1 -= step;
                    pd += step;
                    update_Fx(pt1, pt1 + step);
                }
            }


        }
        std::cout << "Time cost: " << timer.elapsed() << " seconds. \n";
        pt1 = N - block_size;
        double k = (norm_pairs[pt1].first / norm_pairs[pt2].first);
        printf("initial k=%f, m=%d,n=%d,I=%f\n", k, N - pt1, pt2, compute_F());
        n2 = pt2;
        n1 = pt1;
        max_range = pt1 - pt2;
        std::cout << "main_body: [" << n2 << ", " << n1 << "], size=" << max_range << ", ratio=" << (float)max_range / N << "\n";
    }

    inline bool exists_test(const std::string& name) {
        //return false;
        std::ifstream f(name.c_str());
        return f.good();
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

    lsh::srp* srp = nullptr;
    std::vector<int> mips_for_head;
    void buildIndex() {
        int M = 24;
        int efC = 80;
        ips = new IpSpace(dim);
        //apg = new hnsw[parti.numChunks];
        int size = n1 - n2;

        if(size < 0) {
            printf("The size of body is 0\n");
            exit(-1);
        }

        size_t report_every = size / 20;
        if(report_every > 1e5) report_every = size / 100;
        lsh::timer timer, timer_total;
        int j1 = 0;
        apg = new hnsw(ips, size, M, efC);
        auto id = n1 - 1;
        auto data0 = data.val[id];
        apg->addPoint((void*)(data0), (size_t)id);

        //id = norm_pairs[N - 1].second;
        //data0 = data.val[id];
        //apg_head->addPoint((void*)(data0), (size_t)id);
        std::mutex inlock;

        auto vecsize = N;

        if(n1 < N){
            Data data_head;
            data_head.dim = data.dim;
            data_head.N = N - n1;
            data_head.val = new float* [data_head.N];
            for(int i = n1; i < N; ++i) data_head.val[i - n1] = data.val[norm_pairs[i].second];
            srp = new lsh::srp(data_head, index_file + ".head", norm_pairs, n1, 5, floor(log2(N - n1)), 0);

        }

        //std::cout << "Finish building hash tables for head\n";
        //printf("Inserted %d points\n", j1);

#pragma omp parallel for //schedule(dynamic,256)
        for(int k = n2; k < n1 - 1; k++) {
            size_t j2 = norm_pairs[k].second;
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

        std::cout << "Finish building HT-NSW for main body, ";
        printf("Inserted %d points\n", j1);

        mips_for_head.resize(N - n1);
        for(int i = n1; i < N; ++i) {
            int ef = 50;
            auto res = apg->searchKnnInternalLabel(data[norm_pairs[i].second], ef);
            while(res.size() > 1) res.pop();
            mips_for_head[i - n1] = res.top().second;
        }
        std::cout << "Store the MIPS of head point in body!\n ";
        indexing_time = timer_total.elapsed();
        apg->saveIndex(index_file + ".body");

        save_nap();
    }

    void save_nap() {
        std::ofstream out(index_file, std::ios::binary);
        out.write((char*)&n1, sizeof(int));
        out.write((char*)&n2, sizeof(int));
        out.write((char*)mips_for_head.data(), sizeof(int) * mips_for_head.size());
        out.close();
    }

    void load_nap() {
        std::ifstream in(index_file, std::ios::binary);
        in.read((char*)&n1, sizeof(int));
        in.read((char*)&n2, sizeof(int));
        mips_for_head.resize(N - n1);
        in.read((char*)mips_for_head.data(), sizeof(int) * mips_for_head.size());
        in.close();
    }


    void knn(queryN* q) {
        lsh::timer timer;

        q->norm = sqrt(cal_inner_product(q->queryPoint, q->queryPoint, data.dim));
        if(q->norm < 1e-5){
            std::cout << "warning: zero query norm!" << std::endl;
            exit(-1);
        }

        timer.restart();
        int ef = apg->ef_;
        //ef = 200;
        auto& appr_alg = apg;
        auto id = 0;

        //auto qres = apg_head->searchKnn(q->queryPoint, q->k + ef);
        bool f = 1;
        // for(int i = N - 1;i >= main_body_after;--i){
        //     if(qres.size() > q->k && 1.0 - q->resHeap.top().dist < q->norm * norm_pairs[main_body_after - 1].first){ f = 0; break; }
        //     qres.push(std::make_pair(1.0f - cal_inner_product(q->queryPoint, data[norm_pairs[i].second], data.dim), norm_pairs[i].second));
        //     if(qres.size() > q->k) qres.pop();

        // }

        int SEARCH_HEAD = 1;
        switch(SEARCH_HEAD) {
            case 0: {//Linear scan for head
                    std::priority_queue<std::pair<float, unsigned int>> qres;
                    for(int i = n1;i < N;++i)
                        qres.push(std::make_pair(1.0f - cal_inner_product(q->queryPoint, data[norm_pairs[i].second], data.dim), norm_pairs[i].second));

                    while(!qres.empty()) {
                        auto top = qres.top();
                        qres.pop();
                        q->resHeap.emplace(top.second, top.first);
                        while(q->resHeap.size() > q->k) q->resHeap.pop();
                    }
                }
                  break;
            case 1: {//LSH for head
                    srp->knn(q);
                    break;
                }
            default:
                break;
        }


        if(f && !(q->resHeap.size() >= q->k && 1.0f - q->resHeap.top().dist >= q->norm * norm_pairs[n1 - 1].first)){
            std::vector<unsigned> eps;
            eps.reserve(q->k);
            while(q->resCos.size()) {
                auto x = q->resCos.top().id;
                eps.emplace_back(mips_for_head[x]);
                q->resCos.pop();
            }
            auto res = appr_alg->searchBaseLayerST<false, false>(eps, (void*)q->queryPoint, (size_t)(q->k + ef));
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

    void NAP_accuracy(Preprocess& prep, int k, float& r_head, float& r_body) {
        int cnt_head = 0;
        int cnt_body = 0;

        std::vector<int> rank(data.N, 0);
        int cnt = 0;
        for(auto& x : norm_pairs) rank[x.second] = cnt++;

        for(int i = 0;i < prep.benchmark.N;++i){
            for(int j = 0;j < k;++j){
                if(rank[prep.benchmark.indice[i][j]] >= n1) cnt_head++;
                else if(rank[prep.benchmark.indice[i][j]] >= n2) cnt_body++;
            }
        }

        r_head = (float)cnt_head / (prep.benchmark.N * k);
        r_body = (float)cnt_body / (prep.benchmark.N * k);
        std::cout << "delta_thred=" << delta_thred << ", n1=" << n1 << ", n2=" << n2 << ", k=" << k << "\n";
        std::cout << "head ratio: " << r_head << ", body ratio: " << r_body << "\n\n\n";
    }

    ~ht_nsw() {
        delete apg;
    }
};
