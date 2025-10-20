//
//
// https://github.com/ThrunGroup/BanditMIPS/blob/camera_ready/algorithms/napg/python_bindings/bindings.cpp
//
//
// #include "nadglib.h"
#include "napgalg.h"
#include <thread>

// using nadg = hnswlib::NormAdjustedDelaunayGraph<float>;
/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn)
{
    if(numThreads <= 0)
    {
        numThreads = std::thread::hardware_concurrency();
    }

    if(numThreads == 1)
    {
        for(size_t id = start; id < end; id++)
        {
            fn(id, 0);
        }
    }
    else
    {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for(size_t threadId = 0; threadId < numThreads; ++threadId)
        {
            threads.push_back(std::thread([&, threadId]
            {
                while(true) {
                    size_t id = current.fetch_add(1);

                    if((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    }
                    catch(...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                } }));
        }
        for(auto& thread : threads)
        {
            thread.join();
        }
        if(lastException)
        {
            std::rethrow_exception(lastException);
        }
    }
}


using napg = hnswlib::NormAdjustedProximityGraph<float>;

class myNAPG {
    private:
    std::string index_file;
    IpSpace* ips = nullptr;
    napg* appr_alg = nullptr;
    //Preprocess* prep = nullptr;
    Data data;
    std::vector<int> hnsw_maps;//maps between hnsw internel labels and external labels

    public:
    float delta_thred = 0.001;
    int N;
    int dim;
    // Number of hash functions
    int S;
    //#L Tables; 
    int L;
    // Dimension of the hash table
    int K;
    float indexing_time = 0;
    std::string space_name;
    //size_t dim;
    //size_t seed;
    size_t default_ef;
    bool useNormFactor;

    bool index_inited;
    bool ep_added;
    bool normalize = 0;
    int num_threads_default;
    hnswlib::labeltype cur_l;
    IpSpace* l2space = nullptr;

    std::string alg_name = "napg";

    // myNADG(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable){
    // 	N = param_.N;
    // 	dim = param_.dim;
    // 	L = param_.L;
    // 	K = param_.K;
    // 	S = param_.S;
    // 	//prep = &prep_;
    // 	data = prep_.data;
    // 	//GetHash();
    // 	buildIndex();
    // }

    myNAPG(Data& data_, int M, int efC, int ef, const std::string& file, bool norm_adjusted_factor = true) {
        N = data_.N;
        dim = data_.dim;
        //L = param_.L;
        //K = param_.K;
        //S = param_.S;
        //prep = &prep_;
        data = data_;
        //GetHash();`
        //buildIndex();
        index_file = file;
        //
        l2space = new IpSpace(dim);
        appr_alg = nullptr;
        ep_added = true;
        index_inited = false;
        num_threads_default = std::thread::hardware_concurrency();

        default_ef = ef;
        useNormFactor = norm_adjusted_factor;

        //

        lsh::timer timer;
        bool isbuilt = 1;
        if(!(isbuilt && exists_test(index_file))) {
            init_new_index(N, M, efC);
            addItems();
            saveIndex(index_file);
            indexing_time = timer.elapsed();
            std::cout << "Actual memory usage: " << getCurrentRSS() / (1024 * 1024) << " Mb \n";
            std::cout << "DAPG Build time:" << indexing_time << "  seconds.\n";
            FILE* fp = nullptr;
            fopen_s(&fp, "./indexes/ipnsw_plus_info.txt", "a");
            if(fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), (float)getCurrentRSS() / (1024 * 1024), indexing_time);
        }
        else {

            std::cout << "Loading index from " << index_file << ":\n";
            float mem = (float)getCurrentRSS() / (1024 * 1024);
            loadIndex(index_file, N);
            appr_alg->ef_ = default_ef;
            float memf = (float)getCurrentRSS() / (1024 * 1024);
            std::cout << "Actual memory usage: " << memf - mem << " Mb \n";
        }



        //std::cout << "DAPG2 CONSTRUCTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
    }

    inline bool exists_test(const std::string& name) {
        //return false;
        std::ifstream f(name.c_str());
        return f.good();
    }

    void setEf(size_t ef) {
        appr_alg->setEf(ef);
    }

    int getM() {
        return appr_alg->maxM0_;
    }

    void init_new_index(const size_t maxElements, const size_t M, const size_t efConstruction)
    {
        if(appr_alg)
        {
            throw new std::runtime_error("The index is already initiated.");
        }
        cur_l = 0;
        const size_t seed = 100;
        //appr_alg = new napg(l2space, maxElements, M, efConstruction, seed, dim, useNormFactor);
        appr_alg = new napg(l2space, maxElements, M, efConstruction, seed, dim, 1);
        index_inited = true;
        ep_added = false;
        appr_alg->ef_ = default_ef;
        //seed = random_seed;
    }

    void set_num_threads(int num_threads)
    {
        this->num_threads_default = num_threads;
    }

    void saveIndex(const std::string& path_to_index)
    {
        appr_alg->saveIndex(path_to_index);
    }

    void loadIndex(const std::string& path_to_index, size_t max_elements)
    {
        if(appr_alg)
        {
            std::cerr << "Warning: Calling load_index for an already inited index. Old index is being deallocated.";
            delete appr_alg;
        }
        appr_alg = new napg(l2space, path_to_index, false, max_elements);
        cur_l = appr_alg->cur_element_count;
        index_inited = true;
    }

    void normalize_vector(float* data, float* norm_array)
    {
        float norm = 0.0f;
        for(size_t i = 0; i < dim; i++)
            norm += data[i] * data[i];
        norm = 1.0f / (sqrtf(norm) + 1e-30f);
        for(size_t i = 0; i < dim; i++)
            norm_array[i] = data[i] * norm;
    }

    void addItems(int num_threads = -1)
    {
        if(num_threads <= 0)
            num_threads = num_threads_default;

        size_t rows = data.N, features = data.dim;


        if(features != dim)
            throw std::runtime_error("wrong dimensionality of the vectors");

        // avoid using threads when the number of searches is small:
        if(rows <= num_threads * 4)
        {
            num_threads = 1;
        }

        std::vector<size_t> ids;

        {
            int start = 0;
            if(!ep_added)
            {
                size_t id = ids.size() ? ids.at(0) : (cur_l);
                float* vector_data = data[0];
                std::vector<float> norm_array(dim);
                if(normalize)
                {
                    normalize_vector(vector_data, norm_array.data());
                    vector_data = norm_array.data();
                }
                appr_alg->addPoint((float*)vector_data, (size_t)id);
                start = 1;
                ep_added = true;
            }

            if(useNormFactor) {
                // Add data to dataset and calculate the adjusting factors
                // for (int row = 0; row < rows; row++) {
                //     appr_alg->addData((float*)data[row], dim);
                // }
                int row = 0;
                ParallelFor(row, rows, num_threads, [&](size_t row, size_t threadId) {
                    //appr_alg->addData((float*)data[row], dim);
                    appr_alg->addData((float*)data[row], row);
                });

                // Calculate norm ranged based factors
                appr_alg->getNormRangeBasedFactors(data.val);
            }

            if(normalize == false)
            {
                lsh::progress_display pd(rows - 1);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId) {
                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((float*)data[row], (size_t)id);
                    ++pd;
                });
            }
            else
            {
                std::vector<float> norm_array(num_threads * dim);
                ParallelFor(start, rows, num_threads, [&](size_t row, size_t threadId)
                {
                    // normalize vector:
                    size_t start_idx = threadId * dim;
                    normalize_vector((float*)data[row], (norm_array.data() + start_idx));

                    size_t id = ids.size() ? ids.at(row) : (cur_l + row);
                    appr_alg->addPoint((float*)(norm_array.data() + start_idx), (size_t)id); });
            };
            cur_l += rows;
        }
    }



    void knn(queryN* q) {
        lsh::timer timer;
        timer.restart();
        int ef = appr_alg->ef_;
        //auto& appr_alg = appr_alg;
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

    ~myNAPG() {
        delete appr_alg;
    }
};