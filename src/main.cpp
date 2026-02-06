#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>
#include <omp.h>
#include "Preprocess.h"
#include "alg.h"
#include "maria.h"
#include "nbgraph.h"
#include "nbgraph_new.h"
#include "ht-nsw.h"
#include "normAdjustedDG.h"

#include "hnsw_faiss.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;
std::unique_lock<std::mutex>* glock = nullptr;

int main(int argc, char const* argv[])
{

	//set the defualt parameters for index construction and search. 
	//You can change them by passing command line arguments when running the program. 
	//For example, you can run "./fg mnist 8000 5 10 16 80 160" to set dataset=mnist, varied_n=8000, L=5, K=10, M=16, efC=80, and number of threads for OpenMP to 160.
	std::string dataset = "audio";
	int m, L, K;
	int M = 16;
	int efC = 80;
	int varied_n = 0;
	L = 5;
	K = 10;
	efC = 80;
	float delta = 0.001f;

	if (argc > 1) dataset = argv[1];
	if (argc > 2) varied_n = std::atoi(argv[2]); // set the varied_n for index construction, which is the percentage of data used for index construction in the case of varied n. For example, if varied_n=5000, it means using 50% of data for index construction.
	if (argc > 3) L = std::atoi(argv[3]); // set L for index construction, which is the number of hash tables for SRP in HT-NSW
	if (argc > 4) K = std::atoi(argv[4]); // set K for index construction, which is the number of hash functions for SRP in HT-NSW
	if (argc > 5) M = std::atoi(argv[5]); // set M for index construction
	if (argc > 6) efC = std::atoi(argv[6]); // set efC for index construction
	if (argc > 7) omp_set_num_threads(std::atoi(argv[7])); // set the number of threads for OpenMP
	if (argc > 8) delta = std::stof(argv[8]);

	std::cout << "Dataset: " << dataset << ", L=" << L << ", K=" << K << ", M=" << M << ", efC=" << efC << std::endl;
	std::cout << "Number of threads: " << omp_get_max_threads() << std::endl << std::endl;

	std::string argvStr[4];
	argvStr[1] = (dataset);

	argvStr[3] = (dataset + ".bench_graph");

	float c = 0.9f;
	int k = 10;

	std::cout << "Using HT-NSW for " << argvStr[1] << std::endl;
	Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]), varied_n);
	std::vector<resOutput> res;
	m = 1000;
	c = 0.3;

	int minsize_cl = 500;
	int num_cl = 10;
	int max_mst_degree = 3;

	Parameter param(prep, L, K, 1);

	lsh::timer timer;
	Partition parti(c, prep);

	if (varied_n > 0) dataset += std::to_string(varied_n) + "_of10000";
	argvStr[2] = (dataset + ".index");

	enum { GREEDY_NAP = 0, HEURISTIC_NAP = 1 };

	bool run_ht_nsw = 1;
	bool run_ipnsw = 0;
	bool run_napg = 0;
	bool run_hnswfaiss = 0;

	std::vector<int> efs = { 0,10,20,30,40,50,75,100,150,200,250,300,600,900,1200,1600,2000 };
	std::vector<float> deltaLs = { 0.5,0.4,0.3,0.2,0.1,0.05,0.01 };
	efs = { 500 };
	if (run_ht_nsw) {
		ht_nsw<HEURISTIC_NAP, 1> ht(prep, index_fold + (argvStr[2]) + "_ht" + std::to_string((int)(log10(delta))), L, K, efC, delta, M);
		ht.setEf(10);
		for (auto& d : deltaLs) {
			ht.srp->updateTheta(d);
			res.push_back(search_omp(ht, c, 100, k, L, K, prep));
		}

		for (auto& ef : efs) {
			ht.setEf(ef);
			res.push_back(search_omp(ht, c, 100, k, L, K, prep));
		}

	}

	if (run_ipnsw) {
		ipNSW hnsw(prep, param, index_fold + (argvStr[2]) + "_ipnsw", M);

		for (auto& ef : efs) {
			hnsw.setEf(ef);
			res.push_back(search_omp(hnsw, c, 100, k, L, K, prep));
		}
	}

	if (run_napg) {
		myNAPG napg(prep.data, M, 80, 1000, index_fold + (argvStr[2]) + "_napg");
		for (auto& ef : efs) {
			napg.setEf(ef);
			napg.set_num_threads(omp_get_max_threads());
			res.push_back(search_omp(napg, c, 100, k, L, K, prep));
		}
	}

	if (run_hnswfaiss) {
		hnsw_Faiss hnsw_faiss(prep.data, param, index_fold + (argvStr[2]) + "_hnswfaiss", M);
		for (auto& ef : efs) {
			hnsw_faiss.setEf(ef + k);
			res.push_back(search_Faiss(hnsw_faiss, c, 100, k, L, K, prep));
		}
	}

	saveAndShow(c, k, dataset, res);

	return 0;
}
