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

#include "Preprocess.h"
#include "alg.h"
#include "ht-nsw.h"
#include "normAdjustedDG.h"
#include "ipNSW.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;
std::unique_lock<std::mutex>* glock = nullptr;

int main(int argc, char const* argv[])
{
	std::string dataset = "audio2";
	int varied_n = 0;
	if(argc > 1) dataset = argv[1];
	if(argc > 2) varied_n = std::atoi(argv[2]);

	std::string argvStr[4];
	argvStr[1] = (dataset);

	argvStr[3] = (dataset + ".bench_graph");

	float c = 0.9f;
	int k = 50;
	int m, L, K;

	std::cout << "Using HT-NSW for " << argvStr[1] << std::endl;
	Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]), varied_n);
	std::vector<resOutput> res;
	m = 1000;
	L = 5;
	K = 12;
	c = 0.3;

	int minsize_cl = 500;
	int num_cl = 10;
	int max_mst_degree = 3;

	Parameter param(prep, L, K, 1);

	lsh::timer timer;
	Partition parti(c, prep);
	// std::cout << "Partition time: " << timer.elapsed() << " s.\n" << std::endl;

	if(varied_n > 0) dataset += std::to_string(varied_n);
	argvStr[2] = (dataset + ".index");


	enum { GREEDY_NAP = 0, HEURISTIC_NAP = 1 };

	//ht_nsw<HEURISTIC_NAP, 0> ht(prep, index_fold + (argvStr[2]) + "_ht");


	std::vector<int> efs = { 0,10,20,30,40,50,75,100,150,200,250,300,600,900,1200,1600,2000 };
	efs = { 50 };
	std::vector<float> deltas = { 0.1,0.01,0.001,0.0001,0.00001,0.000001 }; //default delta=0.01
	deltas = { 0.001 };
	for(auto& delta : deltas) {
		//if(1) continue;
		float a, b;
		ht_nsw<HEURISTIC_NAP, 1> ht(prep, index_fold + (argvStr[2]) + "_ht" + std::to_string((int)(log10(delta))), delta);
		ht.NAP_accuracy(prep, k, a, b);
		for(auto& ef : efs) {
			ht.setEf(ef);
			//ht.setEf(0);
			res.push_back(Alg0_maria(ht, c, 100, k, L, K, prep));
		}

	}

	if(0) {
		ipNSW hnsw(prep, param, index_fold + (argvStr[2]) + "_ipnsw", data_fold2 + "MyfunctionXTheta.data");
		for(auto& ef : efs) {
			hnsw.setEf(ef);
			res.push_back(Alg0_maria(hnsw, c, 100, k, L, K, prep));
		}
	}


	if(0) {
		myNAPG napg(prep.data, 24, 80, 1000, index_fold + (argvStr[2]) + "_napg");
		for(auto& ef : efs) {
			napg.setEf(ef);
			res.push_back(Alg0_maria(napg, c, 100, k, L, K, prep));
		}
		//res.push_back(Alg0_maria(napg, c, 100, k, L, K, prep));
	}

	saveAndShow(c, k, dataset, res);

	return 0;
}
