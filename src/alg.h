#pragma once
#include <string>
#include "Preprocess.h"
//#include "mf_alsh.h"
#include "performance.h"
#include "basis.h"
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

extern std::atomic<size_t> _G_COST;

struct resOutput{
	std::string algName;
	int L;
	int K;
	float c;
	float delta;
	float time;
	float recall;
	float ratio;
	float cost;
	float kRatio;
	float qps;
};

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

template <typename mariaVx>
inline resOutput Alg0_maria(mariaVx& maria, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;

	Performance<queryN> perform;

	int t = 160;
	t = 1;

	size_t cost1 = _G_COST;
	int nq = t * Qnum;

	std::vector<queryN> qs;
	// #pragma omp parallel for schedule(dynamic)
	for(int j = 0; j < nq; j++) {
		qs.emplace_back(j / t, c_, k_, prep, m_);
	}

	lsh::progress_display pd(nq);

	lsh::timer timer1;

	// #pragma omp parallel for //schedule(dynamic,256)
	for(int j = 0; j < nq; j++) {
		maria.knn(&(qs[j]));
		++pd;
	}
	float qt = (float)(timer1.elapsed() * 1000);


	for(int j = 0; j < nq; j++) {
		perform.update(qs[j], prep);
	}

	std::cout << "Query Time= " << (float)(timer1.elapsed() * 1000) << " ms." << std::endl;
	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);


	cost1 = _G_COST - cost1;

	resOutput res;
	res.algName = maria.alg_name;
	if(res.algName.find("ht") != std::string::npos) res.delta = maria.delta_thred;
	res.L = -1;
	res.K = m_;
	res.c = c_;
	res.qps = (float)nq / (qt / 1000);
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	//res.cost = ((float)cost1) / ((long long)perform.num * (long long)maria.N);
	res.cost = ((float)cost1) / ((long long)perform.num);
	res.kRatio = perform.kRatio / perform.num;
	//delete[] ltm;
	return res;
}



void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res);
