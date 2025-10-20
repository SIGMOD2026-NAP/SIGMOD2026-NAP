#include "Preprocess.h"
#include "basis.h"
#include <fstream>
#include <assert.h>
#include <random>
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <sstream>
#include <numeric>
#include<algorithm>

#define CANDIDATES 100
#define NUM_COURS 160
#define E 2.718281746
#define PI 3.1415926
#define MAXSIZE 40960

#define min(a,b)            (((a) < (b)) ? (a) : (b))

// Preprocess::Preprocess(const std::string& path, const std::string& ben_file_)
// {
// 	lsh::timer timer;
// 	std::cout << "LOADING DATA..." << std::endl;
// 	timer.restart();

// 	if (path.find("deep1B") != std::string::npos || path.find("yandex") != std::string::npos) {
// 		load_fbin(path + ".fbin", data);
// 		load_fbin(path + "_query.fbin", queries);
// 	}
// 	else load_data(path);



// 	std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
// 	cal_SquareLen();

// 	data_file = path;
// 	ben_file = ben_file_;
// 	ben_create();
// }

// void Preprocess::load_data(const std::string& path) {
// 	std::string file = path + ".data_new";
// 	std::ifstream in(file.c_str(), std::ios::binary);
// 	if (!in) {
// 		printf("Fail to open the file: %s\n", file.c_str());
// 		exit(-1);
// 	}

// 	unsigned int header[3] = {};
// 	assert(sizeof header == 3 * 4);
// 	in.read((char*)header, sizeof(header));
// 	assert(header[1] != 0);
// 	data.N = header[1] - 200;
// 	data.dim = header[2];

// 	queries.N = 200;
// 	queries.dim = data.dim;

// 	queries.val = new float* [queries.N];
// 	data.val = new float* [data.N];

// 	for (int i = 0; i < queries.N; ++i) {
// 		queries.val[i] = new float[queries.dim + 1];
// 		in.read((char*)queries.val[i], sizeof(float) * header[2]);
// 		queries.val[i][queries.dim] = 0.0f;
// 		//queries.N = 10000;
// 	}

// 	for (int i = 0; i < data.N; ++i) {
// 		data.val[i] = new float[data.dim + 1];
// 		in.read((char*)data.val[i], sizeof(float) * header[2]);
// 		data.val[i][data.dim] = 0.0f;
// 	}

// 	std::cout << "Load from new file: " << file << "\n";
// 	std::cout << "Nq =  " << queries.N << "\n";
// 	std::cout << "N  =  " << data.N << "\n";
// 	std::cout << "dim=  " << data.dim << "\n\n";

// 	in.close();
// }

// void Preprocess::load_fbin(const std::string& path, Data& data) {
// 	std::string file = path;
// 	std::ifstream in(file.c_str(), std::ios::binary);
// 	if (!in) {
// 		printf("Fail to open the file: %s\n", file.c_str());
// 		exit(-1);
// 	}

// 	unsigned int header[2] = {};
// 	assert(sizeof header == 2 * 4);
// 	in.read((char*)header, sizeof(header));
// 	assert(header[0] != 0);
// 	data.N = header[0];
// 	data.dim = header[1];

// 	size_t size = ((size_t)header[1]) * header[0];
// 	std::cout << "Load from fbin: " << file << "\n";
// 	std::cout << "N   =  " << data.N << "\n";
// 	std::cout << "dim =  " << data.dim << "\n";
// 	std::cout << "size=  " << size << "\n\n";

// 	data.base = new float[size];
// 	data.val = new float* [data.N];

// 	in.read((char*)data.base, sizeof(float) * size);

// 	for (size_t i = 0; i < data.N; ++i) {
// 		data.val[i] = data.base + i * data.dim;
// 	}

// 	std::cout << "Finish Reading File! " << "\n";
// 	in.close();
// }

void Preprocess::cal_SquareLen()
{
	if (data.base == nullptr) return;
	norms = new float[data.N];
	for (int i = 0; i < data.N; ++i) {
		norms[i] = sqrt(cal_inner_product(data.val[i], data.val[i], data.dim));
	}

	MaxLen = *std::max_element(norms, norms + data.N);
}

typedef struct Tuple
{
	int id;
	float inp;
	bool operator < (const Tuple& rhs) {
		return inp < rhs.inp;
	}
}Tuple;
bool comp(const Tuple& a, const Tuple& b)
{
	return a.inp > b.inp;
}

void Preprocess::ben_make()
{
	benchmark.N = 100, benchmark.num = 100;
	benchmark.indice = new int* [benchmark.N];
	benchmark.innerproduct = new float* [benchmark.N];
	for (int j = 0; j < benchmark.N; j++) {
		benchmark.indice[j] = new int[benchmark.num];
		benchmark.innerproduct[j] = new float[benchmark.num];
	}

	//Tuple a;

	lsh::progress_display pd(benchmark.N);

	// #pragma omp parallel for
	for (int j = 0; j < benchmark.N; j++) {
		std::vector<Tuple> dists(data.N);
		//dists.clear();

#pragma omp parallel for schedule(dynamic,512)
		for (int i = 0; i < data.N; i++) {
			dists[i].id = i;
			dists[i].inp = cal_inner_product(data.val[i], queries.val[j], data.dim);
			//dists[i]=a;
		}


		int nsort = data.N / NUM_COURS;//our cpu server has 160 cores.
#pragma omp parallel for schedule(dynamic,1)
		for (int i = 0;i < NUM_COURS;++i) {
			sort(dists.begin() + i * nsort, dists.begin() + (i + 1) * nsort, comp);
		}

		int nrest = benchmark.num * NUM_COURS + data.N - nsort * NUM_COURS;
		//std::vector<Tuple> dist_rest(nrest);
		for (int i = 1;i < NUM_COURS;++i) {
			memcpy(dists.data() + i * benchmark.num, dists.data() + i * nsort, sizeof(Tuple) * benchmark.num);
		}

		if (data.N - nsort * NUM_COURS > 0)
			memcpy(dists.data() + NUM_COURS * benchmark.num, dists.data() + NUM_COURS * nsort,
				sizeof(Tuple) * (data.N - nsort * NUM_COURS));

		dists.resize(nrest);

		sort(dists.begin(), dists.end(), comp);
		for (int i = 0; i < benchmark.num; i++) {
			benchmark.indice[j][i] = (int)dists[i].id;
			benchmark.innerproduct[j][i] = dists[i].inp;
		}
		++pd;
	}

}

void Preprocess::ben_save()
{
	std::ofstream out(ben_file.c_str(), std::ios::binary);
	out.write((char*)&benchmark.N, sizeof(int));
	out.write((char*)&benchmark.num, sizeof(int));

	for (int j = 0; j < benchmark.N; j++) {
		out.write((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
	}

	for (int j = 0; j < benchmark.N; j++) {
		out.write((char*)&benchmark.innerproduct[j][0], sizeof(float) * benchmark.num);
	}

	out.close();
}

void Preprocess::ben_load()
{
	std::ifstream in(ben_file.c_str(), std::ios::binary);
	in.read((char*)&benchmark.N, sizeof(int));
	in.read((char*)&benchmark.num, sizeof(int));

	benchmark.indice = new int* [benchmark.N];
	benchmark.innerproduct = new float* [benchmark.N];
	for (int j = 0; j < benchmark.N; j++) {
		benchmark.indice[j] = new int[benchmark.num];
		in.read((char*)&benchmark.indice[j][0], sizeof(int) * benchmark.num);
	}

	for (int j = 0; j < benchmark.N; j++) {
		benchmark.innerproduct[j] = new float[benchmark.num];
		in.read((char*)&benchmark.innerproduct[j][0], sizeof(float) * benchmark.num);
	}
	in.close();
}

void Preprocess::ben_create()
{
	int a_test = data.N + 1;
	lsh::timer timer;
	std::ifstream in(ben_file.c_str(), std::ios::binary);
	in.read((char*)&a_test, sizeof(int));
	in.close();
	if (a_test > 0 && a_test <= data.N)
	{
		std::cout << "LOADING BENMARK..." << std::endl;
		timer.restart();
		ben_load();
		std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
	}
	else
	{
		std::cout << "MAKING BENMARK..." << std::endl;
		timer.restart();
		ben_make();
		std::cout << "MAKING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

		std::cout << "SAVING BENMARK..." << std::endl;
		timer.restart();
		ben_save();
		std::cout << "SAVING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
	}
}

Preprocess::~Preprocess()
{
	//if (data.val) clear_2d_array(data.val, data.N);
	delete[] data.val;
	if (data.base) delete[] data.base;
	clear_2d_array(benchmark.indice, benchmark.N);
	clear_2d_array(benchmark.innerproduct, benchmark.N);
	delete[] norms;
}

Partition::Partition(float c_, float c0_, Preprocess& prep)
{
	ratio = (pow(c0_, 4.0f) - 1) / (pow(c0_, 4.0f) - c_);
	make_chunks_fargo(prep);
}

Partition::Partition(float c_, Preprocess& prep)
{
	ratio = 0.975;
	float c0_ = 1.5f;

	make_chunks_fargo(prep);
}



void Partition::make_chunks_fargo(Preprocess& prep)
{
	std::vector<Dist_id> distpairs;
	std::vector<int> bucket;
	//Dist_id pair;
	int N_ = prep.data.N;
	int cnt = 0;
	for (int j = 0; j < N_; j++) {
		distpairs.emplace_back(j, prep.norms[j]);
	}
	std::sort(distpairs.begin(), distpairs.end());

	numChunks = 0;
	chunks.resize(N_);
	int j = 0;
	while (j < N_) {
		float M = distpairs[j].dist / ratio;
		cnt = 0;
		bucket.clear();
		while (j < N_)
		{
			if ((distpairs[j].dist > M || cnt >= MAXSIZE)) {
				break;
			}

			chunks[distpairs[j].id] = numChunks;
			bucket.push_back(distpairs[j].id);
			j++;
			cnt++;
		}
		nums.push_back(cnt);
		MaxLen.push_back(distpairs[(size_t)j - 1].dist);
		EachParti.push_back(bucket);
		bucket.clear();
		numChunks++;
	}

	display();
}

void Partition::make_chunks_maria(Preprocess& prep)
{
	std::vector<Dist_id> distpairs;
	std::vector<int> bucket;
	//Dist_id pair;
	int N_ = prep.data.N;
	int n;
	for (int j = 0; j < N_; j++) {
		distpairs.emplace_back(j, prep.norms[j]);
	}
	std::sort(distpairs.begin(), distpairs.end());

	numChunks = 0;
	chunks.resize(N_);
	int j = 0;
	while (j < N_) {
		float M = distpairs[j].dist / ratio;
		n = 0;
		bucket.clear();
		while (j < N_) {
			if ((distpairs[j].dist > M || n >= MAXSIZE)) break;

			chunks[distpairs[j].id] = numChunks;
			bucket.push_back(distpairs[j].id);
			j++;
			n++;
		}
		nums.push_back(n);
		MaxLen.push_back(distpairs[(size_t)j - 1].dist);
		EachParti.push_back(bucket);
		bucket.clear();
		numChunks++;
	}

	display();
}

void Partition::display()
{
	std::vector<int> n_(numChunks, 0);
	int N_ = std::accumulate(nums.begin(), nums.end(), 0);
	for (int j = 0; j < N_; j++)
	{
		n_[chunks[j]]++;
	}
	bool f1 = false, f2 = false;
	for (int j = 0; j < numChunks; j++)
	{
		if (n_[j] != nums[j]) {
			f1 = true;
			break;
		}
	}

	std::cout << "This is the result of partition:"
		<< "\n Blocks       =" << numChunks
		<< "\n ratio_       =" << sqrt(ratio)
		<< "\n n_pts_       =" << N_ << std::endl;
}

Partition::~Partition()
{}

Parameter::Parameter(Preprocess& prep, int L_, int K_, int M_)
{
	N = prep.data.N;
	dim = prep.data.dim;
	L = L_;
	K = K_;
	S = L * K;
	MaxSize = 3;
	KeyLen = 1;
}

Parameter::Parameter(Preprocess& prep, int L_, int K_, int M_, float U_)
{
	N = prep.data.N;
	dim = prep.data.dim;
	M = M_;
	L = L_;
	K = K_;
	S = L * K;
	U = U_;
}

Parameter::Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_)
{
	N = prep.data.N;
	dim = prep.data.dim;
	L = L_;
	K = K_;
	S = L * K;
	U = U_;
	W = W_;
}

Parameter::Parameter(Preprocess& prep, float c_, float S0)
{

	N = prep.data.N;
	dim = prep.data.dim;
	assert(c_ * S0 < 1);
	double pi = atan(1) * 4;
	double p1 = 1 - acos(S0) / pi;
	double p2 = 1 - acos(c_ * S0) / pi;
	K = (int)floor(log(N) / log(1 / p2)) + 1;
	double rho = log(p1) / log(p2);
	L = (int)floor(pow((double)N, rho)) + 1;
	S = L * K;

}

inline float normal_pdf0(			// pdf of Guassian(mean, std)
	float x,							// variable
	float u,							// mean
	float sigma)						// standard error
{
	float ret = exp(-(x - u) * (x - u) / (2.0f * sigma * sigma));
	ret /= sigma * sqrt(2.0f * PI);
	return ret;
}

float new_cdf0(						// cdf of N(0, 1) in range [-x, x]
	float x,							// integral border
	float step)							// step increment
{
	float result = 0.0f;
	for (float i = -x; i <= x; i += step) {
		result += step * normal_pdf0(i, 0.0f, 1.0f);
	}
	return result;
}

inline float calc_p0(			// calc probability
	float x)							// x = w / (2.0 * r)
{
	return new_cdf0(x, 0.001f);		// cdf of [-x, x]
}

Parameter::Parameter(Preprocess& prep, float c_)
{
	K = 0;
	L = 0;
	N = prep.data.N;
	dim = prep.data.dim;

	KeyLen = -1;
	MaxSize = -1;
	float w_ = sqrt((8.0f * c_ * c_ * log(c_)) / (c_ * c_ - 1.0f));

	float beta_;
	float delta_;
	float p1_;
	float p2_;
	float para1;
	float para2;
	float para3;
	float eta;
	float alpha_;
	float m, l;

	int n_pts_ = MAXSIZE;
	beta_ = (float)CANDIDATES / n_pts_;
	delta_ = 1.0f / E;

	p1_ = calc_p0(w_ / 2.0f);
	p2_ = calc_p0(w_ / (2.0f * c_));

	para1 = sqrt(log(2.0f / beta_));
	para2 = sqrt(log(1.0f / delta_));
	para3 = 2.0f * (p1_ - p2_) * (p1_ - p2_);
	eta = para1 / para2;

	alpha_ = (eta * p1_ + p2_) / (1.0f + eta);
	m = (para1 + para2) * (para1 + para2) / para3;
	this->S = (int)ceil(m);
	M = 1;
	W = 1;
}


bool Parameter::operator = (const Parameter& rhs)
{
	bool flag = 1;
	flag *= (L == rhs.L);
	flag *= (K = rhs.K);
	return flag;
}

Parameter::~Parameter()
{}