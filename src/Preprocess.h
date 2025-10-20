#pragma once
#include "StructType.h"
#include "basis.h"
#include <cmath>
#include <assert.h>
#include <string>
#include <vector>
#include <queue>
#include <cfloat>
#include <fstream>

class Preprocess
{
	public:
	Data data;
	Data queries;
	float* norms = nullptr;
	Ben benchmark;
	float MaxLen;
	std::string data_file;
	std::string ben_file;
	public:
	// Preprocess(const std::string& path, const std::string& ben_file_);
	// void load_data(const std::string& path);
	// void load_fbin(const std::string& path, Data& data);

	Preprocess(const std::string& path, const std::string& ben_file_, int varied_n = 0) {
		//for_varied_n = varied_n;
		lsh::timer timer;
		std::cout << "LOADING DATA..." << std::endl;
		timer.restart();
		if(path.find("deep1B") != std::string::npos || path.find("yandex") != std::string::npos) {
			load_fbin(path + "_query.fbin", queries);
			load_fbin(path + ".fbin", data, varied_n);

		}
		else load_data(path, varied_n);
		std::cout << "LOADING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;
		cal_SquareLen();

		data_file = path;
		ben_file = ben_file_;
		if(varied_n > 0) ben_file += std::to_string(varied_n);
		ben_create();

		count_rank();
	}

	std::vector<float> rank;
	std::vector<std::pair<float, int>> len;
	void count_rank(){
		int count = 0;
		float thred = 0.5;
		float rank_sum = 0.0f;

		for(int i = 0;i < data.N;++i) len.emplace_back(norms[i], i);
		std::sort(len.begin(), len.end());
		rank.resize(data.N, 0);
		for(int i = 0;i < data.N;++i) rank[len[i].second] = float(i) / data.N;

		float min_rank = 1.0f, max_rank = 0.0f, avg_rank = 0.0f;
		//for(int i = 0;i < benchmark.N; ++i){	
		for(int i = 0; i < benchmark.N; ++i){
			for(int j = 0; j < benchmark.num; ++j){
				if(rank[benchmark.indice[i][j]] < thred) count++;
				rank_sum += rank[benchmark.indice[i][j]];
				if(rank[benchmark.indice[i][j]] < min_rank) min_rank = rank[benchmark.indice[i][j]];
				if(rank[benchmark.indice[i][j]] > max_rank) max_rank = rank[benchmark.indice[i][j]];
			}
		}
		avg_rank = rank_sum / (benchmark.N * benchmark.num);
		float recall = float(count) / (benchmark.N * benchmark.num);
		std::cout << "Rank Thred: " << thred << ", Count: "
			<< count << ", Recall: " << recall << ", min_rank: " << min_rank << ", max_rank: "
			<< max_rank << ", avg_rank: " << avg_rank << std::endl;

		//exit(0);

	}

	void load_data(const std::string& path, int varied_n = 0) {
		std::string file = path + ".data_new";
		std::ifstream in(file.c_str(), std::ios::binary);
		if(!in) {
			printf("Fail to open the file!\n");
			exit(-1);
		}


		unsigned int header[3] = {};
		assert(sizeof header == 3 * 4);
		in.read((char*)header, sizeof(header));
		assert(header[1] != 0);
		data.N = header[1] - 200;
		data.dim = header[2];

		while(varied_n > 0) {
			data.N /= 10;
			varied_n--;
		}

		queries.N = 200;
		queries.dim = data.dim;

		std::cout << "Load from new file: " << file << "\n";
		std::cout << "Nq =  " << queries.N << "\n";
		std::cout << "N  =  " << data.N << "\n";
		std::cout << "dim=  " << data.dim << "\n\n";

		queries.val = new float* [queries.N];
		data.val = new float* [data.N];

		//data.offset=data.dim+1;
		data.base = new float[(size_t)data.N * data.dim];
		queries.base = new float[(size_t)queries.N * queries.dim];

		for(size_t i = 0; i < queries.N; ++i) {
			// queries.val[i] = new float[queries.dim + 1];
			// in.read((char*)queries.val[i], sizeof(float) * header[2]);
			// queries.val[i][queries.dim - 1] = 0.0f;

			queries.val[i] = queries.base + i * queries.dim;
			in.read((char*)queries.val[i], sizeof(float) * header[2]);
		}

		for(size_t i = 0; i < data.N; ++i) {
			// data.val[i] = new float[data.dim + 1];
			// in.read((char*)data.val[i], sizeof(float) * header[2]);
			// data.val[i][data.dim - 1] = 0.0f;

			data.val[i] = data.base + i * data.dim;
			in.read((char*)data.val[i], sizeof(float) * header[2]);
		}

		std::cout << "Finish loading! " << "\n";

		in.close();
	}

	void load_fbin(const std::string& path, Data& data, int varied_n = 0) {
		std::string file = path;
		std::ifstream in(file.c_str(), std::ios::binary);
		if(!in) {
			printf("Fail to open the file: %s\n", file.c_str());
			exit(-1);
		}

		unsigned int header[2] = {};
		assert(sizeof header == 2 * 4);
		in.read((char*)header, sizeof(header));
		assert(header[0] != 0);
		data.N = header[0];
		data.dim = header[1];

		while(varied_n > 0) {
			data.N /= 10;
			varied_n--;
		}
		size_t size = ((size_t)data.N) * data.dim;
		std::cout << "Load from fbin: " << file << "\n";
		std::cout << "N   =  " << data.N << "\n";
		std::cout << "dim =  " << data.dim << "\n";
		std::cout << "size=  " << size << "\n\n";

		if(data.N >= 1e9) return;

		data.base = new float[size];
		data.val = new float* [data.N];
		for(size_t i = 0; i < data.N; ++i) {
			data.val[i] = data.base + i * data.dim;
		}

		in.read((char*)data.base, sizeof(float) * size);
		// #pragma omp parallel for schedule(dynamic, 256)
		// 		for (int i = 0; i < data.N; ++i) {
		// 			in.read((char*)data.val[i], sizeof(float) * data.dim);
		// 		}

		std::cout << "Finish Reading File! " << "\n";
		in.close();
	}


	void cal_SquareLen();
	void ben_make();
	void ben_save();
	void ben_load();
	void ben_create();
	~Preprocess();
};

struct Dist_id
{
	int id = -1;
	float dist = 0.0f;
	//Dist_id() = default;
	Dist_id(int id_, float dist_) :id(id_), dist(dist_) {}
	bool operator < (const Dist_id& rhs) {
		return dist < rhs.dist;
	}
};

class Partition
{
	private:
	float ratio;
	void make_chunks_fargo(Preprocess& prep);
	void make_chunks_maria(Preprocess& prep);
	public:
	int numChunks;
	std::vector<float> MaxLen;

	//The chunk where each point belongs
	//chunks[i]=j: i-th point is in j-th parti
	std::vector<int> chunks;

	//The data size of each chunks
	//nums[i]=j: i-th parti has j points
	std::vector<int> nums;

	//The buckets by parti;
	//EachParti[i][j]=k: k-th point is the j-th point in i-th parti
	std::vector<std::vector<int>> EachParti;

	//std::vector<Dist_id> distpairs;
	void display();

	Partition(float c_, Preprocess& prep);

	Partition(float c_, float c0_, Preprocess& prep);
	//Partition() {}
	~Partition();
};

class Parameter //N,dim,S, L, K, M, W;
{
	public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;
	//
	int MaxSize = -1;
	//
	int KeyLen = -1;

	int M = 1;

	int W = 0;

	float U;
	Parameter(Preprocess& prep, int L_, int K_, int M);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_);
	Parameter(Preprocess& prep, float c_, float S0);
	Parameter(Preprocess& prep, float c0_);
	bool operator = (const Parameter& rhs);
	~Parameter();
};

struct Res//the result of knns
{
	//dist can be:
	//1. L2-distance
	//2. The opposite number of inner product
	float dist = 0.0f;
	int id = -1;
	Res() = default;
	Res(int id_, float inp_) :id(id_), dist(inp_) {}
	bool operator < (const Res& rhs) const {
		return dist < rhs.dist;
	}

	bool operator > (const Res& rhs) const {
		return dist > rhs.dist;
	}
};

class queryN
{
	public:
	// the parameter "c" in "c-ANN"
	float c;
	//which chunk is accessed
	//int chunks;

	//float R_min = 4500.0f;//mnist
	//float R_min = 1.0f;
	float init_w = 1.0f;

	float* queryPoint = NULL;
	unsigned* hashval = NULL;
	//float** myData = NULL;
	int dim = 1;

	int UB = 0;
	float minKdist = FLT_MAX;
	// Set of points sifted
	std::priority_queue<Res> resHeap;
	std::priority_queue<Res> resCos;

	public:
	// k-NN
	unsigned k = 1;
	// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
	unsigned qid = -1;

	float beta = 0;
	float norm = 0.0f;
	unsigned cost = 0;

	//#access;
	int maxHop = -1;
	//
	unsigned prunings = 0;
	//cost of each partition
	std::vector<int> costs;
	//
	float time_total = 0;
	//
	float timeHash = 0;
	//
	float time_sift = 0;

	float time_verify = 0;
	// query result:<indice of ANN,distance of ANN>
	std::vector<Res> res;

	public:
	queryN(unsigned id, float c_, unsigned k_, Preprocess& prep, float beta_) {
		qid = id;
		c = c_;
		k = k_;
		beta = beta_;
		//myData = prep.data.val;
		dim = prep.data.dim + 1;
		queryPoint = prep.queries[id];
		// queryPoint = new float[dim];
		// memcpy(queryPoint, prep.queries[id], sizeof(float) * dim);

		norm = sqrt(cal_inner_product(queryPoint, queryPoint, dim));
		//search();
	}

	//void search();

	~queryN() {
		delete hashval;
		//delete queryPoint;
	}
};