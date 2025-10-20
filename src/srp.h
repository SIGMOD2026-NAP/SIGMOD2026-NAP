#pragma once
#include "StructType.h"
#include "Preprocess.h"
#include <cmath>
#include <math.h>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>
#include <random>
#include <unordered_map>
#if defined(__GNUC__) && defined(USE_BLAS)
#include <cblas.h>
#endif

//#define USE_SHIFT

//#if undefined(M_PI)
//#define M_PI       3.14159265358979323846
//#endif

namespace lsh
{
	//The prority_queue only for building APG
	struct  fixxed_length_priority_queue
	{
		private:
		int size_ = 0;
		int capacity = 0;
		Res* data_ = nullptr;

		public:
		fixxed_length_priority_queue() = default;
		// fixxed_length_priority_queue(int K) {
		//     reset(K);
		// }

		fixxed_length_priority_queue(int K, Res* data) {
			capacity = K;
			data_ = data;
		}

		Res& operator[](int i) { return data_[i]; }
		// void reset(int K) {
		//     //capacity = K;
		//     delete[] data_;
		//     data_ = new Res[capacity];
		// }

		inline void emplace(int id, float dist) {
			if(size_ == capacity) {
				if(dist > data_[0].dist) return;
				pop();
				data_[size_] = Res(id, dist);
				std::push_heap(data_, data_ + size_);
			}
			data_[size_++] = Res(id, dist);
			std::push_heap(data_, data_ + size_);
		}

		void push(Res res) {
			if(size_ == capacity) {
				if(res.dist > data_[0].dist) return;
				pop();
				data_[size_] = res;
				std::push_heap(data_, data_ + size_);
			}
			data_[size_++] = res;
			std::push_heap(data_, data_ + size_);
		}

		void emplace_with_duplication(int id, float dist) {
			for(int i = 0;i < size_;++i) {
				if(data_[i].id == id) return;
			}
			emplace(id, dist);
		}

		void pop() {
			std::pop_heap(data_, data_ + size_);
			size_--;
		}

		int size() {
			return size_;
		}

		bool empty() {
			return size_ == 0;
		}

		Res& top() {
			return data_[0];
		}

		Res*& data() {
			return data_;
		}

		~fixxed_length_priority_queue() {}
	};

	struct srpPair {
		int id = -1;
		uint16_t val = 0;
		srpPair() = default;
		srpPair(int id_, uint16_t hashval) : id(id_), val(hashval) {}

		bool operator<(const srpPair& rhs) const { return val < rhs.val; }
	};

	struct hash_t {
		void reset(int N_) {
			delete[] base;
			N = N_;
			base = new uint16_t[N * 4];
			memset(base, 0, sizeof(uint16_t) * N * 4);
		}

		uint16_t* operator[](size_t i) { return base + i * 4; }
		size_t size() const { return N; }

		~hash_t() {
			delete[] base;
		}
		private:
		uint16_t* base = nullptr;
		size_t N = 0;
		//const int L = 4;
	};

	//struct hash_tnap {
	//	void reset(int N,int L) {
	//		base.resize(L, std::vector<int>(N, 0));
	//	}

	//	int operator[](size_t id, int l) { return base[l][id]; }
	//	//size_t size() const { return N; }

	//	~hash_tnap() {
	//		//delete[] base;
	//	}
	//private:
	//	std::vector<std::vector<int>> base;
	//	//size_t N = 0;
	//	//const int L = 4;
	//};

	using hash_tnap = std::vector<std::vector<int>>;

	// My implement for a simple sign random prejection LSH function class
	class srp
	{
		// int N=0;

		//N * L;
		std::vector<std::vector<std::vector<int>>> hash_tables;
		//std::vector<std::vector<int>>& part_map;
		Data data;
		std::string index_file;
		std::atomic<size_t> cost { 0 };
		float* rndAs = nullptr;
		int dim = 0;
		// Number of hash functions
		int S = 0;
		// #L Tables;
		int L = 0;
		// Dimension of the hash table
		int K = 0;
		float indexing_time = 0.0f;
		public:
		const std::string alg_name = "hash";
		//std::vector<std::vector<uint16_t>> hashvals;
		//std::vector<uint16_t[4]> hashvals;
		std::vector<std::vector<float>> hash_xc;
		std::vector<float> xc;
		std::vector<std::pair<float, int>> norms;
		std::vector<std::vector<unsigned>> masks;
		std::vector<float> thetas;
		float delta = 0.05f;
		int m;

		hash_tnap hashvals;
		size_t getCost() {
			return cost;
		}

		srp() = default;

		srp(Data& data_, const std::string& index_file_,
			std::vector<std::pair<float, int>>& norm_pairs, int m_, int L_ = 4, int K_ = 16, bool isbuilt = 1)
		{
			data = data_;
			// N=N_;
			dim = data.dim;
			L = L_;
			K = K_;
			S = L * K;
			hashvals.resize(L, std::vector<int>(data.N, 0));
			norms = norm_pairs;
			m = m_;
			index_file = index_file_;
			// if(L > 4 || K > 16) {
			// 	std::cerr << "The valid ranges of L and K are: 1<=L<=4, 1<=K<=16" << std::endl;
			// 	exit(-1);
			// }

			//std::ifstream in(index_file, std::ios::binary);
			lsh::timer timer;

			isbuilt = 1;
			if(!(isbuilt && exists_test(index_file))) {
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				buildIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				indexing_time = timer.elapsed();
				std::cout << "SRP Building time:" << indexing_time << "  seconds.\n";


				// FILE* fp = nullptr;
				// fopen_s(&fp, "./indexes/maria_info.txt", "a");
				// if (fp) fprintf(fp, "%s\nmemory=%f MB, IndexingTime=%f s.\n\n", index_file.c_str(), memf - mem, indexing_time);

				saveIndex();
			}
			else {
				//in.close();
				std::cout << "Loading hash tables from " << index_file << ":\n";
				float mem = (float)getCurrentRSS() / (1024 * 1024);
				loadIndex();
				float memf = (float)getCurrentRSS() / (1024 * 1024);
				std::cout << "Actual memory usage: " << memf - mem << " Mb \n";

			}

			updateTheta();
			masks.resize(K + 1);
			masks[0].emplace_back(0);
			for(int j = 1; j <= K; ++j) findNumbersWithJOnes(K, j);

			// std::cout << "show masks!" << std::endl;
			// for(int j = 0; j <= K; ++j) {
			// 	std::cout << "j=" << j << " , size=" << masks[j].size() << std::endl;
			// 	for(auto& v : masks[j]) {
			// 		std::cout << v << '\t';
			// 	}
			// 	std::cout << std::endl;
			// }
		}

		srp(Data& data_, int L_ = 5, int K_ = 10){
			data = data_;
			// N=N_;
			dim = data.dim;
			L = L_;
			K = K_;
			S = L * K;
		}

		void GetCentertHash() {
			xc.resize(dim, 0.0f);
			for(int i = 0; i < data.N; ++i) {
				for(int j = 0; j < data.dim; ++j) {
					xc[j] += data[i][j];
				}
			}

			for(int j = 0; j < data.dim; ++j) {
				xc[j] /= data.N;
			}

			hash_xc.resize(L, std::vector<float>(K, 0.0f));
			for(int j = 0; j < L; ++j) {
				for(int l = 0; l < K; ++l) {
					hash_xc[j][l] = cal_inner_product(xc.data(), rndAs + (j * K + l) * dim, dim);
				}
			}
		}

		void buildIndex() {
			std::cout << std::endl
				<< "START HASHING..." << std::endl
				<< std::endl;
			lsh::timer timer;

			std::cout << "SETTING HASH PARAMETER..." << std::endl;
			timer.restart();
			SetHash();
			std::cout << "SETTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "COMPUTING HASH..." << std::endl;
			timer.restart();
			GetHash(data);
			std::cout << "COMPUTING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;

			std::cout << "BUILDING INDEX..." << std::endl;
			std::cout << "THERE ARE " << L << " " << K << "-D HASH TABLES." << std::endl;
			timer.restart();

			GetTables();

			std::cout << "BUILDING TIME: " << timer.elapsed() << "s." << std::endl
				<< std::endl;
		}

		void SetHash()
		{
			rndAs = new float[S * dim];
			// hashpar.rndAs2 = new float* [S];

			std::mt19937 rng(int(std::time(0)));
			// std::mt19937 rng(int(0));
			std::normal_distribution<float> nd;
			for(int i = 0; i < S * dim; ++i)
				rndAs[i] = (nd(rng));
		}

		void GetHash(Data& data)
		{
#if defined(__GNUC__) && defined(USE_BLAS)
			int m = hashvals.size();
			int k = dim;
			int n = S;

			float* A = data.base;
			float* B = rndAs;
			float* C = new float[m * n];

			memset(C, 0.0f, m * n * sizeof(float));
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				m, n, k, 1.0, A, k, B, k, 0.0, C, n);

			for(int i = 0; i < hashvals.size(); ++i)
			{
				hashvals[i].resize(L, 0);
				for(int j = 0; j < L; ++j)
				{
					for(int l = 0; l < K; ++l)
					{
						float val = C[i * S + j * K + l];
						// cal_inner_product(data[i],rndAs+(j*K+l)*dim,dim);
						if(val > 0)
							hashvals[i][j] |= (1 << l);
					}
				}
			}
#else

#pragma omp parallel for schedule(dynamic, 256)
			for(int j = 0; j < L; ++j) {
				for(int i = 0; i < data.N; ++i)
				{
					for(int l = 0; l < K; ++l){
						float val = cal_inner_product(data[i], rndAs + (j * K + l) * dim, dim);
#ifdef USE_SHIFT
						if(val > hash_xc[j][l]) hashvals[j][i] |= (1 << l);
#else
						if(val > 0) hashvals[j][i] |= (1 << l);
#endif // 


						//
					}
				}
			}
#endif

			// for(int i=0;i<10;++i){
			// 	for(int j=0;j<L;++j){
			// 		std::cout<<hashvals[i][j]<<" ";

			// 	}
			// 	std::cout<<std::endl;
			// }
		}

		void GetTables(){
			hash_tables.resize(L);
			for(int j = 0; j < L; ++j) hash_tables[j].resize(1 << K);
			for(int i = 0; i < data.N; ++i){
				int id = i;
				for(int j = 0; j < L; ++j){
					//hash_tables[j].emplace_back(id, hashvals[id][j]);
					hash_tables[j][hashvals[j][id]].emplace_back(id);
				}
			}
		}

		void saveIndex() {

			std::string file = index_file;
			std::ofstream out(file, std::ios::binary);

			out.write((char*)(&L), sizeof(int));
			out.write((char*)(&K), sizeof(int));
			out.write((char*)(&dim), sizeof(int));
			S = L * K;

			//save hashpar
			out.write((char*)(rndAs), sizeof(float) * S * dim);

			//save hashvals
			int N = hashvals[0].size();
			out.write((char*)(&N), sizeof(int));
			for(int j = 0; j < L; ++j) {
				out.write((char*)(hashvals[j].data()), sizeof(int) * N);
				//for (int i = 0; i < N; ++i) {
				//	int size = hash_tables[j][i].size();
				//	out.write((char*)(&size), sizeof(int));
				//	out.write((char*)(hash_tables[j].data()), sizeof(int) * size);
				//}
			}

			//save hash tables
			for(int j = 0; j < L; ++j) {
				for(int i = 0;i < hash_tables[j].size();++i){
					int size = hash_tables[j][i].size();
					out.write((char*)(&size), sizeof(int));
					if(size > 0)out.write((char*)(hash_tables[j][i].data()), sizeof(int) * size);
				}
			}
		}

		void loadIndex() {

			std::string file = index_file;
			std::ifstream in(file, std::ios::binary);

			in.read((char*)(&L), sizeof(int));
			in.read((char*)(&K), sizeof(int));
			in.read((char*)(&dim), sizeof(int));
			S = L * K;

			//load hashpar
			rndAs = new float[S * dim];
			in.read((char*)(rndAs), sizeof(float) * S * dim);

			//load hashvals
			int N = 0;
			in.read((char*)(&N), sizeof(int));
			hashvals.resize(L);
			//out.write((char*)(&N), sizeof(int));
			for(int j = 0; j < L; ++j) {
				hashvals[j].resize(N);
				in.read((char*)(hashvals[j].data()), sizeof(int) * N);
				//for (int i = 0; i < N; ++i) {
				//	int size = hash_tables[j][i].size();
				//	out.write((char*)(&size), sizeof(int));
				//	out.write((char*)(hash_tables[j].data()), sizeof(int) * size);
				//}
			}

			//load hash tables
			hash_tables.resize(L);
			for(int j = 0; j < L; ++j) {
				hash_tables[j].resize(1 << K);
				for(int i = 0;i < hash_tables[j].size();++i){
					int size = 0;
					in.read((char*)(&size), sizeof(int));
					hash_tables[j][i].resize(size);
					in.read((char*)(hash_tables[j][i].data()), sizeof(int) * size);
				}
				// int np = 0;
				// in.read((char*)(&np), sizeof(int));
				// hash_tables[j].resize(np);
				// in.read((char*)(hash_tables[j].data()), sizeof(srpPair) * np);
			}
		}

		// std::vector<std::vector<std::vector<int>>> comb(int K){
		// 	return res;
		// }

		void findNumbersWithJOnes(int K, int j) {
			// Initialize the first combination with j ones at the least significant positions
			int combination = (1 << j) - 1;
			int limit = 1 << K; // This is 2^K, the upper limit of K-bit numbers

			while(combination < limit) {
				// Print the current combination
				//std::cout << combination << " (binary: " << std::bitset<32>(combination) << ")" << std::endl;

				masks[j].emplace_back(combination);

				// Gosper's hack to find the next combination with the same number of 1s
				int x = combination & -combination;
				int y = combination + x;
				combination = (((combination & ~y) / x) >> 1) | y;
			}
		}

		void calQHash(queryN* q) {
			//hashvals[i].resize(L, 0);
			//auto& vals = q->srpval;
			q->hashval = new unsigned[L];
			for(int j = 0; j < L; ++j) q->hashval[j] = 0;
			for(int j = 0; j < L; ++j){
				for(int l = 0; l < K; ++l)
				{
					float val = cal_inner_product(q->queryPoint, rndAs + (j * K + l) * dim, dim);
					if(val > 0)
						q->hashval[j] |= (1 << l);
				}
			}
		}

		// Lemma6.2 Eq.8
		double function_p(int j, float theta){
			double pi = 3.14159265358979323846;
			double p1 = (1.0 - theta / pi), p2 = theta / pi;
			int coeff = 1;
			double res = 0.0;
			for(int i = 0;i <= j;++i){
				res += coeff * pow(p1, K - i) * pow(p2, i);
				coeff = coeff * (K - i) / (i + 1);
			}
			return 1.0 - res;
		}

		void updateTheta(){
			double pi = 3.14159265358979323846;
			float thred = pow(delta, 1.0f / L);
			int pos = 0;
			for(float theta = 0.001f;theta <= pi;theta += 0.001f){
				double f = function_p(pos, theta);
				if(f > thred) {
					thetas.push_back(theta);
					pos++;
					if(pos > K) break;
				}
			}

			// std::cout << "thetas: ";
			// pos = 0;
			// for(auto& v : thetas) std::cout << v << ',\t' << function_p(pos++, v) << std::endl;
			// std::cout << std::endl;
		}

		struct pairs{
			int lb;
			int ub;
			pairs(int l, int u) :lb(l), ub(u) {}
			pairs() = default;

		};

		void knn(queryN* q) {
			lsh::timer timer;

			//int np = part_map.size() - 1;
			int cnt = 0;
			int ub = 200;
			//std::vector<bool> visited(data.N, false);
			//visited.resize(data.N);

			//std::cerr << "here!" << std::endl;

			std::unordered_map<int, pairs> visited;
			calQHash(q);

			int unvisited_ub = data.N - 1;

#ifdef USE_SHIFT
			float q_xc = cal_inner_product(q->queryPoint, xc.data(), data.dim);
#endif // 

			//auto& bucket = hash_tables[i][q->hashval[i]];
			for(int j = 0;j < K;++j){
				for(int i = 0;i < L;++i){
					auto& table = hash_tables[i];
					int val = q->hashval[i];
					for(auto& mask : masks[j]){
						int v = val ^ mask;
						auto& bucket = table[v];
						for(auto& id : bucket){
							if(visited.find(id) == visited.end()){
								float ip = cal_inner_product(q->queryPoint, data[id], data.dim);
								q->resHeap.emplace(norms[m + id].second, 1.0 - ip);
								float cos_sim = 1.0 - ip / norms[m + id].first;
								q->resCos.emplace(id, cos_sim);

								if(q->resHeap.size() > q->k) q->resHeap.pop();
								if(q->resCos.size() > 1) q->resCos.pop();
								//}
								//else{
								visited[id] = pairs(id, id);
								if(visited.find(id - 1) != visited.end()){
									if(visited.find(id + 1) != visited.end()){
										visited[visited[id - 1].lb].ub = visited[id + 1].ub;
										visited[visited[id + 1].ub].lb = visited[id - 1].lb;
										//visited.erase(id + 1);
									}
									else{
										visited[visited[id - 1].lb].ub = id;
										visited[id].lb = visited[id - 1].lb;
									}
									//visited[id].lb = visited[id - 1].lb;

									//visited.erase(id - 1);
								}
								else{
									if(visited.find(id + 1) != visited.end()){
										visited[visited[id + 1].ub].lb = id;
										visited[id].ub = visited[id + 1].ub;
										//visited.erase(id + 1);
									}
								}

								if(id == unvisited_ub) unvisited_ub = visited[id].lb - 1;
							}
						}
					}

				}
#ifdef USE_SHIFT
				if(q->resHeap.size() >= q->k && 1.0f - q->resHeap.top().dist - q_xc > norms[m + unvisited_ub].first * cos(thetas[j]))
					break;
#else
				// if(q->resHeap.size() >= q->k && 1.0f - q->resHeap.top().dist > q->norm * norms[m + unvisited_ub].first * cos(thetas[j]))
				// 	break;
#endif // 

			}
		}


	};
}
