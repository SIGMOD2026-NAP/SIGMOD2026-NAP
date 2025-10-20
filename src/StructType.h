#pragma once

struct Data
{
	// Dimension of data
	int dim;
	// Number of data
	int N;
	// Data matrix
	float** val = nullptr;

	// No safety checking!!!
	float*& operator[](int i) { return val[i]; }

	float* base = nullptr;
};

struct Ben
{
	int N;
	int num;
	int** indice;
	float** innerproduct;
};

struct HashParam
{
	// the value of a in S hash functions
	float** rndAs1;
	// the value of a in S hash functions
	float** rndAs2;
};

