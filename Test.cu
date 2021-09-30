#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <assert.h>


#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>


using namespace std;
using namespace std::chrono;


__global__ void vecAdd(int* a, int* b, int* c, int N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

__global__ void matMul(float* a, float* b, float* c, int N) {
	int COL = blockIdx.y * blockDim.y + threadIdx.y;
	int ROW = blockIdx.x * blockDim.x + threadIdx.x;
	if (ROW < N && COL < N) {
		float tmp_sum = 0.0f;
		for (int i = 0; i < N; i++) {
			tmp_sum += a[ROW * N + i] + b[i * N + COL];
		}
		c[ROW * N + COL] = tmp_sum;
	}
}

void verify_result(vector<int> a, vector<int> b, vector<int> c) {
	for (int i = 0; i < a.size(); i++) {
		assert(c[i] == a[i] + b[i]);
	}
}

int main() {

	auto start = high_resolution_clock::now();
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	const int N = 2048;
	int SIZE = N * N;
	size_t bytes = SIZE * sizeof(float);
	vector<float> h_a(SIZE);
	vector<float> h_b(SIZE);
	vector<float> h_c(SIZE);
	float a[N][N], b[N][N], c[N][N];
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			h_a[i * N + j] = sin(i);
			h_b[i * N + j] = cos(j);
			a[i][j] = sin(i);
			b[i][j] = cos(j);
			c[i][j] = 0;
		}
	}

	start = high_resolution_clock::now();
	int num = 0;
	for (int l = 0; l < N; l++) {
		for (int m = 0; m < N; m++) {
			for (int n = 0; n < N; n++) {
				c[l][m] += a[l][n] * b[n][m];
			}
		}
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	cout << "CPU: " << duration.count() << endl;


	start = high_resolution_clock::now();

	int BLOCKSIZE = 1 << 10;
	dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
	int n_blocks = ceil(N / BLOCKSIZE);
	dim3 blocksPerGrid(n_blocks, n_blocks);

	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	matMul <<< blocksPerGrid, threadsPerBlock >>> (d_a, d_b, d_c, N);
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);


	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	cout << "GPU: " << duration.count() << endl;
	

	




	/*

	constexpr int N = 1 << 17;
	size_t bytes = sizeof(int) * N;

	vector<int> a(N);
	vector<int> b(N);
	vector<int> c(N);

	std::generate(begin(a), end(a), []() {return rand() % 100; });
	std::generate(begin(b), end(b), []() {return rand() % 100; });

	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

	int NUM_THREADS = 1 << 10;
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	start = high_resolution_clock::now();
	vecAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, N);

	cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);

	cout << "GPU: " << duration.count() << endl;

	start = high_resolution_clock::now();
	for (int i = 0; i < a.size(); i++) {
		c[i] = a[i] + b[i];
	}
	stop = high_resolution_clock::now();

	duration = duration_cast<microseconds>(stop - start);
	cout << "CPU: " << duration.count() << endl;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	verify_result(a, b, c);
	return 0;
	*/

}