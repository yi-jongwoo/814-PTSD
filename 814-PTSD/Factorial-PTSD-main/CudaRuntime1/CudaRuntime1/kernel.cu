#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <random>
#include <fstream>
using namespace std;
typedef int gene[8][14];
typedef unsigned long long ull;
#define rnd(x) ((x = x * 1103515245 + 12345)>>16)

__device__ int gevalnum(gene g, int now) {
	int tx[8] = { 1,1,-1,-1,1,0,-1,0 };
	int ty[8] = { 1,-1,1,-1,0,1,0,-1 };
	int a[4] = { now % 10,now / 10 % 10,now / 100 % 10,now / 1000 };
	for (int i = 0; i < 8; i++) for (int j = 0; j < 14; j++)
		if (g[i][j] == a[0])for (int k1 = 0; k1 < 8; k1++) {
			int ii = i + tx[k1], jj = j + ty[k1];
			if (ii < 0 || ii>7 || jj < 0 || jj>13)
				continue;
			if (g[ii][jj] == a[1]) for (int k2 = 0; k2 < 8; k2++) {
				int iii = ii + tx[k2], jjj = jj + ty[k2];
				if (iii < 0 || iii>7 || jjj < 0 || jjj>13)
					continue;
				if (g[iii][jjj] == a[2]) for (int k3 = 0; k3 < 8; k3++) {
					int iiii = iii + tx[k3], jjjj = jjj + ty[k3];
					if (iiii < 0 || iiii>7 || jjjj < 0 || jjjj>13)
						continue;
					if (g[iiii][jjjj] == a[3])
						return 1;
				}
			}
		}
	return 0;
}
__device__ int geval(gene g) {
	int cnt = 0;
	for (int now = 0; now <= 9999; now++) {
		cnt += gevalnum(g, now);
	}
	return cnt;
}
__global__ void addKernel(gene* crr, ull gseed) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	ull rseed = tid | gseed << 16;
	gene pool[16];
	gene nextpool[16];
	int epool[16];
	int irr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	for (int i = 0; i < 16; i++)
		memcpy(pool[i], crr[rnd(rseed) % 4096], sizeof(gene));
	for (int round = 0; round <= 16; round++) {
		for (int i = 0; i < 16; i++)
			epool[i] = geval(pool[i]) << 16 | i;
		for (int i = 0; i < 16; i++)
			for (int j = 0; j < 15; j++)
				if (epool[irr[j]] < epool[irr[j + 1]])
					irr[j] ^= irr[j + 1] ^= irr[j] ^= irr[j + 1];
		if (round == 16) {
			memcpy(crr[tid], pool[irr[0]], sizeof(gene));
			break;
		}
		int rsum = 0, bas= epool[irr[15]]>>16;
		for (int i = 0; i < 16; i++)
			rsum += (epool[i]>>16) - bas +1;
		for (int t = 0; t < 16; t++) {
			int p1, p2;
			int ps = rnd(rseed) % rsum;
			for(int i=0;i<15;i++)
				if ((ps -= (epool[i] >> 16)-bas +1) <= 0) {
					p1 = i; break;
				}ps = rnd(rseed) % rsum;
			for (int i = 0; i < 15; i++)
				if ((ps -= (epool[i] >> 16)-bas +1) <= 0) {
					p2 = i; break;
				}
			memcpy(nextpool[t], pool[p1], sizeof(gene));
			int x1 = rnd(rseed) % 8, x2 = rnd(rseed) % 8; if (x1 > x2)x1 ^= x2 ^= x1 ^= x2;
			int y1 = rnd(rseed) % 14, y2 = rnd(rseed) % 14; if (y1 > y2)y1 ^= y2 ^= y1 ^= y2;
			for (int i = x1; i <= x2; i++)for (int j = y1; j <= y2; j++)nextpool[t][i][j] = pool[p2][i][j];
			if (rnd(rseed) % 4 == 0)
				for (int i = 0; i < 8; i++) for (int j = 0; j < 14; j++)
					if (rnd(rseed) % 64 == 0)
						nextpool[t][i][j] = rnd(rseed) % 10;
		}
		memcpy(pool, nextpool, sizeof pool);
	}
}
cudaError_t addWithCuda(gene* c, long long num)
{
	gene* dev_c = 0;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_c, 4096 * sizeof(gene));
	cudaStatus = cudaMemcpy(dev_c, c, 4096 * sizeof(gene), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	addKernel <<<32, 128>>> (dev_c, num);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	cudaStatus = cudaMemcpy(c, dev_c, 4096 * sizeof(gene), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_c);
	return cudaStatus;
}
int hgevalnum(gene g, int now) {
	int tx[8] = { 1,1,-1,-1,1,0,-1,0 };
	int ty[8] = { 1,-1,1,-1,0,1,0,-1 };
	int a[4] = { now % 10,now / 10 % 10,now / 100 % 10,now / 1000 };
	for (int i = 0; i < 8; i++) for (int j = 0; j < 14; j++)
		if (g[i][j] == a[0])for (int k1 = 0; k1 < 8; k1++) {
			int ii = i + tx[k1], jj = j + ty[k1];
			if (ii < 0 || ii>7 || jj < 0 || jj>13)
				continue;
			if (g[ii][jj] == a[1]) for (int k2 = 0; k2 < 8; k2++) {
				int iii = ii + tx[k2], jjj = jj + ty[k2];
				if (iii < 0 || iii>7 || jjj < 0 || jjj>13)
					continue;
				if (g[iii][jjj] == a[2]) for (int k3 = 0; k3 < 8; k3++) {
					int iiii = iii + tx[k3], jjjj = jjj + ty[k3];
					if (iiii < 0 || iiii>7 || jjjj < 0 || jjjj>13)
						continue;
					if (g[iiii][jjjj] == a[3])
						return 1;
				}
			}
		}
	return 0;
}
int hgeval(gene g) {
	int cnt = 0;
	for (int now = 0; now <= 9999; now++) {
		cnt += hgevalnum(g, now);
	}
	return cnt;
}
int main() {
	static gene crr[4096];
	mt19937 rgen; rgen.seed(554444);
	for (int t = 0; t < 4096; t++)
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 14; j++)
				crr[t][i][j] = rgen()%10;
	int rrr = 0;
	for (long long iiii = 1;; iiii++) {
		cout << iiii;
		int tfrom = time(nullptr);
		cudaError_t cudaStatus = addWithCuda(crr, iiii);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
		cout << ':' << time(nullptr) - tfrom << "sec" << endl; tfrom = time(nullptr);
		int rr = 0, ri = 0;
		for (int i = 0; i < 4096; i++) {
			int r = hgeval(crr[i]);
			if (rr < r) {
				rr = r;
				ri = i;
			}
		}
		cout << '!' << rr << ':' << time(nullptr) - tfrom << "sec" << endl;
		if (rrr < rr) {
			rrr = rr;
			ofstream fout; fout.open("813out.txt", ios_base::out);
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 14; j++)
					fout << crr[ri][i][j];
				fout << '\n';
			}
			fout.close();
		}
		int tx[8] = { 1,1,-1,-1,1,0,-1,0 };
		int ty[8] = { 1,-1,1,-1,0,1,0,-1 };
		for (int t = 0; t < 4096; t++){
			int perm[10] = { 0,1,2,3,4,5,6,7,8,9 };
			for (int i = 9; i > 0; i--) {
				int j = rgen() % (i + 1);
				swap(perm[i] , perm[j]);
			}
			for (int i = 0; i < 8; i++)
				for (int j = 0; j < 14; j++)
					crr[t][i][j] = perm[crr[t][i][j]];
			for(int now=1000;now<=8141;now++)
				if (now == 8141) {
					cout << "----end----\n";
					for (int i = 0; i < 8; i++) {
						for (int j = 0; j < 14; j++)
							cout << crr[t][i][j];
						cout << '\n';
					}
					exit(0);
				}
				else if (hgevalnum(crr[t],now) == 0) {
					int a[4] = { now % 10,now / 10 % 10,now / 100 % 10,now / 1000 };
					int i = rgen() % 8, j = rgen() % 14;
					for (int k = 0; k < 4; k++) {
						if (i < 0 || i>7 || j < 0 || j>13)
							break;
						crr[t][i][j] = a[k];
						int kk = rgen() % 8;
						i += tx[kk]; j += tx[kk];
					}
					break;
				}
		}
	}
	return 0;
}