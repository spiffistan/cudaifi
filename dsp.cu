#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "tables.h"
#include "dsp.hcu"
#include "cuPrintf.cuh"
#define ISQRT2 0.70710678118654f
extern "C" {
#include "c63.h"
static void transpose_block(float *in_data, float *out_data) {
	int i, j;
	for (i = 0; i < 8; ++i)
		for (j = 0; j < 8; ++j) {
			out_data[i * 8 + j] = in_data[j * 8 + i];
		}
}

static void dct_1d(float *in_data, float *out_data) {
	int i, j;

	for (j = 0; j < 8; ++j) {
		float dct = 0;

		for (i = 0; i < 8; ++i) {
			dct += in_data[i] * dctlookup[i][j];
		}

		out_data[j] = dct;
	}
}

static void idct_1d(float *in_data, float *out_data) {
	int i, j;

	for (j = 0; j < 8; ++j) {
		float idct = 0;

		for (i = 0; i < 8; ++i) {
			idct += in_data[i] * dctlookup[j][i];
		}

		out_data[j] = idct;
	}
}

static void scale_block(float *in_data, float *out_data) {
	int u, v;

	for (v = 0; v < 8; ++v) {
		for (u = 0; u < 8; ++u) {
			float a1 = !u ? ISQRT2 : 1.0f;
			float a2 = !v ? ISQRT2 : 1.0f;

			/* Scale according to normalizing function */
			out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
		}
	}
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl) {
	int zigzag;
	for (zigzag = 0; zigzag < 64; ++zigzag) {
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[v * 8 + u];

		/* Zig-zag and quantize */
		out_data[zigzag] = round((dct / 4.0) / quant_tbl[zigzag]);
	}
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl) {
	int zigzag;
	for (zigzag = 0; zigzag < 64; ++zigzag) {
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[zigzag];

		/* Zig-zag and de-quantize */
		out_data[v * 8 + u] = round((dct * quant_tbl[zigzag]) / 4.0);
	}
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl) {
	float mb[8 * 8] __attribute((aligned(16)));
	float mb2[8 * 8] __attribute((aligned(16)));

	int i, v;

	for (i = 0; i < 64; ++i)
		mb2[i] = in_data[i];

	for (v = 0; v < 8; ++v) {
		dct_1d(mb2 + v * 8, mb + v * 8);
	}

	transpose_block(mb, mb2);

	for (v = 0; v < 8; ++v) {
		dct_1d(mb2 + v * 8, mb + v * 8);
	}

	transpose_block(mb, mb2);
	scale_block(mb2, mb);
	quantize_block(mb, mb2, quant_tbl);

	for (i = 0; i < 64; ++i)
		out_data[i] = mb2[i];
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl) {
	float mb[8 * 8] __attribute((aligned(16)));
	float mb2[8 * 8] __attribute((aligned(16)));

	int i, v;

	for (i = 0; i < 64; ++i)
		mb[i] = in_data[i];

	dequantize_block(mb, mb2, quant_tbl);

	scale_block(mb2, mb);

	for (v = 0; v < 8; ++v) {
		idct_1d(mb + v * 8, mb2 + v * 8);
	}

	transpose_block(mb2, mb);

	for (v = 0; v < 8; ++v) {
		idct_1d(mb + v * 8, mb2 + v * 8);
	}

	transpose_block(mb2, mb);

	for (i = 0; i < 64; ++i)
		out_data[i] = mb[i];
}

void catchCudaError(const char *message) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
		exit(-1);
	}
}

} /* end extern "C" */

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//meant for 512 threads
//threadDim(8,8,4);


#define LENGTH 40
#define MB_X blockIdx.x
#define MB_Y blockIdx.y
#define NUM_MB_X gridDim.x
#define NUM_MB_Y gridDim.y
#define MX (MB_X * 8)
#define MY (MB_Y * 8)
#define WIDTH (gridDim.x * 8)
#define HEIGHT (gridDim.y * 8)
#define RANGE 16

__shared__ uint8_t ref[LENGTH * LENGTH]; // <= (40) * (40)
__shared__ uint8_t orig[64];
__shared__ min_helper minimum[32 * 32];

__device__
inline void calculate_usad(int mb_h, int mb_w, int res_index, int xyz_index) {
	if (threadIdx.z < mb_w) {
		for (int y = 0; y < 4; y++) {
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					uint16_t current = minimum[(y * 8 * 32) + res_index].value = __usad(ref[(y * 8 * 40) + (i * 40) + xyz_index + j], orig[(i * 8) + j], minimum[(y * 8 * 32) + res_index].value);
				}
			}
		}
	}
	__syncthreads();
}

__device__
inline void zero_buffers(int res_index) {
	for (int i = 0; i < 4; i++) {
		minimum[(i * 32 * 8) + res_index].value = 0;
	}
}

__device__
inline void copy_min(int res_index_to, int res_index_from) {
	minimum[res_index_to].value = minimum[res_index_from].value;
	minimum[res_index_to].x = minimum[res_index_from].x;
	minimum[res_index_to].y = minimum[res_index_from].y;
}

__device__
inline void reduce_min(int num_block_rows, int num_block_columns, int res_index) {
	//find minimum
	//first setup coordinates
	for (int i = 0; i < 4; i++) {
		minimum[(i * 32 * 8) + res_index].x = threadIdx.z * 8 + threadIdx.x;
		minimum[(i * 32 * 8) + res_index].y = i * 8 + threadIdx.y;
	}

	//reduce minimum

	uint16_t current;
	if (num_block_rows == 4) {
		if (threadIdx.z < num_block_columns) {
			current = min(minimum[res_index].value, minimum[(16 * 32) + res_index].value);
			if (current != minimum[res_index].value) {
				minimum[res_index] = minimum[(16 * 32) + res_index];
			}
			current = min(minimum[(8 * 32) + res_index].value, minimum[(24 * 32) + res_index].value);
			if (current != minimum[(8 * 32) + res_index].value) {
				minimum[(8 * 32) + res_index] = minimum[(24 * 32) + res_index];
			}
		}
	} else if (num_block_rows == 3) {
		if (threadIdx.z < num_block_columns) {
			current = min(minimum[res_index].value, minimum[(16 * 32) + res_index].value);
			if (current != minimum[res_index].value) {
				minimum[res_index] = minimum[(16 * 32) + res_index];
			}
		}
	}
	__syncthreads();
	if (threadIdx.z < num_block_columns) {
		current = min(minimum[res_index].value, minimum[(8 * 32) + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[(8 * 32) + res_index];
		}
	}
	__syncthreads();
	if (threadIdx.y < 4 && threadIdx.z < num_block_columns) {
		current = min(minimum[res_index].value, minimum[(4 * 32) + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[(4 * 32) + res_index];
		}
	}
	__syncthreads();

	if (threadIdx.y < 2 && threadIdx.z < num_block_columns) {
		current = min(minimum[res_index].value, minimum[(2 * 32) + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[(2 * 32) + res_index];
		}
	}
	__syncthreads();

	if (threadIdx.y == 0 && threadIdx.z < num_block_columns) {
		current = min(minimum[res_index].value, minimum[32 + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[32 + res_index];
		}
	}
	__syncthreads();
	if (num_block_columns == 4) {
		if (threadIdx.y == 0 && threadIdx.z < 2) {
			current = min(minimum[res_index].value, minimum[16 + res_index].value);
			if (current != minimum[res_index].value) {
				minimum[res_index] = minimum[16 + res_index];
			}
		}
	} else if (num_block_columns == 3) {
		if (threadIdx.y == 0 && threadIdx.z == 0) {
			current = min(minimum[res_index].value, minimum[16 + res_index].value);
			if (current != minimum[res_index].value) {
				minimum[res_index] = minimum[16 + res_index];
			}
		}
	}
	__syncthreads();
	if (threadIdx.y == 0 && threadIdx.z == 0) {
		current = min(minimum[res_index].value, minimum[8 + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[8 + res_index];
		}
	}
	__syncthreads();
	if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < 4) {
		current = min(minimum[res_index].value, minimum[4 + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[4 + res_index];
		}
	}
	__syncthreads();
	if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x < 2) {
		current = min(minimum[res_index].value, minimum[2 + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[2 + res_index];
		}
	}
	__syncthreads();
	if (threadIdx.y == 0 && threadIdx.z == 0 && threadIdx.x == 0) {
		current = min(minimum[res_index].value, minimum[1 + res_index].value);
		if (current != minimum[res_index].value) {
			minimum[res_index] = minimum[1 + res_index];
		}
	}
	__syncthreads();
}

__device__
inline void load_values(uint8_t *original, uint8_t *reference, int stride, uint32_t mb_w, uint32_t mb_h, bool rightEdge, bool bottomEdge, int xyz_index) {

	int ref_index = (threadIdx.y * stride) + (threadIdx.z * blockDim.x) + threadIdx.x;
	//@TODO try to read reference as a 32bit in to get coalesced values on compute 1.1
	//load ref
	for (int i = 0; i < 4; i++) {
		ref[(i * LENGTH * 8) + xyz_index] = reference[(i * 8 * stride) + ref_index];
	}
	if (!bottomEdge) { //if we still got more room below, we need to read it.
		ref[(32 * LENGTH) + xyz_index] = reference[(32 * stride) + ref_index];
	}
	if (!rightEdge) { //need to fill the gap on the right side, instead of horizontal read, we do vertical.
		ref[(threadIdx.z * LENGTH * 8) + (threadIdx.y * LENGTH) + 32 + threadIdx.x] = reference[(threadIdx.z * 8 * stride) + (stride * threadIdx.y) + (32 + threadIdx.x)];
	}
	//Load orig with 64 threads, and or bottomright corner if it exists
	if (threadIdx.z == 0) {
		orig[(threadIdx.y * blockDim.x) + threadIdx.x] = original[(threadIdx.y * stride) + threadIdx.x];
	} else if (threadIdx.z == 1 && !bottomEdge && !rightEdge) {
		ref[(32 * LENGTH) + 32 + (threadIdx.y * LENGTH) + threadIdx.x] = reference[(32 * stride) + 32 + (threadIdx.y * stride) + threadIdx.x];
	}

	__syncthreads();
}

__global__
void cuda_me2(uint8_t *original, uint8_t *reference, int stride, macroblock *mb, min_helper *test) {
	int left = MX - RANGE;
	int top = MY - RANGE;
	int right = MX + RANGE;
	int bottom = MY + RANGE;
	bool rightEdge = false;
	bool bottomEdge = false;
	if (left < 0)
		left = 0;
	if (top < 0)
		top = 0;
	if (right > (WIDTH - 8)) {
		right = WIDTH - 8;
		rightEdge = true;
	}
	if (bottom > (HEIGHT - 8)) {
		bottom = HEIGHT - 8;
		bottomEdge = true;
	}

	reference = reference + (top * stride) + left;
	original = original + (MY * stride) + MX;

	int diff_ref_orig_x = MX - left;
	int diff_ref_orig_y = MY - top;

	int res_index = (threadIdx.y * 32) + (threadIdx.z * 8) + threadIdx.x;
	int xyz_index = (threadIdx.y * LENGTH) + (threadIdx.z * blockDim.x) + threadIdx.x;
	int mb_w = (right - left) / 8;
	int mb_h = (bottom - top) / 8;
	load_values(original, reference, stride, mb_h, mb_w, rightEdge, bottomEdge, xyz_index);

	//compute SAD
	zero_buffers(res_index);
	calculate_usad(mb_h, mb_w, res_index, xyz_index);
	reduce_min(mb_h, mb_w, res_index);
//
//	//TEST
//	if (MB_X == 0 && MB_Y == 135) {
//		for (int i = 0; i < 4; i++) {
//			test[i * 32 * 8 + res_index] = minimum[i * 32 * 8 + res_index];
//		}
//		test[0].x = minimum[0].x - diff_ref_orig_x;
//		test[0].y = minimum[0].y - diff_ref_orig_y;
//		test[1].x = minimum[0].x;
//		test[1].y = minimum[1].y;
//		test[2].x = left;
//		test[2].y = top;
//		test[3].x = right;
//		test[3].y = bottom;
//		test[4].x = WIDTH;
//		test[4].y = HEIGHT;
//	}
	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
		mb[MB_Y * gridDim.x + MB_X].mv_x = minimum[0].x - diff_ref_orig_x;
		mb[MB_Y * gridDim.x + MB_X].mv_y = minimum[0].y - diff_ref_orig_y;
		mb[MB_Y * gridDim.x + MB_X].use_mv = 1;
	}
	__syncthreads();
}

/*
 __constant__ int image_stride = 16;
 __constant__ int result_stride = 64;
 __constant__ int n = 64;
 __global__
 void happy_block_8x8(uint8_t *orig, uint8_t *ref, int stride, uint32_t *result_block)
 {
 extern __shared__ uint32_t results[];
 int res_index = (blockIdx.y * gridDim.x) + blockIdx.x; //index to the result_block
 ref = ref + (blockIdx.y * stride) + blockIdx.x; //ref image
 int i = threadIdx.y * stride + threadIdx.x; //index to the memory in the images.
 int j = blockDim.x * threadIdx.y + threadIdx.x; //index to the results
 if(j < 64) {
 results[j] = abs(orig[i] - ref[i]);

 }

 __syncthreads();
 for(int offset = 1;offset < n; offset *= 2) {
 if(j >= offset)
 results[j] += results[j - offset];

 __syncthreads();
 }

 __syncthreads();
 if(j == 63)
 result_block[res_index] = results[j];
 }
 */
