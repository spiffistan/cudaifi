
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "tables.h"
#include "dsp.hcu"

#define ISQRT2 0.70710678118654f
extern "C" {
static void transpose_block(float *in_data, float *out_data)
{
    int i,j;
    for (i=0; i<8; ++i)
        for (j=0; j<8; ++j)
        {
            out_data[i*8+j] = in_data[j*8+i];
        }
}

static void dct_1d(float *in_data, float *out_data)
{
    int i,j;

    for (j=0; j<8; ++j)
    {
        float dct = 0;

        for (i=0; i<8; ++i)
        {
            dct += in_data[i] * dctlookup[i][j];
        }

        out_data[j] = dct;
    }
}

static void idct_1d(float *in_data, float *out_data)
{
    int i,j;

    for (j=0; j<8; ++j)
    {
        float idct = 0;

        for (i=0; i<8; ++i)
        {
            idct += in_data[i] * dctlookup[j][i];
        }

        out_data[j] = idct;
    }
}


static void scale_block(float *in_data, float *out_data)
{
    int u,v;

    for (v=0; v<8; ++v)
    {
        for (u=0; u<8; ++u)
        {
            float a1 = !u ? ISQRT2 : 1.0f;
            float a2 = !v ? ISQRT2 : 1.0f;

            /* Scale according to normalizing function */
            out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
        }
    }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[v*8+u];

        /* Zig-zag and quantize */
        out_data[zigzag] = round((dct / 4.0) / quant_tbl[zigzag]);
    }
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
    int zigzag;
    for (zigzag=0; zigzag < 64; ++zigzag)
    {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[zigzag];

        /* Zig-zag and de-quantize */
        out_data[v*8+u] = round((dct * quant_tbl[zigzag]) / 4.0);
    }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
    float mb[8*8] __attribute((aligned(16)));
    float mb2[8*8] __attribute((aligned(16)));

    int i, v;

    for (i=0; i<64; ++i)
        mb2[i] = in_data[i];

    for (v=0; v<8; ++v)
    {
        dct_1d(mb2+v*8, mb+v*8);
    }

    transpose_block(mb, mb2);

    for (v=0; v<8; ++v)
    {
        dct_1d(mb2+v*8, mb+v*8);
    }

    transpose_block(mb, mb2);
    scale_block(mb2, mb);
    quantize_block(mb, mb2, quant_tbl);

    for (i=0; i<64; ++i)
        out_data[i] = mb2[i];
}


void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl)
{
    float mb[8*8] __attribute((aligned(16)));
    float mb2[8*8] __attribute((aligned(16)));

    int i, v;

    for (i=0; i<64; ++i)
        mb[i] = in_data[i];

    dequantize_block(mb, mb2, quant_tbl);

    scale_block(mb2, mb);

    for (v=0; v<8; ++v)
    {
        idct_1d(mb+v*8, mb2+v*8);
    }

    transpose_block(mb2, mb);

    for (v=0; v<8; ++v)
    {
        idct_1d(mb+v*8, mb2+v*8);
    }

    transpose_block(mb2, mb);

    for (i=0; i<64; ++i)
        out_data[i] = mb[i];
}

void catchCudaError(const char *message)
{
   cudaError_t error = cudaGetLastError();
   if(error!=cudaSuccess) {
      fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
      exit(-1);
   }                         
}

} /* end extern "C" */


__host__
void happy_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{	
	*result = 0;
	
	const uint8_t BLOCKSIZE = 64;
		
	int *result_d = 0;
		
	uint8_t *block1_d, *block2_d;
	
	uint32_t *block1_2 = (uint32_t*) malloc(64*sizeof(uint32_t));
	uint32_t *result_block_d, *result_block_2_d;
		
	cudaMalloc((void **) &block1_d, 8*stride);
	cudaMalloc((void **) &block2_d, 8*stride);
	cudaMalloc((void **) &result_block_d, BLOCKSIZE*sizeof(uint32_t));
	cudaMalloc((void **) &result_block_2_d, BLOCKSIZE*sizeof(uint32_t));
	cudaMalloc((void **) &result_d, sizeof(int));

	cudaMemcpy(block1_d, block1, 8*stride, cudaMemcpyHostToDevice);
	cudaMemcpy(block2_d, block2, 8*stride, cudaMemcpyHostToDevice);
	

	for(int i = 0; i < 8*352; i+=352) {
		for(int j = 0; j < 8; j++) {
			if(j % 8 == 0)
				printf("\n");
			printf("%3d ", block1[i+j]);
		}
	}
	printf("\n-------------------------------");
	for(int i = 0; i < 8*352; i+=352) {
		for(int j = 0; j < 8; j++) {
			if(j % 8 == 0)
				printf("\n");
			printf("%3d ", block2[i+j]);
		}
	}
	printf("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
	happy_block_8x8_d<<<8,8>>>(block1_d, block2_d, result_block_d, stride);
	
	cudaThreadSynchronize();

	catchCudaError("Failed Execution 1");
	
	cudaMemcpy(block1_2, result_block_d, 64*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	
	catchCudaError("Failed Execution 2");
			
	for(int i = 0; i < 64; i++) {
			if(i % 8 == 0)
				printf("\n");
			printf("%3d ", block1_2[i]);
	}


	reduce0<<<1, 64, 64*sizeof(uint32_t)>>>(result_block_d, result_block_2_d, 64);

	cudaThreadSynchronize();

	//printf("\nresult: %d\n", *result_d);
	//printf("\n\n\n\n");

	catchCudaError("Failed Execution 3");

	cudaMemcpy(result, &result_block_2_d[63], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	printf("\nresult: %d\n", *result);

	catchCudaError("Failed Execution 4");

	cudaFree(result_block_d);
	cudaFree(block1_d);
	cudaFree(block2_d);
	cudaFree(result_d);

}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__global__
void happy_block_8x8_d(uint8_t *block1_d, uint8_t *block2_d, uint32_t *result_block_d, int stride)
{
	int i = blockIdx.x * stride + threadIdx.x;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if(j < 64) {
		result_block_d[j] = abs(block1_d[i] - block2_d[i]);

	}
}

__global__
void diff_abs_frame(uint8_t *block1_d, uint8_t *block2_d, uint8_t *result_block_d, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < size) {
		result_block_d[i] = abs(block1_d[i] - block2_d[i]);

	}
}


///////////////////////////////////////////////////////////////////////////////

__global__
void reduce0(uint32_t *g_idata, uint32_t *g_odata, uint32_t n) {

    extern __shared__ uint32_t temp[];
    int thid = threadIdx.x;

    temp[thid] = g_idata[thid];

    __syncthreads();

    for(int offset = 1;offset < n; offset *= 2) {
        if(thid >= offset)
            temp[thid] += temp[thid - offset];

        __syncthreads();
    }

   	g_odata[thid] = temp[thid];
}

