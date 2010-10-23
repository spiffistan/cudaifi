
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

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__constant__ int image_stride = 16;
__constant__ int result_stride = 64;

//meant for 512 threads
//threadDim(8,8,4);



#define LENGTH 40

__global__
void cuda_me(uint8_t *original, uint8_t *reference, int stride, uint16_t *result_block, uint32_t mb_w, uint32_t mb_h, uint8_t rightEdge, uint8_t bottomEdge)
{
	__shared__ uint8_t ref[LENGTH*LENGTH]; // <= (40) * (40)
	__shared__ uint8_t orig[64];
	__shared__ uint16_t uSAD[4][64];
	__shared__ uint16_t results[32*32];
	//@TODO try to read reference as a 32bit in to get coalesced values on compute 1.1
	int xyz_index = (threadIdx.y * LENGTH) + (threadIdx.z * blockDim.x)+ threadIdx.x;
	int ref_index = (threadIdx.y * stride) + (threadIdx.z * blockDim.x)+ threadIdx.x;
	int xy_index = (threadIdx.y * blockDim.x) + threadIdx.x;
	//load ref
	for(int i = 0; i < 4; i++) {
		ref[(i*LENGTH*8)+xyz_index] = reference[(i*blockDim.y*stride)+ref_index];
	}


	if(!bottomEdge) {
		ref[(32*LENGTH)+xyz_index] = reference[(32*stride)+ref_index];
	}
	if(!rightEdge) { //need to fill the gap on the right side, instead of horizontal read, we do vertical.
		ref[(threadIdx.z * LENGTH * 8)+(threadIdx.y * LENGTH)+32+threadIdx.x] = reference[(threadIdx.z*8*stride)+(stride*threadIdx.y)+(32+threadIdx.x)];
	}
	__syncthreads();

	//Load orig with 64 threads, and a special case in ref with 64 more
	if(threadIdx.z == 0) {
		orig[(threadIdx.y*blockDim.x)+threadIdx.x] = original[(threadIdx.y*stride)+threadIdx.x];
	}
	else if(threadIdx.z == 1 && !bottomEdge && !rightEdge) {
		ref[(32*LENGTH)+32+(threadIdx.y*LENGTH)+threadIdx.x] = reference[(32*stride)+32+(threadIdx.y*stride)+threadIdx.x];
	}
	__syncthreads();
	//TEST



	for(int offsetY = 0; offsetY < 32; offsetY++) {
		for(int offsetX = 0; offsetX < 8; offsetX++) {
			//uSAD[threadIdx.z][xy_index] = abs(ref[xyz_index] - orig[xy_index]); //TODO use __usad()
			uSAD[threadIdx.z][xy_index] = 0;
			uSAD[threadIdx.z][xy_index] = __usad(ref[(offsetY*LENGTH)+offsetX+xyz_index], orig[xy_index], uSAD[threadIdx.z][xy_index]);
			__syncthreads();
			if(xy_index < 32)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+32];
			__syncthreads();
			if(xy_index < 16)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+16];
			__syncthreads();
			if(xy_index < 8)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+8];
			__syncthreads();
			if(xy_index < 4)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+4];
			__syncthreads();
			if(xy_index < 2)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+2];
			__syncthreads();
			if(xy_index < 1)
				uSAD[threadIdx.z][xy_index] += uSAD[threadIdx.z][xy_index+1];
			__syncthreads();
			if(threadIdx.x == 0 && threadIdx.y == 0) {
				results[(offsetY*32)+offsetX+(threadIdx.z*8)] = uSAD[threadIdx.z][0];
			}
			__syncthreads();
		}
	}

	for(int i = 0; i < mb_h; i++) {
		result_block[(i*40*8)+xyz_index] = results[(i*32*8)+(threadIdx.y*32)+(threadIdx.z*8)+threadIdx.x];
	}
}



/*
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
