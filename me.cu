#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include "c63.h"
#include "cuda_util.hcu"

#define MX (blockIdx.x * 8)
#define MY (blockIdx.y * 8)
#define RANGE 16
#define LENGTH 40
#define TX threadIdx.x
#define TY threadIdx.y
#define COMPSAD(i,j); \
  minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value); \
  minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);

struct min_helper {
	uint16_t value;
	int8_t x;
	int8_t y;
};

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS HOST //////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

texture<uint8_t, 2, cudaReadModeElementType> tex_ref;
texture<uint8_t, 2, cudaReadModeElementType> tex_orig;

__host__
void load_orig(uint8_t *host_ptr, uint8_t* dev_ptr, size_t width, size_t height, size_t pitch)
{
    cudaMemcpy2D(dev_ptr, pitch, host_ptr, width, width, height, cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, &tex_orig, dev_ptr, &tex_orig.channelDesc, width, height, pitch);
}

__host__
void load_ref(uint8_t *host_ptr, uint8_t* dev_ptr, size_t width, size_t height, size_t pitch)
{
    cudaMemcpy2D(dev_ptr, pitch, host_ptr, width, width, height, cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, &tex_ref, dev_ptr, &tex_ref.channelDesc, width, height, pitch);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS DEVICE ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


__shared__ uint32_t ref[LENGTH * LENGTH]; // <= (40) * (40)
__shared__ uint32_t orig[64];
__shared__ min_helper minimum[32 * 32];

__device__
inline void load_texture_values(int left, int top, int ref_index)
{
    ref[ref_index] = tex2D(tex_ref, left + TX, top + TY);
    ref[16 * 40 + ref_index] = tex2D(tex_ref, left + TX, top + 16 + TY);

    if (TY < 8) { //TODO Fix warp serialization
        //load vertically the blocks to the right
        ref[TX * 40 + 32 + TY] = tex2D(tex_ref, left + 32 + TY, top + TX);
    } else {
        //load the bottom row
        int y = TY - 8;
        ref[(32 + y) * 40 + TX] = tex2D(tex_ref, left + TX, top + 32 + y);
    }
    if (TY < 8 && TX < 8) {
        ref[32 * 40 + 32 + TY * 40 + TX] = tex2D(tex_ref, left + 32 + TX, top + 32 + TY);
        orig[TY * 8 + TX] = tex2D(tex_orig, MX + TX, MY + TY);
    }
    __syncthreads();
}




__device__
inline void calculate_usad(int res_index, int ref_index) {

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value);
        }
    }

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);
        }
    }
    __syncthreads();
}



__device__
inline void setup_min(int res_index)
{
    minimum[res_index].x = TX;
    minimum[res_index].y = TY;
    minimum[res_index].value = 0;
    minimum[32 * 16 + res_index].x = TX;
    minimum[32 * 16 + res_index].y = 16 + TY;
    minimum[32 * 16 + res_index].value = 0;

    __syncthreads();
}



#define MIN2(a,b) (a.value) < (b.value) ? (a) : (b);
#define COMPMIN(idx) minimum[res_index] = MIN2(minimum[res_index], minimum[(idx)]);

__device__
inline void reduce_min(int res_index)
{
	COMPMIN(16 * 32 + res_index); __syncthreads();			   // reduce to 2 block_rows
    if (TY <  8) COMPMIN(8 * 32 + res_index); __syncthreads(); // reduce to 1 block_row
    if (TY <  4) COMPMIN(4 * 32 + res_index); __syncthreads(); // reduce to 4 rows
    if (TY <  2) COMPMIN(2 * 32 + res_index); __syncthreads(); // reduce to 2 rows
    if (TY == 0) COMPMIN(32 + res_index); 					   // reduce to 1 row, no need to sync anymore, within 1 warp
    if (TY == 0 && TX < 16) COMPMIN(16 + res_index);  	       // reduce to 16 values
    if (TY == 0 && TX <  8) COMPMIN(8  + res_index);  		   // reduce to 8 values
    if (TY == 0 && TX <  4) COMPMIN(4  + res_index);  		   // reduce to 4 values
    if (TY == 0 && TX <  2) COMPMIN(2  + res_index);  	       // reduce to 2 values
    if (TY == 0 && TX == 0) COMPMIN(1);               		   // reduce to 1 value
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS GLOBAL ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__global__
void cuda_me_texture(int width, int height, macroblock * mb)
{
    int left = MX - 16;
    int top = MY - 16;

    int right = MX + 16;
    int bottom = MY + 16;

    if (left < 0)
        left = 0;

    if (top < 0)
        top = 0;

    if (right > (width - 8)) // Increase search area towards the left if we're out of bounds
        left += (width - 8) - right;

    if (bottom > (height - 8)) // Increase search area towards the top if we're out of bounds
        top += (height - 8) - bottom;


    int res_index = TY * 32 + TX;
    int ref_index = TY * 40 + TX;

	// Run kernels

    load_texture_values(left, top, ref_index);
    setup_min(res_index);
    calculate_usad(res_index, ref_index);
    reduce_min(res_index);

    if (TX == 0 && TY == 0)
    {
        mb[blockIdx.y * gridDim.x + blockIdx.x].mv_x = minimum[0].x + (left - MX);
        mb[blockIdx.y * gridDim.x + blockIdx.x].mv_y = minimum[0].y + (top - MY);
        mb[blockIdx.y * gridDim.x + blockIdx.x].use_mv = 1;
    }
}

/* Motion estimation */
extern "C" void c63_motion_estimate(struct c63_common *cm) {
	/* Compare this frame with previous reconstructed frame */
	uint8_t *image_orig_Y, *image_ref_Y;
	uint8_t *image_orig_U, *image_ref_U;
	uint8_t *image_orig_V, *image_ref_V;

	size_t pitch_orig_Y, pitch_ref_Y;
	size_t pitch_orig_U, pitch_ref_U;
	size_t pitch_orig_V, pitch_ref_V;

	macroblock *mb_dev;

	//MALLOCS
	cudaMallocPitch((void**) &image_orig_Y, &pitch_orig_Y, cm->width, cm->height);
	cudaMallocPitch((void**) &image_ref_Y, &pitch_ref_Y, cm->width, cm->height);
	cudaMallocPitch((void**) &image_orig_U, &pitch_orig_U, cm->width / 2, cm->height) / 2;
	cudaMallocPitch((void**) &image_ref_U, &pitch_ref_U, cm->width / 2, cm->height / 2);
	cudaMallocPitch((void**) &image_orig_V, &pitch_orig_V, cm->width / 2, cm->height / 2);
	cudaMallocPitch((void**) &image_ref_V, &pitch_ref_V, cm->width / 2, cm->height / 2);
	cudaMalloc((void**) &mb_dev, cm->mb_cols * cm->mb_rows * sizeof(*mb_dev));
	catchCudaError("MALLOC TEST");

	// Luma
	dim3 thread_dim(32, 16, 1);
	dim3 block_dim(cm->mb_cols, cm->mb_rows, 1);

	load_orig(cm->curframe->orig->Y, image_orig_Y, cm->width, cm->height, pitch_orig_Y);
	load_ref(cm->refframe->recons->Y, image_ref_Y, cm->width, cm->height, pitch_ref_Y);
	catchCudaError("TEXTURES");

	cuda_me_texture<<<block_dim, thread_dim>>>(cm->width, cm->height, mb_dev);
	catchCudaError("RUN");
	cudaMemcpy(cm->curframe->mbs[0], mb_dev, cm->mb_cols * cm->mb_rows * sizeof(*mb_dev), cudaMemcpyDeviceToHost);

	// Chroma
	dim3 block_dim_chroma(cm->mb_cols / 2, cm->mb_rows / 2, 1);

	//V
	load_orig(cm->curframe->orig->U, image_orig_U, cm->upw, cm->uph, pitch_orig_U);
	load_ref(cm->refframe->recons->U, image_ref_U, cm->upw, cm->uph, pitch_ref_U);
	catchCudaError("TEXTURE");

	cuda_me_texture<<<block_dim_chroma, thread_dim>>>(cm->upw, cm->uph, mb_dev);

	cudaMemcpy(cm->curframe->mbs[1], mb_dev, cm->mb_cols * cm->mb_rows / 4 * sizeof(*mb_dev), cudaMemcpyDeviceToHost);

	//U
	load_orig(cm->curframe->orig->V, image_orig_V, cm->vpw, cm->vph, pitch_orig_V);
	load_ref(cm->refframe->recons->V, image_ref_V, cm->vpw, cm->vph, pitch_ref_V);
	catchCudaError("TEXTURE");

	cuda_me_texture<<<block_dim_chroma, thread_dim>>>(cm->vpw, cm->vph, mb_dev);

	cudaMemcpy(cm->curframe->mbs[2], mb_dev, cm->mb_cols * cm->mb_rows / 4 * sizeof(*mb_dev), cudaMemcpyDeviceToHost);

	cudaFree(image_orig_Y);
	cudaFree(image_ref_Y);
	cudaFree(image_orig_U);
	cudaFree(image_ref_U);
	cudaFree(image_orig_V);
	cudaFree(image_ref_V);
	cudaFree(mb_dev);
}

