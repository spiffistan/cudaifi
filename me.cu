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
#define COMPSAD(i,j); \
  minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value); \
  minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);

struct min_helper {
	uint16_t value;
	int8_t x;
	int8_t y;
};

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS DEVICE ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


__shared__ uint32_t ref[LENGTH * LENGTH]; // <= (40) * (40)
__shared__ uint32_t orig[64];
__shared__ min_helper minimum[32 * 32];

__device__
inline void load_texture_values(int left, int top, int ref_index) {
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
	} else if(TX < 8) {
		int y = TY - 8;
		orig[y * 8 + TX] = tex2D(tex_orig, MX + TX, MY + y);
	}
	__syncthreads();
}

__device__
inline void calculate_usad(int res_index, int ref_index) {
#pragma unroll
	for (int j = 0; j < 8; j++) {
		#pragma unroll
		for (int i = 0; i < 8; i++) {
			minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value);
			minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);

		}
	}
	__syncthreads();
}

__device__
inline void setup_min(int res_index) {
	minimum[res_index].x = TX;
	minimum[res_index].y = TY;
	minimum[res_index].value = 0;
	minimum[32 * 16 + res_index].x = TX;
	minimum[32 * 16 + res_index].y = 16 + TY;
	minimum[32 * 16 + res_index].value = 0;

	__syncthreads();
}

#define MIN2(a,b) (a.value) > (b.value) ? (b) : (a);
#define COMPMIN(idx) minimum[res_index] = MIN2(minimum[res_index], minimum[(idx)]);

__device__
inline void reduce_min(int res_index) {
	COMPMIN(16 * 32 + res_index);
	__syncthreads(); // reduce to 2 block_rows
	if (TY < 8) COMPMIN(8 * 32 + res_index);
	__syncthreads(); // reduce to 1 block_row
	if (TY < 4) COMPMIN(4 * 32 + res_index);
	__syncthreads(); // reduce to 4 rows
	if (TY < 2) COMPMIN(2 * 32 + res_index);
	__syncthreads(); // reduce to 2 rows
	if (TY == 0) COMPMIN(32 + res_index); // reduce to 1 row, no need to sync anymore, within 1 warp
	if (TY == 0 && TX < 16) COMPMIN(16 + res_index); // reduce to 16 values
	if (TY == 0 && TX < 8) COMPMIN(8 + res_index); // reduce to 8 values
	if (TY == 0 && TX < 4) COMPMIN(4 + res_index); // reduce to 4 values
	if (TY == 0 && TX < 2) COMPMIN(2 + res_index); // reduce to 2 values
	if (TY == 0 && TX == 0) COMPMIN(1); // reduce to 1 value
	__syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS GLOBAL ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

__global__
void cuda_me_texture(int width, int height, size_t pitch, macroblock * mb, uint8_t *prediction) {
	int left = MX - 16;
	int top = MY - 16;

	int right = MX + 16;
	int bottom = MY + 16;

	if (left < 0) left = 0;

	if (top < 0) top = 0;

	if (right > (width - 8)) // Increase search area towards the left if we're out of bounds
	left += (width - 8) - right;

	if (bottom > (height - 8)) // Increase search area towards the top if we're out of bounds
	top += (height - 8) - bottom;

	int res_index = TY * 32 + TX;
	int ref_index = TY * 40 + TX;

	load_texture_values(left, top, ref_index);
	setup_min(res_index);
	calculate_usad(res_index, ref_index);
	reduce_min(res_index);
	if (TX == 0 && TY == 10) {
		mb[blockIdx.y * gridDim.x + blockIdx.x].mv_x = minimum[0].x + (left - MX);
		mb[blockIdx.y * gridDim.x + blockIdx.x].mv_y = minimum[0].y + (top - MY);
		mb[blockIdx.y * gridDim.x + blockIdx.x].use_mv = 1;
	} else if (TX < 8 && TY < 8) {
		prediction[(BY * 8 + TY)  * pitch + BX * 8 + TX] = ref[(minimum[0].y + TY) * 40 + minimum[0].x + TX];
	}
}

/* Motion estimation */
extern "C" void c63_motion_estimate(struct c63_common *cm, struct cuda_frame *cframe) {
	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->Y, &tex_ref.channelDesc, cm->ypw, cm->yph, cframe->last_recons_pitch[0]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->Y, &tex_orig.channelDesc, cm->ypw, cm->yph, cframe->image_pitch[0]);

	cuda_me_texture<<<cframe->me_blockDim_Y, cframe->me_threadDim,0,cframe->stream>>>(cm->ypw, cm->yph, cframe->image_pitch[0], cframe->mbs[0], cframe->predicted->Y);

	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->U, &tex_ref.channelDesc, cm->upw, cm->uph, cframe->last_recons_pitch[1]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->U, &tex_orig.channelDesc, cm->upw, cm->uph, cframe->image_pitch[1]);
	cuda_me_texture<<<cframe->me_blockDim_UV, cframe->me_threadDim,0,cframe->stream>>>(cm->upw, cm->uph,cframe->image_pitch[1],  cframe->mbs[1], cframe->predicted->U);

	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->V, &tex_ref.channelDesc, cm->vpw, cm->vph, cframe->last_recons_pitch[2]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->V, &tex_orig.channelDesc, cm->vpw, cm->vph, cframe->image_pitch[2]);
	cuda_me_texture<<<cframe->me_blockDim_UV, cframe->me_threadDim,0,cframe->stream>>>(cm->vpw, cm->vph, cframe->image_pitch[2], cframe->mbs[2], cframe->predicted->V);
}

