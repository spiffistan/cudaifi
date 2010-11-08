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

#define BIX (TY * 4 + TZ)
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

typedef struct point {
	int16_t x;
	int16_t y;
} point_t;

__shared__ uint8_t orig[16][64];
__shared__ min_helper current_best[16];
__shared__ min_helper current[16][16];
__shared__ point_t direction[16][16];

__device__
int16_t reducesum() {
	if (TX < 4) {
		current[BIX][TX].value += current[BIX][TX + 4].value;
	}
	__syncthreads();

	if (TX < 2) {
		current[BIX][TX].value += current[BIX][TX + 2].value;
	}
	__syncthreads();

	if (TX == 0) {
		current[BIX][TX].value += current[BIX][TX + 1].value;
	}
	__syncthreads();

	return current[BIX][0].value;
}

__device__
void reducemin() {
	if (TX < 4) {
		if (current[BIX][TX].value > current[BIX][TX + 4].value) {
			current[BIX][TX] = current[BIX][TX + 4];
		}
	}
	if (TX < 2) {
		if (current[BIX][TX].value > current[BIX][TX + 2].value) {
			current[BIX][TX] = current[BIX][TX + 2];
		}
	}
	if (TX < 1) {
		if (current[BIX][TX].value > current[BIX][TX + 1].value) {
			current[BIX][TX] = current[BIX][TX + 1];
		}
		if (current[BIX][0].value < current_best[BIX].value) {
			current_best[BIX] = current[BIX][0];
		}
	}
}

__device__
void log_step(int step_size, point_t orig_pos, int width, int height) {
	current[BIX][TX].x = current_best[BIX].x + step_size * direction[BIX][TX].x;
	current[BIX][TX].y = current_best[BIX].y + step_size * direction[BIX][TX].y;
	point_t real_pos;
	real_pos.x = orig_pos.x + current[BIX][TX].x;
	real_pos.y = orig_pos.y + current[BIX][TX].y;
	//calculate USAD
	current[BIX][TX].value = 0;
	#pragma unroll
	for (int i = 0; i < 8; i++) {
		#pragma unroll
		for (int j = 0; j < 8; j++) {
			current[BIX][TX].value = __usad(orig[BIX][i * 8 + j], tex2D(tex_ref, orig_pos.x + current[BIX][TX].x + j, orig_pos.y + current[BIX][TX].y + i), current[BIX][TX].value);
		}
	}
	if (real_pos.x < 0 || real_pos.y < 0 || real_pos.x >= (width - 8) || real_pos.y >= (height - 8)) {
		current[BIX][TX].value = 65535;
	}
	__syncthreads();
	reducemin();

}
///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS GLOBAL ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//threadDim(8,1,1)

__global__
void cuda_me_log(int width, int height, int mb_width, int mb_height, macroblock * mb, uint8_t *prediction) {

	point_t orig_pos;
	orig_pos.x = (BX * DY + TY) * 8;
	orig_pos.y = (BY * DZ + TZ) * 8;

	if (orig_pos.x > (width - 8) || orig_pos.y > (height - 8))
		return;
	//load orig
	for (int i = 0; i < 8; i++) {
		orig[BIX][i * 8 + TX] = tex2D(tex_orig, orig_pos.x + TX, orig_pos.y + i);
	}
	__syncthreads();

	current[BIX][TX].value = 0;
	//calculate center SAD, use current to store intermediate
	for (int i = 0; i < 8; i++) {
		current[BIX][TX].value = __usad(orig[BIX][i * 8 + TX], tex2D(tex_ref, orig_pos.x + TX, orig_pos.y + i), current[BIX][TX].value);
	}
	//reduce it
	current_best[BIX].value = reducesum();
	current_best[BIX].x = 0;
	current_best[BIX].y = 0;

	//set directions
	if (TX < 3) {
		direction[BIX][TX].y = -1;
	} else if (TX > 4) {
		direction[BIX][TX].y = 1;
	} else {
		direction[BIX][TX].y = 0;
	}

	if (TX == 0 || TX == 3 || TX == 5) {
		direction[BIX][TX].x = -1;
	} else if (TX == 2 || TX == 4 || TX == 7) {
		direction[BIX][TX].x = 1;
	} else {
		direction[BIX][TX].x = 0;
	}

	log_step(8, orig_pos, width, height);
	log_step(4, orig_pos, width, height);
	log_step(2, orig_pos, width, height);
	log_step(1, orig_pos, width, height);

	if (TX == 0) {
#define MB_IX (BY * DZ + TZ) * mb_width + BX * DY + TY
		mb[MB_IX].mv_x = current_best[BIX].x;
		mb[MB_IX].mv_y = current_best[BIX].y;
		mb[MB_IX].use_mv = 1;
	}
	__syncthreads();

	for (int i = 0; i < 8; i++) {
		prediction[(BY * DZ + TZ) * 8 * width + (BX * DY + TY) * 64 + i * 8 + TX] = tex2D(tex_ref, orig_pos.x + current_best[BIX].x + TX, orig_pos.y + current_best[BIX].y + i);
	}

}

/* Motion estimation */
extern "C" void c63_motion_estimate_log(struct c63_common *cm, struct cuda_frame *cframe) {
	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->Y, &tex_ref.channelDesc, cm->ypw, cm->yph, cframe->last_recons_pitch[0]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->Y, &tex_orig.channelDesc, cm->ypw, cm->yph, cframe->image_pitch[0]);
	cuda_me_log<<<cframe->me_blockDim_Y, cframe->me_threadDim,0,cframe->stream>>>(cm->ypw, cm->yph, cframe->mb_width_Y, cframe->mb_height_Y, cframe->mbs[0], cframe->predicted->Y);

	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->U, &tex_ref.channelDesc, cm->upw, cm->uph, cframe->last_recons_pitch[1]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->U, &tex_orig.channelDesc, cm->upw, cm->uph, cframe->image_pitch[1]);
	cuda_me_log<<<cframe->me_blockDim_UV, cframe->me_threadDim,0,cframe->stream>>>(cm->upw, cm->uph,cframe->mb_width_UV, cframe->mb_height_UV, cframe->mbs[1], cframe->predicted->U);

	cudaBindTexture2D(0, &tex_ref, cframe->last_recons->V, &tex_ref.channelDesc, cm->vpw, cm->vph, cframe->last_recons_pitch[2]);
	cudaBindTexture2D(0, &tex_orig, cframe->image->V, &tex_orig.channelDesc, cm->vpw, cm->vph, cframe->image_pitch[2]);
	cuda_me_log <<<cframe->me_blockDim_UV, cframe->me_threadDim,0,cframe->stream>>>(cm->vpw, cm->vph,cframe->mb_width_UV, cframe->mb_height_UV, cframe->mbs[2],cframe->predicted->V);
}

