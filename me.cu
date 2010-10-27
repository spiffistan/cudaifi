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
#include "me.hcu"
#include "dsp.hcu"
extern "C" {

/* Motion estimation for 8x8 block */

void testEqual(uint16_t *results, uint8_t *ref, int width, int height, int result_stride, int ref_stride) {
	uint16_t *res_host = (uint16_t*) malloc(height * result_stride * sizeof(uint16_t));
	uint8_t *ref_host = (uint8_t*) malloc(height * ref_stride * sizeof(uint8_t));

	cudaMemcpy(res_host, results, height * result_stride * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(ref_host, ref, height * ref_stride * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	int x = 0, y = 0;
	for (y = 0; y < height; y++) {
		printf("\nROW %d= ", y);
		for (x = 0; x < width; x++) {
			const char *c = ref_host[y * ref_stride + x] == res_host[y * result_stride + x] ? "C" : "W";
			printf("%s", c);
		}
	}
	printf("\n");
	free(res_host);
	free(ref_host);
	exit(-1);
}

void printResult(uint16_t *results, int width, int height, int result_stride) {
	uint16_t *res_host = (uint16_t*) malloc(height * result_stride * sizeof(uint16_t));

	cudaMemcpy(res_host, results, height * result_stride * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	int x = 0, y = 0;
	for (y = 0; y < height; y++) {
		printf("ROW %d= ", y);
		for (x = 0; x < width; x++) {
			printf("%d, ", res_host[y * result_stride + x]);
		}
		printf("\n");
	}
	free(res_host);
}

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

/* Motion compensation for 8x8 block */
__host__
void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int cc) {
	struct macroblock *mb = &cm->curframe->mbs[cc][mb_y * cm->padw[cc] / 8 + mb_x];

	if (!mb->use_mv)
		return;

	int left = mb_x * 8;
	int top = mb_y * 8;
	int right = left + 8;
	int bottom = top + 8;

	int w = cm->padw[cc];

	/* Copy block from ref mandated by MV */
	int x, y;
	for (y = top; y < bottom; ++y) {
		for (x = left; x < right; ++x) {
			predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
		}
	}
}

extern void c63_motion_compensate(struct c63_common *cm) {
	int mb_x, mb_y;

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y) {
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x) {
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, 0);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y) {
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x) {
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, 1);
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, 2);
		}
	}
}

}
