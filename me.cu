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

void testEqual(uint16_t *results, uint8_t *ref,int width,int height, int result_stride, int ref_stride) {
	uint16_t *res_host = (uint16_t*)malloc(height*result_stride*sizeof(uint16_t));
	uint8_t *ref_host = (uint8_t*)malloc(height*ref_stride*sizeof(uint8_t));

	cudaMemcpy(res_host,results, height*result_stride*sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(ref_host,ref, height*ref_stride*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	int x = 0, y = 0;
	for(y = 0; y < height; y++) {
		printf("\nROW %d= ", y);
		for(x = 0; x < width;x++) {
			const char *c = ref_host[y*ref_stride+x] == res_host[y*result_stride+x] ? "C" : "W";
			printf("%s",c);
		}
	}
	printf("\n");
	free(res_host);
	free(ref_host);
	exit(-1);
}


void printResult(uint16_t *results,int width,int height, int result_stride) {
	uint16_t *res_host = (uint16_t*)malloc(height*result_stride*sizeof(uint16_t));

	cudaMemcpy(res_host,results, height*result_stride*sizeof(uint16_t), cudaMemcpyDeviceToHost);
	int x = 0, y = 0;
	for(y = 0; y < height; y++) {
		printf("ROW %d= ", y);
		for(x = 0; x < width;x++) {
			printf("%d, ", res_host[y*result_stride+x]);
		}
		printf("\n");
	}
	free(res_host);
}



extern "C" void c63_motion_estimate(struct c63_common *cm)
{
    /* Compare this frame with previous reconstructed frame */
	uint8_t *image_orig, *image_ref;
	int size = cm->width*cm->height;
    macroblock *mb_dev;
	cudaMalloc((void**)&image_orig, size * sizeof(uint8_t));
	cudaMalloc((void**)&image_ref, size * sizeof(uint8_t));
	cudaMalloc((void**)&mb_dev, cm->mb_cols * cm->mb_rows * sizeof(*mb_dev));

    // Luma
	cudaMemcpy(image_orig, cm->curframe->orig->Y, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref, cm->refframe->recons->Y, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

	dim3 thread_dim(8,8,4);
	dim3 block_dim(cm->mb_cols, cm->mb_rows,1);
	cuda_me2<<<block_dim, thread_dim>>>(image_orig, image_ref, cm->width,mb_dev);

	cudaMemcpy(cm->curframe->mbs[0], mb_dev, cm->mb_cols*cm->mb_rows*sizeof(*mb_dev), cudaMemcpyDeviceToHost);

	// Chroma
	cudaMemcpy(image_orig, cm->curframe->orig->U, size/4 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref, cm->refframe->recons->U, size/4 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	dim3 block_dim_chroma(cm->mb_cols / 2, cm->mb_rows / 2,1);
	cuda_me2<<<block_dim_chroma, thread_dim>>>(image_orig, image_ref, cm->width,mb_dev);

	cudaMemcpy(cm->curframe->mbs[1], mb_dev, cm->mb_cols*cm->mb_rows/4*sizeof(*mb_dev), cudaMemcpyDeviceToHost);

	cudaMemcpy(image_orig, cm->curframe->orig->V, size/4 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref, cm->refframe->recons->V, size/4 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cuda_me2<<<block_dim_chroma, thread_dim>>>(image_orig, image_ref, cm->width,mb_dev);

	cudaMemcpy(cm->curframe->mbs[2], mb_dev, cm->mb_cols*cm->mb_rows/4*sizeof(*mb_dev), cudaMemcpyDeviceToHost);


    cudaFree(image_orig);
    cudaFree(image_ref);
    cudaFree(mb_dev);
}

/* Motion compensation for 8x8 block */
__host__
void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int cc)
{
    struct macroblock *mb = &cm->curframe->mbs[cc][mb_y * cm->padw[cc]/8 + mb_x];

    if (!mb->use_mv)
        return;

    int left = mb_x*8;
    int top = mb_y*8;
    int right = left + 8;
    int bottom = top + 8;

    int w = cm->padw[cc];

    /* Copy block from ref mandated by MV */
    int x,y;
    for (y=top; y < bottom; ++y)
    {
        for (x=left; x < right; ++x)
        {
            predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
        }
    }
}

extern void c63_motion_compensate(struct c63_common *cm)
{
    int mb_x, mb_y;

    /* Luma */
    for (mb_y=0; mb_y < cm->mb_rows; ++mb_y)
    {
        for (mb_x=0; mb_x < cm->mb_cols; ++mb_x)
        {
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, 0);
        }
    }

    /* Chroma */
    for (mb_y=0; mb_y < cm->mb_rows/2; ++mb_y)
    {
        for (mb_x=0; mb_x < cm->mb_cols/2; ++mb_x)
        {
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, 1);
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, 2);
        }
    }
}

}
