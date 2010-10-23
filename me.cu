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


void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int cc,uint16_t *result_sad, uint16_t *sums)
{
    struct macroblock *mb = &cm->curframe->mbs[cc][mb_y * cm->padw[cc]/8 + mb_x];
    int range = cm->me_search_range;

    int left = mb_x*8 - range;
    int top = mb_y*8 - range;
    int right = mb_x*8 + range;
    int bottom = mb_y*8 + range;

    int w = cm->padw[cc];
    int h = cm->padh[cc];

    /* Make sure we are within bounds of reference frame */
    // TODO: Support partial frame bounds
    bool rightEdge = false;
    bool bottomEdge = false;
    if (left < 0)
        left = 0;
    if (top < 0)
        top = 0;
    if (right > (w - 8)) {
        right = w - 8;
        rightEdge = true;
    }
    if (bottom > (h - 8)) {
        bottom = h - 8;
        bottomEdge = true;
    }


    int x,y;
    int mx = mb_x * 8;
    int my = mb_y * 8;

    int best_sad = INT_MAX;
    int search_width = right-left;
    int search_height = bottom-top;
    int num_mb_width = search_width / 8;
    int num_mb_height = search_height / 8;


    dim3 threadSize(8,8,4);

    cuda_me<<<1,threadSize>>>(orig + my*w+mx, ref + top*w+left, w, result_sad, num_mb_width,num_mb_height,rightEdge, bottomEdge);

    catchCudaError("FAILED happy_block_8x8");

	//print_buffer32(result_block1, 64*64);
	cudaMemcpy(sums,result_sad, 40*40 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	//#pragma unroll loop 64

    for (y=0; y<search_height; ++y)
    {
        for (x=0; x<search_width; ++x)
        {
            int sad = sums[y*40+x];
            //sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

            //printf("(%4d,%4d) %d\n", x, y, sad);

            if (sad < best_sad)
            {
                mb->mv_x = left + x - mx;
                mb->mv_y = top + y - my;
                best_sad = sad;
                //printf("new best sad for (%d,%d) @ (%4d,%4d) = %d\n", mb_x, mb_y, x, y, sad);
            }
        }
    }

    /* Here, there should be a threshold on SAD that checks if the motion vector is
     * cheaper than intraprediction. We always assume MV to be beneficial */

//    printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad);

    mb->use_mv = 1;
}

extern "C" void c63_motion_estimate(struct c63_common *cm)
{
    /* Compare this frame with previous reconstructed frame */
	uint8_t *image_orig, *image_ref, *image_orig_2, *image_ref_2;
	int size = cm->width*cm->height;

    uint16_t *result_sad, *sums;
    cudaMalloc((void**)&result_sad, 40*40 * sizeof(uint16_t));
	cudaMalloc((void**)&image_orig, size * sizeof(uint8_t));
	cudaMalloc((void**)&image_ref, size * sizeof(uint8_t));
	sums = (uint16_t*)malloc(40*40 * sizeof(uint16_t));

	cudaMemcpy(image_orig, cm->curframe->orig->Y, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref, cm->refframe->recons->Y, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Luma
	int mb_y, mb_x;
    for (mb_y=0; mb_y < cm->mb_rows; ++mb_y)
    {
        for (mb_x=0; mb_x < cm->mb_cols; ++mb_x)
        {
            me_block_8x8(cm, mb_x, mb_y, image_orig, image_ref, 0, result_sad, sums);
        }
    }

    cudaMalloc((void**)&image_orig_2, size * sizeof(uint8_t));
   	cudaMalloc((void**)&image_ref_2, size * sizeof(uint8_t));

	cudaMemcpy(image_orig, cm->curframe->orig->U, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref, cm->refframe->recons->U, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_orig_2, cm->curframe->orig->V, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(image_ref_2, cm->refframe->recons->V, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Chroma
    for (mb_y=0; mb_y < cm->mb_rows/2; ++mb_y)
    {
        for (mb_x=0; mb_x < cm->mb_cols/2; ++mb_x)
        {
            me_block_8x8(cm, mb_x, mb_y, image_orig, image_ref, 1, result_sad,sums);
            me_block_8x8(cm, mb_x, mb_y, image_orig_2, image_ref_2, 2,result_sad,sums);
        }
    }

    cudaFree(image_orig);
    cudaFree(image_orig_2);
    cudaFree(image_ref);
    cudaFree(image_ref_2);
    cudaFree(result_sad);
    free(sums);

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
