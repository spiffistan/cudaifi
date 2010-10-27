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

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;

    int u,v;
    for (v=0; v<8; ++v)
        for (u=0; u<8; ++u)
            *result += abs(block2[v*stride+u] - block1[v*stride+u]);
}


/* Motion estimation for 8x8 block */
static void pre_me_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int cc, uint16_t *test, uint16_t *best_sad)
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
    if (left < 0)
        left = 0;
    if (top < 0)
        top = 0;
    if (right > (w - 8))
        right = w - 8;
    if (bottom > (h - 8))
        bottom = h - 8;


    int x,y;
    int mx = mb_x * 8;
    int my = mb_y * 8;

    *best_sad = 60000;
    int i = 0;
    int j = 0;
    for (y=top; y<bottom; ++y)
    {
        for (x=left; x<right; ++x)
        {
            int sad;
            sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);
            test[(i * 40) + j] = sad;
            j++;
            //printf("(%4d,%4d) - %d\n", x, y, sad);

            if (sad < *best_sad)
            {
                mb->mv_x = x - mx;
                mb->mv_y = y - my;
                *best_sad = sad;
            }
        }
        i++;
        j = 0;
    }
    /* Here, there should be a threshold on SAD that checks if the motion vector is
     * cheaper than intraprediction. We always assume MV to be beneficial */

//    printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad);

    mb->use_mv = 1;
}
