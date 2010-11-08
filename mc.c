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

/* Motion compensation for 8x8 block */
void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int cc)
{
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
	for (y = top; y < bottom; ++y)
	{
		for (x = left; x < right; ++x)
		{
			predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
		}
	}
}

void c63_motion_compensate(struct c63_common *cm)
{
	int mb_x, mb_y;

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
		{
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, 0);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, 1);
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, 2);
		}
	}
}
