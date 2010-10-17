#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include "c63.h"
#include "tables.h"

void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h, int y,
			 uint8_t *out_data, uint8_t *quantization)
{
    int x;

    int16_t block[8*8];

    /* Perform the dequantization and iDCT */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        dequant_idct_block_8x8(in_data+(x*8), block, quantization);


        for (i=0; i<8; ++i)
            for (j=0; j<8; ++j)
            {
                /* Add prediction block. Note: DCT is not precise - Clamp to legal values */
                int16_t tmp = block[i*8+j] + (int16_t)prediction[i*w+j+x];
                if (tmp < 0)
                    tmp = 0;
                else if (tmp > 255)
                    tmp = 255;

                out_data[i*w+j+x] = tmp;
            }
    }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
			 uint8_t *out_data, uint8_t *quantization)
{
    int y;
    for (y=0; y<height; y+=8)
    {
        dequantize_idct_row(in_data+y*width, prediction+y*width, width, height, y, out_data+y*width, quantization);
    }
}

void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w, int h,
        int16_t *out_data, uint8_t *quantization)
{
    int x;

    int16_t block[8*8];

    /* Perform the DCT and quantization */
    for(x = 0; x < w; x += 8)
    {
        int i,j;
        for (i=0; i<8; ++i)
            for (j=0; j<8; ++j)
                block[i*8+j] = ((int16_t)in_data[i*w+j+x] - prediction[i*w+j+x]);

        /* Store MBs linear in memory, i.e. the 64 coefficients are stored continous.
         * This allows us to ignore stride in DCT/iDCT and other functions. */
        dct_quant_block_8x8(block, out_data+(x*8), quantization);
    }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t *quantization)
{
    int y;
    for (y=0; y<height; y+=8)
    {
        dct_quantize_row(in_data+y*width, prediction+y*width, width, height, out_data+y*width, quantization);
    }
}

void destroy_frame(struct frame *f)
{
    if (!f) // First frame
        return;

    free(f->recons->Y);
    free(f->recons->U);
    free(f->recons->V);
    free(f->recons);

    free(f->residuals->Ydct);
    free(f->residuals->Udct);
    free(f->residuals->Vdct);
    free(f->residuals);

    free(f->predicted->Y);
    free(f->predicted->U);
    free(f->predicted->V);
    free(f->predicted);

    free(f->mbs[0]);
    free(f->mbs[1]);
    free(f->mbs[2]);

    free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
    struct frame *f = malloc(sizeof(struct frame));

    f->orig = image;

    f->recons = malloc(sizeof(yuv_t));
    f->recons->Y = malloc(cm->ypw * cm->yph);
    f->recons->U = malloc(cm->upw * cm->uph);
    f->recons->V = malloc(cm->vpw * cm->vph);

    f->predicted = malloc(sizeof(yuv_t));
    f->predicted->Y = calloc(cm->ypw * cm->yph, sizeof(uint8_t));
    f->predicted->U = calloc(cm->upw * cm->uph, sizeof(uint8_t));
    f->predicted->V = calloc(cm->vpw * cm->vph, sizeof(uint8_t));

    f->residuals = malloc(sizeof(dct_t));
    f->residuals->Ydct = calloc(cm->ypw * cm->yph, sizeof(int16_t));
    f->residuals->Udct = calloc(cm->upw * cm->uph, sizeof(int16_t));
    f->residuals->Vdct = calloc(cm->vpw * cm->vph, sizeof(int16_t));

    f->mbs[0] = calloc(cm->ypw * cm->yph, sizeof(struct macroblock));
    f->mbs[1] = calloc(cm->upw * cm->uph, sizeof(struct macroblock));
    f->mbs[2] = calloc(cm->vpw * cm->vph, sizeof(struct macroblock));

    return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
    fwrite(image->Y, 1, w*h, fp);
    fwrite(image->U, 1, w*h/4, fp);
    fwrite(image->V, 1, w*h/4, fp);
}

