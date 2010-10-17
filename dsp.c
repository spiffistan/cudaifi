#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "tables.h"

#define ISQRT2 0.70710678118654f

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

void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    *result = 0;

    int u,v;
    for (v=0; v<8; ++v)
        for (u=0; u<8; ++u)
            *result += abs(block2[v*stride+u] - block1[v*stride+u]);
}
