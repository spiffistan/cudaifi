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

int frequencies[2][12];


static void write_SOI(struct c63_common *cm)
{
    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xd8);
}

static void write_DQT(struct c63_common *cm)
{
    int16_t size = 2 + (3 * 65);

    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xdb);

    put_byte(cm->e_ctx.fp, size >> 8);
    put_byte(cm->e_ctx.fp, size & 0xff);

    put_byte(cm->e_ctx.fp, 0);
    put_bytes(cm->e_ctx.fp, cm->quanttbl[0], 64);

    put_byte(cm->e_ctx.fp, 1);
    put_bytes(cm->e_ctx.fp, cm->quanttbl[1], 64);

    put_byte(cm->e_ctx.fp, 2);
    put_bytes(cm->e_ctx.fp, cm->quanttbl[2], 64);
}

static void write_SOF0(struct c63_common *cm)
{
    int16_t size = 8 + 3 * COLOR_COMPONENTS + 1;

    /* Header marker */
    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xc0);

    /* Size of header */
    put_byte(cm->e_ctx.fp, size >> 8);
    put_byte(cm->e_ctx.fp, size & 0xff);

    /* Precision */
    put_byte(cm->e_ctx.fp, 8);

    /* Width and height */
    put_byte(cm->e_ctx.fp, cm->height >> 8);
    put_byte(cm->e_ctx.fp, cm->height & 0xff);
    put_byte(cm->e_ctx.fp, cm->width >> 8);
    put_byte(cm->e_ctx.fp, cm->width & 0xff);

    put_byte(cm->e_ctx.fp, COLOR_COMPONENTS);

    put_byte(cm->e_ctx.fp, 1); /* Component id */
    put_byte(cm->e_ctx.fp, 0x22); /* hor | ver sampling factor FIXME Y(2,2), U(1,1), V(1,1) */
    put_byte(cm->e_ctx.fp, 0); /* Quant. tbl. id */

    put_byte(cm->e_ctx.fp, 2); /* Component id */
    put_byte(cm->e_ctx.fp, 0x11); /* hor | ver sampling factor */
    put_byte(cm->e_ctx.fp, 1); /* Quant. tbl. id */

    put_byte(cm->e_ctx.fp, 3); /* Component id */
    put_byte(cm->e_ctx.fp, 0x11); /* hor | ver sampling factor */
    put_byte(cm->e_ctx.fp, 2); /* Quant. tbl. id */

    /* Is this a keyframe or not? */
    put_byte(cm->e_ctx.fp, cm->curframe->keyframe);
}

static void write_DHT_HTS(struct c63_common *cm, uint8_t id, uint8_t *numlength, uint8_t* data)
{
    /* Find out how many codes we are to write */
    int i;
    int n = 0;
    for(i = 0; i < 16; ++i)
        n += numlength[i];

    put_byte(cm->e_ctx.fp, id);
    put_bytes(cm->e_ctx.fp, numlength, 16);
    put_bytes(cm->e_ctx.fp, data, n);
}

static void write_DHT(struct c63_common *cm)
{
    int16_t size = 0x01A2; /* 2 + n*(17+mi); */

    /* Define Huffman Table marker */
    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xc4);

    /* Length of segment */
    put_byte(cm->e_ctx.fp, size >> 8);
    put_byte(cm->e_ctx.fp, size & 0xff);

    /* Write the four huffman table specifications */
    write_DHT_HTS(cm, 0x00, DCVLC_num_by_length[0], DCVLC_data[0]); /* DC table 0 */
    write_DHT_HTS(cm, 0x01, DCVLC_num_by_length[1], DCVLC_data[1]); /* DC table 1 */
    write_DHT_HTS(cm, 0x10, ACVLC_num_by_length[0], ACVLC_data[0]); /* AC table 0 */
    write_DHT_HTS(cm, 0x11, ACVLC_num_by_length[1], ACVLC_data[1]); /* AC table 1 */
}

static void write_SOS(struct c63_common *cm)
{
    int16_t size = 6 + 2 * COLOR_COMPONENTS;

    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xda);

    put_byte(cm->e_ctx.fp, size >> 8);
    put_byte(cm->e_ctx.fp, size & 0xff);

    put_byte(cm->e_ctx.fp, COLOR_COMPONENTS);

    put_byte(cm->e_ctx.fp, 1); /* Component id */
    put_byte(cm->e_ctx.fp, 0x00); /* DC | AC huff tbl */
    put_byte(cm->e_ctx.fp, 2); /* Component id */
    put_byte(cm->e_ctx.fp, 0x11); /* DC | AC huff tbl */
    put_byte(cm->e_ctx.fp, 3); /* Component id */
    put_byte(cm->e_ctx.fp, 0x11); /* DC | AC huff tbl */
    put_byte(cm->e_ctx.fp, 0); /* ss, first AC */
    put_byte(cm->e_ctx.fp, 63); /* se, last AC */
    put_byte(cm->e_ctx.fp, 0); /* ah | al */
}

static void write_EOI(struct c63_common *cm)
{
    put_byte(cm->e_ctx.fp, 0xff);
    put_byte(cm->e_ctx.fp, 0xd9);
}

static inline uint8_t bit_width(int16_t i)
{
    if (__builtin_expect(!i, 0))
        return 0;

    int r = 0;
    int v = abs(i);

    while(v >>= 1)
        ++r;

    return r+1;
}


static void write_block(struct c63_common *cm, int16_t *in_data, uint32_t width, uint32_t height,
        uint32_t uoffset, uint32_t voffset, int16_t *prev_DC,
        int32_t cc, int channel)
{
    uint32_t i, j;

    /* Write motion vector */
    struct macroblock *mb = &cm->curframe->mbs[channel][voffset/8 * cm->padw[channel]/8 + uoffset/8];

    /* Use inter pred? */
    put_bits(&cm->e_ctx, mb->use_mv, 1);

    if (mb->use_mv)
    {
        int reuse_prev_mv = 0;
        if (uoffset && (mb-1)->use_mv && (mb-1)->mv_x == mb->mv_x && (mb-1)->mv_y == mb->mv_y )
            reuse_prev_mv = 1;

        put_bits(&cm->e_ctx, reuse_prev_mv, 1);

        if (!reuse_prev_mv)
        {
            uint8_t sz;
            int16_t val;

            /* Encode MV x-coord */
            val = mb->mv_x;
            sz = bit_width(val);
            if (val < 0)
                --val;

            put_bits(&cm->e_ctx, MVVLC[sz], MVVLC_Size[sz]);
            put_bits(&cm->e_ctx, val, sz);
//            ++frequencies[cc][sz];

            /* Encode MV y-coord */
            val = mb->mv_y;
            sz = bit_width(val);
            if (val < 0)
                --val;

            put_bits(&cm->e_ctx, MVVLC[sz], MVVLC_Size[sz]);
            put_bits(&cm->e_ctx, val, sz);
//            ++frequencies[cc][sz];
        }
    }

    /* Write residuals */

    /* Residuals stored linear in memory */
    int16_t *block = &in_data[uoffset * 8 + voffset * width];
    int32_t num_ac = 0;

#if 0
    static int blocknum;
    ++blocknum;
    printf("Dump block %d:\n", blocknum);

    for(i=0; i<8; ++i) {
        for (j=0; j<8; ++j)
            printf(", %5d", block[i*8+j]);
        printf("\n");
    }
    printf("Finished block\n\n");
#endif

    /* Calculate DC component, and write to stream */
    int16_t dc = block[0] - *prev_DC;
    *prev_DC = block[0];
    uint8_t size = bit_width(dc);
    put_bits(&cm->e_ctx, DCVLC[cc][size],DCVLC_Size[cc][size]);

    if(dc < 0)
        dc = dc - 1;
    put_bits(&cm->e_ctx, dc, size);

    /* find the last nonzero entry of the ac-coefficients */
    for(j = 64; j > 1 && !block[j-1]; j--)
        ;

    /* Put the nonzero ac-coefficients */
    for(i = 1; i < j; i++)
    {
        int16_t ac = block[i];
        if(ac == 0)
        {
            if(++num_ac == 16)
            {
                put_bits(&cm->e_ctx, ACVLC[cc][15][0], ACVLC_Size[cc][15][0]);

                num_ac = 0;
            }
        }
        else
        {
            uint8_t size = bit_width(ac);
            put_bits(&cm->e_ctx, ACVLC[cc][num_ac][size], ACVLC_Size[cc][num_ac][size]);

            if(ac < 0)
                --ac;

            put_bits(&cm->e_ctx, ac, size);

            num_ac = 0;
        }
    }

    /* Put end of block marker */
    if(j < 64)
    {
        put_bits(&cm->e_ctx, ACVLC[cc][0][0], ACVLC_Size[cc][0][0]);
    }
}

static void write_interleaved_data_MCU(struct c63_common *cm, int16_t *dct, uint32_t wi, uint32_t he,
        uint32_t h, uint32_t v, uint32_t x,
        uint32_t y, int16_t *prev_DC, int32_t cc, int channel)
{
    uint32_t i, j, ii, jj;
    for(j = y*v*8; j < (y+1)*v*8; j += 8)
    {
        jj = he-8;
        jj = MIN(j, jj);

        for(i = x*h*8; i < (x+1)*h*8; i += 8)
        {
            ii = wi-8;
            ii = MIN(i, ii);

            write_block(cm, dct, wi, he, ii, jj, prev_DC, cc, channel);
        }
    }
}

static void write_interleaved_data(struct c63_common *cm)
{
    int16_t prev_DC[3] = {0, 0, 0};
    uint32_t u, v;

    /* Set up which huffman tables we want to use */
    int32_t yhtbl = 0;
    int32_t uhtbl = 1;
    int32_t vhtbl = 1;

    /* Find the number of MCU's for the intensity */
    uint32_t ublocks = (uint32_t) (ceil(cm->ypw/(float)(8.0f*YX)));
    uint32_t vblocks = (uint32_t) (ceil(cm->yph/(float)(8.0f*YY)));

    /* Write the MCU's interleaved */
    for(v = 0; v < vblocks; ++v)
    {
        for(u = 0; u < ublocks; ++u)
        {
            write_interleaved_data_MCU(cm, cm->curframe->residuals->Ydct, cm->ypw, cm->yph, YX, YY, u, v, &prev_DC[0], yhtbl, 0);
            write_interleaved_data_MCU(cm, cm->curframe->residuals->Udct, cm->upw, cm->uph, UX, UY, u, v, &prev_DC[1], uhtbl, 1);
            write_interleaved_data_MCU(cm, cm->curframe->residuals->Vdct, cm->vpw, cm->vph, VX, VY, u, v, &prev_DC[2], vhtbl, 2);
        }
    }

    flush_bits(&cm->e_ctx);
}

void write_frame(struct c63_common *cm)
{
    /* Write headers */

    /* Start Of Image */
    write_SOI(cm);
    /* Define Quantization Table(s) */
    write_DQT(cm);
    /* Start Of Frame 0(Baseline DCT) */
    write_SOF0(cm);
    /* Define Huffman Tables(s) */
    write_DHT(cm);
    /* Start of Scan */
    write_SOS(cm);

    write_interleaved_data(cm);

    /* End Of Image */
    write_EOI(cm);
}
