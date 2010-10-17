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

#define MARKER_START 0xff
#define MARKER_DQT 0xdb
#define MARKER_SOI 0xd8
#define MARKER_SOS 0xda
#define MARKER_SOF0 0xc0
#define MARKER_EOI 0xd9
#define MARKER_DHT 0xc4


#define HUFF_AC_ZERO 16
#define HUFF_AC_SIZE 11

/* Decode VLC token */
static uint8_t get_vlc_token(struct entropy_ctx *c, uint16_t *table, uint8_t *table_sz, int tablelen)
{
    uint16_t bits = 0;

    int i, n;
    for (n=1; n <= 16; ++n)
    {
        bits <<= 1;
        bits |= get_bits(c, 1);

        /* See if this string matches a token in VLC table */
        for (i=0; i<tablelen; ++i)
        {
            if (table_sz[i] < n)
            {
                /* Too small token. */
                continue;
            }
            if (table_sz[i] == n)
            {
                if (bits == (table[i] & ((1 << n) - 1)))
                {
                    /* Found it */
                    return i;
                }
            }
        }
    }

    fprintf(stdout, "VLC token not found.\n");
    exit(1);
}

/* Decode AC VLC token.
   Optimized VLC decoder that takes advantage of the increasing order
   of our ACVLC_Size table. Does not work with a general table, so use the
   unoptimized get_vlc_token. A better approach would be to use an index table.
 */
static uint8_t get_vlc_token_ac(struct entropy_ctx *c, uint16_t table[HUFF_AC_ZERO][HUFF_AC_SIZE], uint8_t table_sz[HUFF_AC_ZERO][HUFF_AC_SIZE])
{
    uint16_t bits = 0;

    int n;
    int x,y;
    for (n=1; n <= 16; ++n)
    {
        bits <<= 1;
        bits |= get_bits(c, 1);

        uint16_t mask = (1 << n) - 1;

        for (x=1; x<HUFF_AC_SIZE; ++x)
        {
            for (y=0; y<HUFF_AC_ZERO; ++y)
            {
                if(table_sz[y][x] < n)
                    continue;
                else if (table_sz[y][x] > n)
                    break;
                else if (bits == (table[y][x] & mask))
                {
                    /* Found it */
                    return y*HUFF_AC_SIZE + x;
                }
            }

            if (table_sz[x][0] > n)
                break;
        }

        /* Check if it's a special token (bitsize 0) */
        for (y=0; y<HUFF_AC_ZERO; y+=(HUFF_AC_ZERO-1))
        {
            if (table_sz[y][0] == n && bits == (table[y][0] & mask))
                {
                    /* Found it */
                    return y*HUFF_AC_SIZE;
                }
        }
    }

    printf("VLC token not found (ac).\n");
    exit(1);
}



/* Decode sign of value from VLC. See Figure F.12 in spec. */
static int16_t extend_sign(int16_t v, int sz)
{
    int vt = 1 << (sz - 1);

    if (v >= vt)
        return v;

    int range = (1 << sz) - 1;

    v = -(range - v);

    return v;
}

static void read_block(struct c63_common *cm, int16_t *out_data,
        uint32_t width, uint32_t height, uint32_t uoffset, uint32_t voffset,
        int16_t *prev_DC, int32_t cc, int channel)
{
    uint8_t size;
    int i, num_zero=0;

    /* Read motion vector */
    struct macroblock *mb = &cm->curframe->mbs[channel][voffset/8 * cm->padw[channel]/8 + uoffset/8];

    /* Use inter pred? */
    mb->use_mv = get_bits(&cm->e_ctx, 1);

    if (mb->use_mv)
    {
        int reuse_prev_mv = get_bits(&cm->e_ctx, 1);
        if (reuse_prev_mv)
        {
            mb->mv_x = (mb-1)->mv_x;
            mb->mv_y = (mb-1)->mv_y;
        }
        else
        {
            int16_t val;
            size = get_vlc_token(&cm->e_ctx, MVVLC, MVVLC_Size, ARRAY_SIZE(MVVLC));
            val = get_bits(&cm->e_ctx, size);
            mb->mv_x = extend_sign(val, size);

            size = get_vlc_token(&cm->e_ctx, MVVLC, MVVLC_Size, ARRAY_SIZE(MVVLC));
            val = get_bits(&cm->e_ctx, size);
            mb->mv_y = extend_sign(val, size);
        }
    }

    /* Read residuals */

    // Linear block in memory
    int16_t *block = &out_data[uoffset * 8 + voffset * width];
    memset(block, 0, 64 * sizeof(int16_t));

    /* Decode DC */
    size = get_vlc_token(&cm->e_ctx, DCVLC[cc], DCVLC_Size[cc], ARRAY_SIZE(DCVLC[cc]));
    int16_t dc = get_bits(&cm->e_ctx, size);

    dc = extend_sign(dc, size);

    block[0] = dc + *prev_DC;
    *prev_DC = block[0];

    /* Decode AC RLE */
    for (i=1; i<64; ++i)
    {
        uint16_t token = get_vlc_token_ac(&cm->e_ctx, ACVLC[cc], ACVLC_Size[cc]);

        num_zero = token / 11;
        size = token % 11;

        i += num_zero;

        if (num_zero == 15 && size == 0)
        {
            continue;
        }

        if (num_zero == 0 && size == 0)
            break;

        int16_t ac = get_bits(&cm->e_ctx, size);

        block[i] = extend_sign(ac, size);
    }

#if 0
    int j;
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
}

static void read_interleaved_data_MCU(struct c63_common *cm, int16_t *dct, uint32_t wi, uint32_t he,
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

      read_block(cm, dct, wi, he, ii, jj, prev_DC, cc, channel);
    }
  }
}

void read_interleaved_data(struct c63_common *cm)
{
    int16_t prev_DC[3] = {0, 0, 0};
    int u,v;

    uint32_t ublocks = (uint32_t) (ceil(cm->ypw/(float)(8.0f*2)));
    uint32_t vblocks = (uint32_t) (ceil(cm->yph/(float)(8.0f*2)));

    /* Write the MCU's interleaved */
    for(v = 0; v < vblocks; ++v)
    {
        for(u = 0; u < ublocks; ++u)
        {
            read_interleaved_data_MCU(cm, cm->curframe->residuals->Ydct, cm->ypw, cm->yph, YX, YY, u, v, &prev_DC[0], 0, 0);
            read_interleaved_data_MCU(cm, cm->curframe->residuals->Udct, cm->upw, cm->uph, UX, UY, u, v, &prev_DC[1], 1, 1);
            read_interleaved_data_MCU(cm, cm->curframe->residuals->Vdct, cm->vpw, cm->vph, VX, VY, u, v, &prev_DC[2], 1, 2);
        }
    }
}

// Define quantization tables
void parse_dqt(struct c63_common *cm)
{
    uint16_t size;
    size = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);

    int i;
    for (i=0; i<3; ++i)
    {
        int idx = get_byte(cm->e_ctx.fp);
        if (idx != i)
        {
            fprintf(stderr, "DQT: Expected %d - got %d\n", i, idx);
            exit(1);
        }

        read_bytes(cm->e_ctx.fp, cm->quanttbl[i], 64);
    }
}

// Start of scan
void parse_sos(struct c63_common *cm)
{
    uint16_t size;
    size = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);

    /* Don't care currently */

    uint8_t buf[size];
    read_bytes(cm->e_ctx.fp, buf, size-2);
}

// Baseline DCT
void parse_sof0(struct c63_common *cm)
{
    uint16_t size;
    size = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);


    uint8_t precision = get_byte(cm->e_ctx.fp);
    if (precision != 8)
    {
        printf("Only 8-bit precision supported\n");
        exit(1);
    }

    uint16_t height = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);
    uint16_t width = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);

    // Discard subsampling info. We assume 4:2:0
    uint8_t buf[10];
    read_bytes(cm->e_ctx.fp, buf, 10);


    /* First frame? */
    if (cm->framenum == 0)
    {
        cm->width = width;
        cm->height = height;

        cm->padw[0] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
        cm->padh[0] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
        cm->padw[1] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
        cm->padh[1] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
        cm->padw[2] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
        cm->padh[2] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

        cm->mb_cols = cm->ypw / 8;
        cm->mb_rows = cm->yph / 8;

        cm->curframe = 0;
    }


    /* Advance to next frame */
    destroy_frame(cm->refframe);
    cm->refframe = cm->curframe;
    cm->curframe = create_frame(cm, 0);

    /* Is this a keyframe */
    cm->curframe->keyframe = get_byte(cm->e_ctx.fp);
}

// Define Huffman tables
void parse_dht(struct c63_common *cm)
{
    uint16_t size;
    size = (get_byte(cm->e_ctx.fp) << 8) | get_byte(cm->e_ctx.fp);

    // XXX: Should be handeled properly. However, we currently only use static tables
    uint8_t buf[size];
    read_bytes(cm->e_ctx.fp, buf, size-2);
}

int parse_c63_frame(struct c63_common *cm)
{
    // SOI
    if (get_byte(cm->e_ctx.fp) != MARKER_START || get_byte(cm->e_ctx.fp) != MARKER_SOI)
    {
        fprintf(stderr, "Not an JPEG file\n");
        exit(1);
    }

    while(1)
    {
        int c;
        c = get_byte(cm->e_ctx.fp);

        if (c == 0)
            c = get_byte(cm->e_ctx.fp);

        if (c != MARKER_START)
        {
            fprintf(stderr, "Expected marker.\n");
            exit(1);
        }

        uint8_t marker = get_byte(cm->e_ctx.fp);

        if (marker == MARKER_DQT)
            parse_dqt(cm);
        else if (marker == MARKER_SOS) {
            parse_sos(cm);
            read_interleaved_data(cm);
            cm->e_ctx.bit_buffer = cm->e_ctx.bit_buffer_width = 0;
        }
        else if (marker == MARKER_SOF0)
            parse_sof0(cm);
        else if (marker == MARKER_DHT)
            parse_dht(cm);
        else if (marker == MARKER_EOI)
            return 1;
        else
        {
            fprintf(stderr, "Invalid marker: 0x%02x\n", marker);
            exit(1);
        }

    }

    return 1;
}


void decode_c63_frame(struct c63_common *cm, FILE *fout)
{
    /* Motion Compensation */
    if (!cm->curframe->keyframe)
        c63_motion_compensate(cm);

    /* Decode residuals */
    dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[0]);
    dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[1]);
    dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[2]);

    /* Write result */
    dump_image(cm->curframe->recons, cm->width, cm->height, fout);

    /* XXX: To dump the predicted frames, use this instead */
    //dump_image(cm->curframe->predicted, cm->width, cm->height, fout);

    ++cm->framenum;
}

static void print_help()
{
    fprintf(stderr, "Usage: ./c63dec input.c63 output.yuv\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Tip! Use vlc to playback raw YUV file: vlc --rawvid-width 352 --rawvid-height 288 foreman.yuv\n");
    fprintf(stderr, "\n");

    exit(EXIT_FAILURE);
}
int main(int argc, char **argv)
{

    if(argc < 3 || argc > 3)
    {
        print_help();
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) {
        perror("fopen");
        exit(1);
    }

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("fopen");
        exit(1);
    }

    struct c63_common *cm = calloc(1, sizeof(struct c63_common));
    cm->e_ctx.fp = fin;

    int framenum = 0;
    while(!feof(fin))
    {
        printf("Decoding frame %d\n", framenum++);

        parse_c63_frame(cm);
        decode_c63_frame(cm, fout);
    }


    fclose(fin);
    fclose(fout);

    return 0;
}
