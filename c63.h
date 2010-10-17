#ifndef CPU_MJPEG_H
#define CPU_MJPEG_H

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define ISQRT2 0.70710678118654f
#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
  uint8_t *Y;
  uint8_t *U;
  uint8_t *V;

  uint8_t *Ybase;
  uint8_t *Ubase;
  uint8_t *Vbase;
};

struct dct
{
  int16_t *Ydct;
  int16_t *Udct;
  int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
    FILE *fp;
    unsigned int bit_buffer;
    unsigned int bit_buffer_width;
};

struct macroblock
{
    int use_mv;
    int8_t mv_x, mv_y;
};

struct frame
{
    yuv_t *orig;        // Original input image
    yuv_t *recons;      // Reconstructed image
    yuv_t *predicted;   // Predicted frame from intra-prediction

    dct_t *residuals;   // Difference between original image and predicted frame

    // Meh

    struct macroblock *mbs[3];

    int keyframe;
};

struct c63_common
{
    int width, height;
    int ypw, yph, upw, uph, vpw, vph;

    int padw[3], padh[3];

    int mb_cols, mb_rows;

    uint8_t qp;                         // Quality parameter

    int me_search_range;


    uint8_t quanttbl[3][64];

    struct frame *refframe;
    struct frame *curframe;

    int framenum;

    int keyframe_interval;
    int frames_since_keyframe;

    struct entropy_ctx e_ctx;
};


void put_bytes(FILE *fp, const void* data, unsigned int len);
void put_byte(FILE *fp, int byte);
void put_bits(struct entropy_ctx *c, uint16_t bits, uint8_t n);
void flush_bits(struct entropy_ctx *c);
uint8_t get_byte(FILE *fp);
int read_bytes(FILE *fp, void *data, unsigned int sz);
uint16_t get_bits(struct entropy_ctx *c, uint8_t n);

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl);
void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl);
void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result);

void write_frame(struct c63_common *cm);
void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
			 uint8_t *out_data, uint8_t *quantization);
void dct_quantize(uint8_t *in_data, uint8_t *prediction,
        uint32_t width, uint32_t height,
        int16_t *out_data, uint8_t *quantization);

void destroy_frame(struct frame *f);
struct frame* create_frame(struct c63_common *cm, yuv_t *image);
void c63_motion_estimate(struct c63_common *cm);
void c63_motion_compensate(struct c63_common *cm);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif /* mjpeg_encoder.h */
