#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "tables.h"
#include "dsp.hcu"
#define ISQRT2 0.70710678118654f
extern "C" {
#include "c63.h"
static void transpose_block(float *in_data, float *out_data) {
    int i, j;
    for (i = 0; i < 8; ++i)
        for (j = 0; j < 8; ++j) {
            out_data[i * 8 + j] = in_data[j * 8 + i];
        }
}

static void dct_1d(float *in_data, float *out_data) {
    int i, j;

    for (j = 0; j < 8; ++j) {
        float dct = 0;

        for (i = 0; i < 8; ++i) {
            dct += in_data[i] * dctlookup[i][j];
        }

        out_data[j] = dct;
    }
}

static void idct_1d(float *in_data, float *out_data) {
    int i, j;

    for (j = 0; j < 8; ++j) {
        float idct = 0;

        for (i = 0; i < 8; ++i) {
            idct += in_data[i] * dctlookup[j][i];
        }

        out_data[j] = idct;
    }
}

static void scale_block(float *in_data, float *out_data) {
    int u, v;

    for (v = 0; v < 8; ++v) {
        for (u = 0; u < 8; ++u) {
            float a1 = !u ? ISQRT2 : 1.0f;
            float a2 = !v ? ISQRT2 : 1.0f;

            /* Scale according to normalizing function */
            out_data[v * 8 + u] = in_data[v * 8 + u] * a1 * a2;
        }
    }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl) {
    int zigzag;
    for (zigzag = 0; zigzag < 64; ++zigzag) {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[v * 8 + u];

        /* Zig-zag and quantize */
        out_data[zigzag] = round((dct / 4.0) / quant_tbl[zigzag]);
    }
}

static void dequantize_block(float *in_data, float *out_data, uint8_t *quant_tbl) {
    int zigzag;
    for (zigzag = 0; zigzag < 64; ++zigzag) {
        uint8_t u = zigzag_U[zigzag];
        uint8_t v = zigzag_V[zigzag];

        float dct = in_data[zigzag];

        /* Zig-zag and de-quantize */
        out_data[v * 8 + u] = round((dct * quant_tbl[zigzag]) / 4.0);
    }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl) {
    float mb[8 * 8] __attribute((aligned(16)));
    float mb2[8 * 8] __attribute((aligned(16)));

    int i, v;

    for (i = 0; i < 64; ++i)
        mb2[i] = in_data[i];

    for (v = 0; v < 8; ++v) {
        dct_1d(mb2 + v * 8, mb + v * 8);
    }

    transpose_block(mb, mb2);

    for (v = 0; v < 8; ++v) {
        dct_1d(mb2 + v * 8, mb + v * 8);
    }

    transpose_block(mb, mb2);
    scale_block(mb2, mb);
    quantize_block(mb, mb2, quant_tbl);

    for (i = 0; i < 64; ++i)
        out_data[i] = mb2[i];
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data, uint8_t *quant_tbl) {
    float mb[8 * 8] __attribute((aligned(16)));
    float mb2[8 * 8] __attribute((aligned(16)));

    int i, v;

    for (i = 0; i < 64; ++i)
        mb[i] = in_data[i];

    dequantize_block(mb, mb2, quant_tbl);

    scale_block(mb2, mb);

    for (v = 0; v < 8; ++v) {
        idct_1d(mb + v * 8, mb2 + v * 8);
    }

    transpose_block(mb2, mb);

    for (v = 0; v < 8; ++v) {
        idct_1d(mb + v * 8, mb2 + v * 8);
    }

    transpose_block(mb2, mb);

    for (i = 0; i < 64; ++i)
        out_data[i] = mb[i];
}

void catchCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

} /* end extern "C" */

///////////////////////////////////////////////////////////////////////////////
// TEXTURE LOADING ////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
texture<uint8_t, 2, cudaReadModeElementType> tex_ref;
texture<uint8_t, 2, cudaReadModeElementType> tex_orig;

void load_orig(uint8_t *host_ptr, uint8_t* dev_ptr, size_t width, size_t height, size_t pitch) {
    cudaMemcpy2D(dev_ptr, pitch, host_ptr, width, width, height, cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, &tex_orig, dev_ptr, &tex_orig.channelDesc, width, height, pitch);
}

void load_ref(uint8_t *host_ptr, uint8_t* dev_ptr, size_t width, size_t height, size_t pitch) {
    cudaMemcpy2D(dev_ptr, pitch, host_ptr, width, width, height, cudaMemcpyHostToDevice);
    cudaBindTexture2D(0, &tex_ref, dev_ptr, &tex_ref.channelDesc, width, height, pitch);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//meant for 512 threads
//threadDim(8,8,4);


#define LENGTH 40
#define MX (blockIdx.x * 8)
#define MY (blockIdx.y * 8)
#define RANGE 16

__shared__ uint32_t ref[LENGTH * LENGTH]; // <= (40) * (40)
__shared__ uint32_t orig[64];
__shared__ min_helper minimum[32 * 32];

__device__
inline void load_texture_values(int left, int top, int ref_index) 
{
    ref[ref_index] = tex2D(tex_ref, left + threadIdx.x, top + threadIdx.y);
    ref[16 * 40 + ref_index] = tex2D(tex_ref, left + threadIdx.x, top + 16 + threadIdx.y);

    if (threadIdx.y < 8) { //TODO Fix warp serialization
        //load vertically the blocks to the right
        ref[threadIdx.x * 40 + 32 + threadIdx.y] = tex2D(tex_ref, left + 32 + threadIdx.y, top + threadIdx.x);
    } else {
        //load the bottom row
        int y = threadIdx.y - 8;
        ref[(32 + y) * 40 + threadIdx.x] = tex2D(tex_ref, left + threadIdx.x, top + 32 + y);
    }
    if (threadIdx.y < 8 && threadIdx.x < 8) {
        ref[32 * 40 + 32 + threadIdx.y * 40 + threadIdx.x] = tex2D(tex_ref, left + 32 + threadIdx.x, top + 32 + threadIdx.y);
        orig[threadIdx.y * 8 + threadIdx.x] = tex2D(tex_orig, MX + threadIdx.x, MY + threadIdx.y);
    }
    __syncthreads();
}

#define COMPSAD(i,j); \
  minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value); \
  minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);

__device__
inline void calculate_usad(int res_index, int ref_index) {
    /*    
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            minimum[res_index].value = __usad(ref[ref_index + j * 40 + i], orig[j * 8 + i], minimum[res_index].value);
        }
    }
    
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            minimum[16 * 32 + res_index].value = __usad(ref[(16 * 40) + ref_index + j * 40 + i], orig[j * 8 + i], minimum[16 * 32 + res_index].value);
        }
    }
    */

    // Manual unrolling	
    COMPSAD(0,0); COMPSAD(0,1); COMPSAD(0,2); COMPSAD(0,3); COMPSAD(0,4); COMPSAD(0,5); COMPSAD(0,6); COMPSAD(0,7);
    COMPSAD(1,0); COMPSAD(1,1); COMPSAD(1,2); COMPSAD(1,3); COMPSAD(1,4); COMPSAD(1,5); COMPSAD(1,6); COMPSAD(1,7);
    COMPSAD(2,0); COMPSAD(2,1); COMPSAD(2,2); COMPSAD(2,3); COMPSAD(2,4); COMPSAD(2,5); COMPSAD(2,6); COMPSAD(2,7);
    COMPSAD(3,0); COMPSAD(3,1); COMPSAD(3,2); COMPSAD(3,3); COMPSAD(3,4); COMPSAD(3,5); COMPSAD(3,6); COMPSAD(3,7);
    COMPSAD(4,0); COMPSAD(4,1); COMPSAD(4,2); COMPSAD(4,3); COMPSAD(4,4); COMPSAD(4,5); COMPSAD(4,6); COMPSAD(4,7);
    COMPSAD(5,0); COMPSAD(5,1); COMPSAD(5,2); COMPSAD(5,3); COMPSAD(5,4); COMPSAD(5,5); COMPSAD(5,6); COMPSAD(5,7);
    COMPSAD(6,0); COMPSAD(6,1); COMPSAD(6,2); COMPSAD(6,3); COMPSAD(6,4); COMPSAD(6,5); COMPSAD(6,6); COMPSAD(6,7);
    COMPSAD(7,0); COMPSAD(7,1); COMPSAD(7,2); COMPSAD(7,3); COMPSAD(7,4); COMPSAD(7,5); COMPSAD(7,6); COMPSAD(7,7);
    
    __syncthreads();
}
/*
__device__
inline min_helper min_swap(min_helper m1, min_helper m2) {
    if (m1.value < m2.value) {
        return m1;
    } else {
        return m2;
    }
}*/

__device__
inline void setup_min(int res_index)
{
    minimum[res_index].x = threadIdx.x;
    minimum[res_index].y = threadIdx.y;
    minimum[res_index].value = 0;
    minimum[32 * 16 + res_index].x = threadIdx.x;
    minimum[32 * 16 + res_index].y = 16 + threadIdx.y;
    minimum[32 * 16 + res_index].value = 0;

    __syncthreads();
}

#define MIN2(m1,m2) (m1.value) < (m2.value) ? (m1) : (m2);
#define COMPMINSYNC(idx) minimum[res_index] = MIN2(minimum[res_index], minimum[(idx)]); __syncthreads();
__device__
inline void reduce_min(int res_index)
{    
    minimum[res_index] = MIN2(minimum[res_index], minimum[16*32+res_index]); __syncthreads();

    if (threadIdx.y <  8) COMPMINSYNC(8 * 32 + res_index); // reduce to 1 block_row
    if (threadIdx.y <  4) COMPMINSYNC(4 * 32 + res_index); // reduce to 4 rows
    if (threadIdx.y <  2) COMPMINSYNC(2 * 32 + res_index); // reduce to 2 rows
    if (threadIdx.y == 0) COMPMINSYNC(1 * 32 + res_index); // reduce to 1 row
    
    if (threadIdx.y == 0 && threadIdx.x < 16) COMPMINSYNC(16 + res_index);  // reduce to 16 values
    if (threadIdx.y == 0 && threadIdx.x <  8) COMPMINSYNC(8  + res_index);  // reduce to 8 values
    if (threadIdx.y == 0 && threadIdx.x <  4) COMPMINSYNC(4  + res_index);  // reduce to 4 values
    if (threadIdx.y == 0 && threadIdx.x <  2) COMPMINSYNC(2  + res_index);  // reduce to 2 values
    if (threadIdx.y == 0 && threadIdx.x == 0) COMPMINSYNC(1);               // reduce to 1 value
}

__global__ 
void cuda_me_texture(int width, int height, macroblock * mb) 
{
    int left = MX - 16;
    int top = MY - 16;

    int right = MX + 16;
    int bottom = MY + 16;
    
    if (left < 0) 
        left = 0;
    
    if (top < 0)  
        top = 0;
    
    if (right > (width - 8)) // Increase search area towards the left if we're out of bounds
        left += (width-8) - right;
        
    if (bottom > (height - 8)) // Increase search area towards the top if we're out of bounds
        top += (height - 8) - bottom;
    
    int res_index = threadIdx.y * 32 + threadIdx.x;
    int ref_index = threadIdx.y * 40 + threadIdx.x;
    
    load_texture_values(left, top, ref_index);
    setup_min(res_index);
    calculate_usad(res_index, ref_index);
    reduce_min(res_index);

    if (threadIdx.x == 0 && threadIdx.y == 0) 
    {
        mb[blockIdx.y * gridDim.x + blockIdx.x].mv_x = minimum[0].x + (left - MX);
        mb[blockIdx.y * gridDim.x + blockIdx.x].mv_y = minimum[0].y + (top - MY);
        mb[blockIdx.y * gridDim.x + blockIdx.x].use_mv = 1;
    }
}

