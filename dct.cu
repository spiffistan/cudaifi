#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include "c63.h"
#include "cuda_util.hcu"

#define ISQRT2 0.70710678118654f

__constant__ uint8_t zigzag_U[64] = { 0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7,
		7, 6, 5, 4, 5, 6, 7, 7, 6, 7, };

__constant__ uint8_t zigzag_V[64] = { 0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3,
		4, 5, 6, 7, 7, 6, 5, 6, 7, 7, };

__constant__ float dctlookup[8][8] = { { 1.000000f, 0.980785f, 0.923880f, 0.831470f, 0.707107f, 0.555570f, 0.382683f, 0.195090f }, { 1.000000f, 0.831470f, 0.382683f, -0.195090f, -0.707107f,
		-0.980785f, -0.923880f, -0.555570f }, { 1.000000f, 0.555570f, -0.382683f, -0.980785f, -0.707107f, 0.195090f, 0.923880f, 0.831470f }, { 1.000000f, 0.195090f, -0.923880f, -0.555570f, 0.707107f,
		0.831470f, -0.382683f, -0.980785f }, { 1.000000f, -0.195090f, -0.923880f, 0.555570f, 0.707107f, -0.831470f, -0.382683f, 0.980785f }, { 1.000000f, -0.555570f, -0.382683f, 0.980785f,
		-0.707107f, -0.195090f, 0.923880f, -0.831470f }, { 1.000000f, -0.831470f, 0.382683f, 0.195090f, -0.707107f, 0.980785f, -0.923880f, 0.555570f }, { 1.000000f, -0.980785f, 0.923880f, -0.831470f,
		0.707107f, -0.555570f, 0.382683f, -0.195090f } };

#define SMEM_SIZE 256
__shared__ float dct[SMEM_SIZE];
__shared__ float block[SMEM_SIZE];
__shared__ float quanttable[64];

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS DEVICE ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// DCT ////////////////////////////////////////////////////////////////////////

__device__
void dct1d(int block_start) {
	int i;
	float dct_ = 0;
	for (i = 0; i < 8; ++i) {
		dct_ += block[block_start + TY * 8 + i] * dctlookup[i][TX];
	}
	dct[block_start + TY * 8 + TX] = dct_;
	__syncthreads();
}

__device__
void idct1d(int block_start) {
	int i;
	float idct = 0;
	for (i = 0; i < 8; ++i) {
		idct += block[block_start + TY * 8 + i] * dctlookup[TX][i];
	}

	dct[block_start + TY * 8 + TX] = idct;
	__syncthreads();
}

// TRANSPOSE //////////////////////////////////////////////////////////////////

__device__
void transpose(int tid, int block_start) {
	block[block_start + tid] = dct[block_start + TX * 8 + TY];
	__syncthreads();
}

// SCALE //////////////////////////////////////////////////////////////////////

__device__
void scale(int global_tid) {
	float a1 = !TY ? ISQRT2 : 1.0f;
	float a2 = !TX ? ISQRT2 : 1.0f;

	/* Scale according to normalizing function */
	dct[global_tid] = block[global_tid] * a1 * a2;
	__syncthreads();
}

__device__
void unscale(int global_tid) {
	float a1 = !TY ? ISQRT2 : 1.0f;
	float a2 = !TX ? ISQRT2 : 1.0f;

	/* Unscale according to normalizing function */
	block[global_tid] = dct[global_tid] * a1 * a2;
	__syncthreads();
}

// QUANTIZE ///////////////////////////////////////////////////////////////////

__device__
void quantize(int tid, int block_start) {
	uint8_t u = zigzag_U[tid];
	uint8_t v = zigzag_V[tid];

	float dct_ = dct[block_start + v * 8 + u];

	/* Zig-zag and quantize */
	block[block_start + tid] = roundf((dct_ / 4.0) / quanttable[tid]);
	__syncthreads();
}

__device__
void dequantize(int tid, int block_start) {
	uint8_t u = zigzag_U[tid];
	uint8_t v = zigzag_V[tid];

	float dct_ = block[block_start + tid];

	/* Zig-zag and dequantize */
	dct[block_start + v * 8 + u] = roundf((dct_ * quanttable[tid]) / 4.0);
	__syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS GLOBAL ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define TEX_SELECT_X (DX * BX * DZ + (TZ * 8) + TX)
#define TEX_SELECT_Y (DY * BY + TY)
#define TEX_FETCH ((DY * BY * width) + (DX * DY * DZ * BX) + global_tid)

__global__
void dct_quantize_cuda(int width, int height, int16_t *out_data, uint8_t *qtable) {
	int block_start = TZ * 64;
	int tid = TY * 8 + TX;
	int global_tid = block_start + tid;

	// Fetch from texture memory into shared memory
	block[global_tid] = __int2float_rz(tex2D(tex_orig, TEX_SELECT_X, TEX_SELECT_Y) - tex1Dfetch(tex_pred, TEX_FETCH));

	if (TZ == 0) quanttable[tid] = __int2float_rz(qtable[tid]);

	__syncthreads();

	// Execute quant sequence
	dct1d(block_start);
	transpose(tid, block_start);
	dct1d(block_start);
	transpose(tid, block_start);
	scale(global_tid);
	quantize(tid, block_start);

	// Write from shared memory to device memory
	if (TEX_SELECT_X < width) out_data[DY * BY * width + DX * DY * DZ * BX + global_tid] = (int16_t) block[global_tid];
}

__global__
void idct_dequantize_cuda(int width, int height, size_t pitch, uint8_t *out_data, uint8_t *qtable) {
	int block_start = TZ * 64;
	int tid = TY * 8 + TX;
	int global_tid = block_start + tid;

	// Fetch from texture memory into shared memory
	block[global_tid] = tex1Dfetch(tex_residual, DY * BY * width + BX * DX * DY * DZ + global_tid);

	if (TZ == 0) quanttable[tid] = qtable[tid];

	__syncthreads();

	// Execute dequant sequence
	dequantize(tid, block_start);
	unscale(global_tid);
	idct1d(block_start);
	transpose(tid, block_start);
	idct1d(block_start);
	transpose(tid, block_start);

	//add the prediction
	int16_t tmp = __int2float_rn(block[global_tid]) + (int16_t) tex1Dfetch(tex_pred, TEX_FETCH);
	//Clamp values
	if (tmp < 0) tmp = 0;
	else if (tmp > 255) tmp = 255.0;

	// Write from shared memory to device memory
	if (TEX_SELECT_X < width) out_data[TEX_SELECT_Y * pitch + TEX_SELECT_X] = tmp;
}

extern "C" void dct_quantize_frame(c63_common *cm, struct cuda_frame *cframe) {
	cudaBindTexture2D(0, &tex_orig, cframe->image->Y, &tex_orig.channelDesc, cm->ypw, cm->yph, cframe->image_pitch[0]);
	cudaBindTexture(0, &tex_pred, cframe->predicted->Y, &tex_pred.channelDesc, cm->ypw * cm->yph);
	dct_quantize_cuda<<<cframe->dct_blockDim_Y, cframe->dct_threadDim>>>(cm->ypw,cm->yph,cframe->residuals->Ydct,cframe->qtables[0]);

	cudaBindTexture2D(0, &tex_orig, cframe->image->U, &tex_orig.channelDesc, cm->upw, cm->uph, cframe->image_pitch[1]);
	cudaBindTexture(0, &tex_pred, cframe->predicted->U, &tex_pred.channelDesc, cm->upw * cm->uph);
	dct_quantize_cuda<<<cframe->dct_blockDim_UV, cframe->dct_threadDim>>>(cm->upw,cm->uph,cframe->residuals->Udct,cframe->qtables[1]);

	cudaBindTexture2D(0, &tex_orig, cframe->image->V, &tex_orig.channelDesc, cm->vpw, cm->vph, cframe->image_pitch[2]);
	cudaBindTexture(0, &tex_pred, cframe->predicted->V, &tex_pred.channelDesc, cm->vpw * cm->vph);
dct_quantize_cuda<<<cframe->dct_blockDim_UV, cframe->dct_threadDim>>>(cm->vpw,cm->vph,cframe->residuals->Vdct,cframe->qtables[2]);
}

extern "C" void idct_dequantize_frame(c63_common *cm, struct cuda_frame *cframe) {
	cudaBindTexture(0, &tex_residual, cframe->residuals->Ydct, &tex_residual.channelDesc, cm->ypw * cm->yph * sizeof(int16_t));
	cudaBindTexture(0, &tex_pred, cframe->predicted->Y, &tex_pred.channelDesc, cm->ypw * cm->yph);
	idct_dequantize_cuda<<<cframe->dct_blockDim_Y, cframe->dct_threadDim>>>(cm->ypw,cm->yph,cframe->curr_recons_pitch[0],cframe->curr_recons->Y,cframe->qtables[0]);

	cudaBindTexture(0, &tex_residual, cframe->residuals->Udct, &tex_residual.channelDesc, cm->upw * cm->uph * sizeof(int16_t));
	cudaBindTexture(0, &tex_pred, cframe->predicted->U, &tex_pred.channelDesc, cm->upw * cm->uph);
	idct_dequantize_cuda<<<cframe->dct_blockDim_UV, cframe->dct_threadDim>>>(cm->upw,cm->uph,cframe->curr_recons_pitch[1],cframe->curr_recons->U,cframe->qtables[1]);

	cudaBindTexture(0, &tex_residual, cframe->residuals->Vdct, &tex_residual.channelDesc, cm->vpw * cm->vph * sizeof(int16_t));
	cudaBindTexture(0, &tex_pred, cframe->predicted->V, &tex_pred.channelDesc, cm->vpw * cm->vph);
idct_dequantize_cuda<<<cframe->dct_blockDim_UV, cframe->dct_threadDim>>>(cm->vpw,cm->vph,cframe->curr_recons_pitch[2],cframe->curr_recons->V,cframe->qtables[2]);
}
