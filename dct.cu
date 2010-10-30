#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include "c63.h"
#include "cuda_util.hcu"

#define TX threadIdx.x
#define TY threadIdx.y
#define TZ threadIdx.z

#define BX blockIdx.x
#define BY blockIdx.y

#define DX blockDim.x
#define DY blockDim.y
#define DZ blockDim.z

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

texture<uint8_t, 2, cudaReadModeElementType> tex_orig;
texture<uint8_t, 2, cudaReadModeElementType> tex_pred;
texture<int16_t, 1, cudaReadModeElementType> tex_residual;

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
	block[block_start + tid] = round((dct_ * 0.25) / quanttable[tid]);
	__syncthreads();
}

__device__
void dequantize(int tid, int block_start) {
	uint8_t u = zigzag_U[tid];
	uint8_t v = zigzag_V[tid];

	float dct_ = block[block_start + tid];

	/* Zig-zag and dequantize */
	dct[block_start + v * 8 + u] = round((dct_ * quanttable[tid]) * 0.25);
	__syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
// CUDA KERNELS GLOBAL ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#define TEX_SELECT_X (DX * BX * DZ + (TZ * 8) + TX)
#define TEX_SELECT_Y (DY * BY + TY)

__global__
void dct_quantize_cuda(int width, int16_t *out_data, uint8_t *qtable) {
	int block_start = TZ * 64;
	int tid = TY * 8 + TX;
	int global_tid = block_start + tid;

	// Fetch from texture memory into shared memory
	block[global_tid] = __int2float_rn(tex2D(tex_orig, TEX_SELECT_X, TEX_SELECT_Y) - tex2D(tex_pred, TEX_SELECT_X, TEX_SELECT_Y));

	if (TZ == 0)
		quanttable[tid] = __int2float_rn(qtable[tid]);

	__syncthreads();

	// Execute quant sequence
	dct1d(block_start);
	transpose(tid, block_start);
	dct1d(block_start);
	transpose(tid, block_start);
	scale(global_tid);
	quantize(tid, block_start);

	// Write from shared memory to device memory
	out_data[DY * BY * width + DX * DY * DZ * BX + global_tid] = __float2int_rn(block[global_tid]);
}

__global__
void idct_dequantize_cuda(int width, uint8_t *out_data, uint8_t *qtable) {
	int block_start = TZ * 64;
	int tid = TY * 8 + TX;
	int global_tid = block_start + tid;

	// Fetch from texture memory into shared memory
	block[global_tid] = __int2float_rn(tex1Dfetch(tex_residual, DY * BY * width + BX * DX * DY * DZ + global_tid));

	if (TZ == 0)
		quanttable[tid] = __int2float_rn(qtable[tid]);

	__syncthreads();

	// Execute dequant sequence
	dequantize(tid, block_start);
	unscale(global_tid);
	idct1d(block_start);
	transpose(tid, block_start);
	idct1d(block_start);
	transpose(tid, block_start);

	//add the prediction

	block[global_tid] += tex2D(tex_pred, TEX_SELECT_X, TEX_SELECT_Y);
	__syncthreads();
	//Clamp values
	if (block[global_tid] < 0.0)
		block[global_tid] = 0.0;
	else if (block[global_tid] > 255.0)
		block[global_tid] = 255.0;
	// Write from shared memory to device memory
	out_data[TEX_SELECT_Y * width + TEX_SELECT_X] = __float2int_rn(block[global_tid]);
}

extern "C" void dct_quantize_frame(c63_common *cm) {

	uint8_t *image_orig_Y, *image_pred_Y;
	uint8_t *image_orig_U, *image_pred_U;
	uint8_t *image_orig_V, *image_pred_V;

	size_t pitch_orig_Y, pitch_pred_Y;
	size_t pitch_orig_U, pitch_pred_U;
	size_t pitch_orig_V, pitch_pred_V;

	uint8_t *quanttables[3];
	int16_t *output;

	dim3 threads(8, 8, 4);
	dim3 blocks_Y(ceil(cm->mb_cols / threads.z), cm->mb_rows);
	dim3 blocks_UV(ceil((cm->vpw / 8.0f) / threads.z), ceil(cm->vph / 8));

	// MALLOC /////////////////////////////////////////////////////////////////

	cudaMalloc((void**) &output, cm->ypw * cm->yph * sizeof(*output));
	cudaMallocPitch((void**) &image_orig_Y, &pitch_orig_Y, cm->ypw, cm->yph);
	cudaMallocPitch((void**) &image_pred_Y, &pitch_pred_Y, cm->ypw, cm->yph);
	cudaMallocPitch((void**) &image_orig_U, &pitch_orig_U, cm->upw, cm->uph);
	cudaMallocPitch((void**) &image_pred_U, &pitch_pred_U, cm->upw, cm->uph);
	cudaMallocPitch((void**) &image_orig_V, &pitch_orig_V, cm->vpw, cm->vph);
	cudaMallocPitch((void**) &image_pred_V, &pitch_pred_V, cm->vpw, cm->vph);
	cudaMalloc(&quanttables[0], 64);
	cudaMalloc(&quanttables[1], 64);
	cudaMalloc(&quanttables[2], 64);

	cudaMemcpy(quanttables[0], cm->quanttbl[0], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[1], cm->quanttbl[1], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[2], cm->quanttbl[2], 64, cudaMemcpyHostToDevice);

	// Y //////////////////////////////////////////////////////////////////////

	cudaMemcpy2D(image_orig_Y, pitch_orig_Y, cm->curframe->orig->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_Y, pitch_pred_Y, cm->curframe->predicted->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);

	cudaBindTexture2D(0, &tex_orig, image_orig_Y, &tex_orig.channelDesc, cm->ypw, cm->yph, pitch_orig_Y);
	cudaBindTexture2D(0, &tex_pred, image_pred_Y, &tex_pred.channelDesc, cm->ypw, cm->yph, pitch_pred_Y);

	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(int16_t));
	dct_quantize_cuda<<<blocks_Y, threads>>>(cm->width,output,quanttables[0]);

	cudaMemcpy(cm->curframe->residuals->Ydct, output, cm->ypw * cm->yph * sizeof(int16_t), cudaMemcpyDeviceToHost);

	// U //////////////////////////////////////////////////////////////////////
	cudaMemcpy2D(image_orig_U, pitch_orig_U, cm->curframe->orig->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_U, pitch_pred_U, cm->curframe->predicted->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);


	cudaBindTexture2D(0, &tex_orig, image_orig_U, &tex_orig.channelDesc, cm->upw, cm->upw, pitch_orig_U);
	cudaBindTexture2D(0, &tex_pred, image_pred_U, &tex_pred.channelDesc, cm->upw, cm->upw, pitch_pred_U);
	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(int16_t));

	dct_quantize_cuda<<<blocks_UV, threads>>>(cm->upw,output,quanttables[1]);

	cudaMemcpy(cm->curframe->residuals->Udct, output, cm->upw * cm->uph * sizeof(int16_t), cudaMemcpyDeviceToHost);

	// V //////////////////////////////////////////////////////////////////////

	cudaMemcpy2D(image_orig_V, pitch_orig_V, cm->curframe->orig->V, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_V, pitch_pred_V, cm->curframe->predicted->V, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, &tex_orig, image_orig_V, &tex_orig.channelDesc, cm->vpw, cm->vph, pitch_orig_V);
	cudaBindTexture2D(0, &tex_pred, image_pred_V, &tex_pred.channelDesc, cm->vpw, cm->vph, pitch_pred_V);
	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(int16_t));

	dct_quantize_cuda<<<blocks_UV, threads>>>(cm->vpw,output,quanttables[2]);

	cudaMemcpy(cm->curframe->residuals->Vdct, output, cm->vpw * cm->vph * sizeof(int16_t), cudaMemcpyDeviceToHost);

	// FREE ///////////////////////////////////////////////////////////////////

	catchCudaError("ERROR IN DCTQUANTIZE");

	cudaFree(image_orig_Y);
	cudaFree(image_orig_U);
	cudaFree(image_orig_V);
	cudaFree(image_pred_Y);
	cudaFree(image_pred_U);
	cudaFree(image_pred_V);
	cudaFree(quanttables[0]);
	cudaFree(quanttables[1]);
	cudaFree(quanttables[2]);
	cudaFree(output);
}

extern "C" void idct_dequantize_frame(c63_common *cm) {

	uint16_t *residuals_Y;
	uint16_t *residuals_U;
	uint16_t *residuals_V;

	uint8_t *image_pred_Y;
	uint8_t *image_pred_U;
	uint8_t *image_pred_V;

	size_t pitch_pred_Y;
	size_t pitch_pred_U;
	size_t pitch_pred_V;

	uint8_t *quanttables[3];
	uint8_t *output;

	dim3 threads(8, 8, 1);
	dim3 blocks_Y(ceil(cm->mb_cols / threads.z), cm->mb_rows);
	dim3 blocks_UV(ceil((cm->vpw / 8.0f) / threads.z), cm->vph/8);

	// MALLOC /////////////////////////////////////////////////////////////////

	cudaMalloc((void**) &output, cm->ypw * cm->yph * sizeof(*output));
	cudaMalloc((void**) &residuals_Y, cm->ypw * cm->yph * sizeof(*residuals_Y));
	cudaMalloc((void**) &residuals_U, cm->vpw * cm->vph * sizeof(*residuals_V));
	cudaMalloc((void**) &residuals_V, cm->upw * cm->uph * sizeof(*residuals_U));

	catchCudaError("TESTER");


	cudaMallocPitch((void**) &image_pred_Y, &pitch_pred_Y, cm->ypw, cm->yph);
	cudaMallocPitch((void**) &image_pred_U, &pitch_pred_U, cm->upw, cm->uph);
	cudaMallocPitch((void**) &image_pred_V, &pitch_pred_V, cm->vpw, cm->vph);
	cudaMalloc(&quanttables[0], 64);
	cudaMalloc(&quanttables[1], 64);
	cudaMalloc(&quanttables[2], 64);

	cudaMemcpy(quanttables[0], cm->quanttbl[0], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[1], cm->quanttbl[1], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[2], cm->quanttbl[2], 64, cudaMemcpyHostToDevice);

	// Y //////////////////////////////////////////////////////////////////////


	cudaMemcpy(residuals_Y, cm->curframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(*residuals_Y), cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_Y, pitch_pred_Y, cm->curframe->predicted->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);

	cudaBindTexture(0, &tex_residual, residuals_Y, &tex_residual.channelDesc, cm->ypw * cm->yph * sizeof(*residuals_Y));
	cudaBindTexture2D(0, &tex_pred, image_pred_Y, &tex_pred.channelDesc, cm->ypw, cm->yph, pitch_pred_Y);

	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(*output));

	idct_dequantize_cuda<<<blocks_Y, threads>>>(cm->width,output,quanttables[0]);
	cudaMemcpy(cm->curframe->recons->Y, output, cm->ypw * cm->yph * sizeof(*output), cudaMemcpyDeviceToHost);

	// U //////////////////////////////////////////////////////////////////////

	cudaMemcpy(residuals_U, cm->curframe->residuals->Udct, cm->upw * cm->uph * sizeof(*residuals_U), cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_U, pitch_pred_U, cm->curframe->predicted->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);
	cudaBindTexture(0, &tex_residual, residuals_U, &tex_residual.channelDesc, cm->upw * cm->uph * sizeof(*residuals_U));
	cudaBindTexture2D(0, &tex_pred, image_pred_U, &tex_pred.channelDesc, cm->upw, cm->upw, pitch_pred_U);
	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(*output));

	idct_dequantize_cuda<<<blocks_UV, threads>>>(cm->upw,output,quanttables[1]);

	cudaMemcpy(cm->curframe->recons->U, output, cm->upw * cm->uph * sizeof(*output), cudaMemcpyDeviceToHost);

	// V //////////////////////////////////////////////////////////////////////

	cudaMemcpy(residuals_V, cm->curframe->residuals->Vdct, cm->vpw * cm->vph * sizeof(*residuals_Y), cudaMemcpyHostToDevice);
	cudaMemcpy2D(image_pred_V, pitch_pred_V, cm->curframe->predicted->V, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);

	cudaBindTexture(0, &tex_residual, residuals_V, &tex_residual.channelDesc, cm->vpw * cm->vph * sizeof(*residuals_V));
	cudaBindTexture2D(0, &tex_pred, image_pred_V, &tex_pred.channelDesc, cm->vpw, cm->vpw, pitch_pred_V);

	cudaMemset(output, 0, cm->yph * cm->ypw * sizeof(*output));

	idct_dequantize_cuda<<<blocks_UV, threads>>>(cm->vpw,output,quanttables[2]);

	cudaMemcpy(cm->curframe->recons->V, output, cm->vpw * cm->vph * sizeof(*output), cudaMemcpyDeviceToHost);
	catchCudaError("ERROR IN IDCTQUANTIZE");

	// FREE ///////////////////////////////////////////////////////////////////


	cudaFree(residuals_Y);
	cudaFree(residuals_U);
	cudaFree(residuals_V);
	cudaFree(image_pred_Y);
	cudaFree(image_pred_U);
	cudaFree(image_pred_V);
	cudaFree(quanttables[0]);
	cudaFree(quanttables[1]);
	cudaFree(quanttables[2]);
	cudaFree(output);
}
