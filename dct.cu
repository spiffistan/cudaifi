#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include "c63.h"

#define TX threadIdx.x
#define TY threadIdx.y
#define BX blockIdx.x
#define BY blockIdx.y
#define DX blockDim.x
#define DY blockDim.y

#define ISQRT2 0.70710678118654f
__constant__ uint8_t zigzag_U[64] =
{
    0,
    1, 0,
    0, 1, 2,
    3, 2, 1, 0,
    0, 1, 2, 3, 4,
    5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6,
    7, 6, 5, 4, 3, 2, 1, 0,
    1, 2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2,
    3, 4, 5, 6, 7,
    7, 6, 5, 4,
    5, 6, 7,
    7, 6,
    7,
};

__constant__ uint8_t zigzag_V[64] =
{
    0,
    0, 1,
    2, 1, 0,
    0, 1, 2, 3,
    4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5,
    6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2, 1,
    2, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3,
    4, 5, 6, 7,
    7, 6, 5,
    6, 7,
    7,
};

texture<uint8_t, 2, cudaReadModeElementType> tex_orig;
texture<uint8_t, 2, cudaReadModeElementType> tex_pred;

__constant__ float dctlookup_x[64] = { 1.000000f, 0.980785f, 0.923880f, 0.831470f, 0.707107f, 0.555570f, 0.382683f, 0.195090f, 1.000000f, 0.831470f, 0.382683f, -0.195090f, -0.707107f, -0.980785f,
		-0.923880f, -0.555570f, 1.000000f, 0.555570f, -0.382683f, -0.980785f, -0.707107f, 0.195090f, 0.923880f, 0.831470f, 1.000000f, 0.195090f, -0.923880f, -0.555570f, 0.707107f, 0.831470f,
		-0.382683f, -0.980785f, 1.000000f, -0.195090f, -0.923880f, 0.555570f, 0.707107f, -0.831470f, -0.382683f, 0.980785f, 1.000000f, -0.555570f, -0.382683f, 0.980785f, -0.707107f, -0.195090f,
		0.923880f, -0.831470f, 1.000000f, -0.831470f, 0.382683f, 0.195090f, -0.707107f, 0.980785f, -0.923880f, 0.555570f, 1.000000f, -0.980785f, 0.923880f, -0.831470f, 0.707107f, -0.555570f,
		0.382683f, -0.195090f };

__constant__ float dctlookup_y[64] = { 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.980785, 0.831470, 0.555570,
		0.195090 - 0.195090 - 0.555570 - 0.831470 - 0.980785, 0.923880, 0.382683 - 0.382683 - 0.923880 - 0.923880 - 0.382683, 0.382683, 0.923880, 0.831470 - 0.195090 - 0.980785 - 0.555570, 0.555570,
		0.980785, 0.195090 - 0.831470, 0.707107 - 0.707107 - 0.707107, 0.707107, 0.707107 - 0.707107 - 0.707107, 0.707107, 0.555570 - 0.980785, 0.195090, 0.831470 - 0.831470 - 0.195090, 0.980785
				- 0.555570, 0.382683 - 0.923880, 0.923880 - 0.382683 - 0.382683, 0.923880 - 0.923880, 0.382683, 0.195090 - 0.555570, 0.831470 - 0.980785, 0.980785 - 0.831470, 0.555570 - 0.195090, };

__constant__ uint32_t mask = 0xFFFFFFF8;

__shared__ float dct[256];

__global__
void dct_quantize_cuda(int width, int16_t *out_data, uint8_t *quanttable) {
	extern __shared__ float data_f[];
	extern __shared__ uint32_t data_i[];

	int tid = TY * DX + TX;
	int tidx = (TX & mask) ^ TX; //modulus 8;

	data_i[tid] = tex2D(tex_orig, DX * BX + TX, DY * BY + TY) - tex2D(tex_pred,DX * BX + TX, DY * BY + TY);
	data_f[tid] = __int2float_rn(data_i[tid]);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			dct[tid] += data_f[(i << 3) * j] * dctlookup_y[(TY<<3) + i] * dctlookup_x[(j << 3) + tidx];
		}
	}

	float a1 = (!TY) ? M_SQRT1_2 : 1.0f;
	float a2 = (!TX) ? M_SQRT1_2 : 1.0f;

	dct[tid] *= a1 * a2 / 4.0f;

	uint8_t u = zigzag_U[TY * 8 + tidx];
	uint8_t v = zigzag_V[TY * 8 + tidx];

	float dct_ = dct[v * 32 + u];

	/* Zig-zag and quantize */
	out_data[DY * BY * width + DX * BX + TY * BX + TX] = int16_t(round(dct_ / quanttable[TY * 8 + tidx]));
}
void catchCudaError(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

extern "C" void dct_quantize_frame(c63_common *cm) {

	uint8_t *image_orig_Y, *image_pred_Y;
	uint8_t *image_orig_U, *image_pred_U;
	uint8_t *image_orig_V, *image_pred_V;

	size_t pitch_orig_Y, pitch_pred_Y;
	size_t pitch_orig_U, pitch_pred_U;
	size_t pitch_orig_V, pitch_pred_V;

	uint8_t *quanttables[3];
	cudaMalloc(&quanttables[0], 64);
	cudaMalloc(&quanttables[1], 64);
	cudaMalloc(&quanttables[2], 64);
	cudaMemcpy(quanttables[0], cm->quanttbl[0], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[1], cm->quanttbl[1], 64, cudaMemcpyHostToDevice);
	cudaMemcpy(quanttables[2], cm->quanttbl[2], 64, cudaMemcpyHostToDevice);

	//MALLOCS
	cudaMallocPitch((void**) &image_orig_Y, &pitch_orig_Y, cm->width, cm->height);
	cudaMallocPitch((void**) &image_pred_Y, &pitch_pred_Y, cm->width, cm->height);
	cudaMallocPitch((void**) &image_orig_U, &pitch_orig_U, cm->width / 2, cm->height) / 2;
	cudaMallocPitch((void**) &image_pred_U, &pitch_pred_U, cm->width / 2, cm->height / 2);
	cudaMallocPitch((void**) &image_orig_V, &pitch_orig_V, cm->width / 2, cm->height / 2);
	cudaMallocPitch((void**) &image_pred_V, &pitch_pred_V, cm->width / 2, cm->height / 2);

	cudaMemcpy2D(image_orig_Y, pitch_orig_Y, cm->curframe->orig->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);
    cudaMemcpy2D(image_pred_Y, pitch_pred_Y, cm->curframe->predicted->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);

	cudaBindTexture2D(0, &tex_orig, image_orig_Y, &tex_orig.channelDesc, cm->ypw, cm->ypw, pitch_orig_Y);
    cudaBindTexture2D(0, &tex_pred, image_pred_Y, &tex_pred.channelDesc, cm->ypw, cm->ypw, pitch_pred_Y);

    int16_t *output;
    cudaMalloc((void**)&output, cm->width*cm->height*sizeof(*output));
    dim3 blocks_Y(cm->mb_cols/4, cm->mb_rows);
    dim3 blocks_UV(cm->mb_cols/8, cm->mb_rows/2);
    dim3 threads(32,8,1);

    dct_quantize_cuda<<<blocks_Y, threads, 256*sizeof(float)>>>(cm->width,output,quanttables[0]);
    cudaMemcpy(cm->curframe->residuals->Ydct, output,cm->width*cm->height*sizeof(*output), cudaMemcpyDeviceToHost );

	cudaMemcpy2D(image_orig_U, pitch_orig_U, cm->curframe->orig->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);
    cudaMemcpy2D(image_pred_U, pitch_pred_U, cm->curframe->predicted->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);

	cudaBindTexture2D(0, &tex_orig, image_orig_U, &tex_orig.channelDesc, cm->upw, cm->upw, pitch_orig_U);
    cudaBindTexture2D(0, &tex_pred, image_pred_U, &tex_pred.channelDesc, cm->upw, cm->upw, pitch_pred_U);


    dct_quantize_cuda<<<blocks_UV, threads, 256*sizeof(float)>>>(cm->width/2,output,quanttables[1]);
    cudaMemcpy(cm->curframe->residuals->Udct, output,cm->width*cm->height*sizeof(*output)/4, cudaMemcpyDeviceToHost );

	cudaMemcpy2D(image_orig_V, pitch_orig_V, cm->curframe->orig->U, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);
    cudaMemcpy2D(image_pred_V, pitch_pred_V, cm->curframe->predicted->U, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);

	cudaBindTexture2D(0, &tex_orig, image_orig_V, &tex_orig.channelDesc, cm->vpw, cm->vph, pitch_orig_V);
    cudaBindTexture2D(0, &tex_pred, image_pred_V, &tex_pred.channelDesc, cm->vpw, cm->vph, pitch_pred_V);

    dct_quantize_cuda<<<blocks_UV, threads, 256*sizeof(float)>>>(cm->width/2,output,quanttables[2]);
    cudaMemcpy(cm->curframe->residuals->Vdct, output, cm->width * cm->height * sizeof(*output) / 4, cudaMemcpyDeviceToHost );

    catchCudaError("TEST");
}
