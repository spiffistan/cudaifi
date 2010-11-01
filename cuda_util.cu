#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include "c63.h"
#include "cuda_util.hcu"

void catchCudaError(const char *message) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
		exit(-1);
	}
}
struct cuda_frame *cframe;

extern "C" void cuda_init(c63_common *cm) {

	cframe = (cuda_frame*) malloc(sizeof(cuda_frame));
	cframe->image = (yuv_t*) malloc(sizeof(yuv_t));
	cframe->curr_recons = (yuv_t*) malloc(sizeof(yuv_t));
	cframe->last_recons = (yuv_t*) malloc(sizeof(yuv_t));
	cframe->predicted = (yuv_t*) malloc(sizeof(yuv_t));
	cframe->residuals = (dct_t*) malloc(sizeof(dct_t));

	cframe->mb_width_Y = cm->mb_cols;
	cframe->mb_height_Y = cm->mb_rows;
	cframe->mb_width_UV = cm->mb_cols / 2;
	cframe->mb_height_UV = cm->mb_rows / 2;

	cframe->dct_threadDim = dim3(8, 8, 4);
	cframe->dct_blockDim_Y = dim3(ceil(cm->mb_cols / cframe->dct_threadDim.z), cm->mb_rows);
	cframe->dct_blockDim_UV = dim3(ceil((cm->vpw / 8.0f) / cframe->dct_threadDim.z), cm->vph / 8);

	cframe->mc_threadDim = dim3(8, 8, 4);
	cframe->mc_blockDim_Y = dim3(ceil(cm->mb_cols / cframe->mc_threadDim.z), cm->mb_rows);
	cframe->mc_blockDim_UV = dim3(ceil((cm->vpw / 8.0f) / cframe->mc_threadDim.z), cm->vph / 8);

	cframe->me_threadDim = dim3(32, 16, 1);
	cframe->me_blockDim_Y = dim3(cframe->mb_width_Y, cframe->mb_height_Y);
	cframe->me_blockDim_UV = dim3(cframe->mb_width_UV, cframe->mb_height_UV);

	cudaMallocPitch(&cframe->image->Y, &cframe->image_pitch[0], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->image->U, &cframe->image_pitch[1], cm->upw, cm->uph);
	cudaMallocPitch(&cframe->image->V, &cframe->image_pitch[2], cm->vpw, cm->vph);

	cudaMallocPitch(&cframe->curr_recons->Y, &cframe->curr_recons_pitch[0], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->curr_recons->U, &cframe->curr_recons_pitch[1], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->curr_recons->V, &cframe->curr_recons_pitch[2], cm->ypw, cm->yph);

	cudaMallocPitch(&cframe->last_recons->Y, &cframe->last_recons_pitch[0], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->last_recons->U, &cframe->last_recons_pitch[1], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->last_recons->V, &cframe->last_recons_pitch[2], cm->ypw, cm->yph);

	cudaMallocPitch(&cframe->predicted->Y, &cframe->predicted_pitch[0], cm->ypw, cm->yph);
	cudaMallocPitch(&cframe->predicted->U, &cframe->predicted_pitch[1], cm->upw, cm->uph);
	cudaMallocPitch(&cframe->predicted->V, &cframe->predicted_pitch[2], cm->vpw, cm->vph);

	cudaMalloc(&cframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(dct_t));
	cudaMalloc(&cframe->residuals->Udct, cm->vpw * cm->vph * sizeof(dct_t));
	cudaMalloc(&cframe->residuals->Vdct, cm->upw * cm->uph * sizeof(dct_t));

	cudaMalloc(&cframe->mbs[0], cframe->mb_width_Y * cframe->mb_height_Y * sizeof(macroblock));
	cudaMalloc(&cframe->mbs[1], cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock));
	cudaMalloc(&cframe->mbs[2], cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock));

	cudaMalloc(&cframe->qtables[0], 64 * sizeof(uint8_t));
	cudaMalloc(&cframe->qtables[1], 64 * sizeof(uint8_t));
	cudaMalloc(&cframe->qtables[2], 64 * sizeof(uint8_t));

	cudaMemcpy(cframe->qtables[0], cm->quanttbl[0], 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cframe->qtables[1], cm->quanttbl[1], 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(cframe->qtables[2], cm->quanttbl[2], 64 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	catchCudaError("CUDA_INIT");

}

void cuda_new_frame(c63_common *cm) {

	yuv_t *tmp = cframe->last_recons;
	cframe->last_recons = cframe->curr_recons;
	cframe->curr_recons = tmp;

	size_t pitch[3];
	pitch[0] = cframe->last_recons_pitch[0];
	pitch[1] = cframe->last_recons_pitch[1];
	pitch[2] = cframe->last_recons_pitch[2];
	cframe->last_recons_pitch[0] = cframe->curr_recons_pitch[0];
	cframe->last_recons_pitch[1] = cframe->curr_recons_pitch[1];
	cframe->last_recons_pitch[2] = cframe->curr_recons_pitch[2];
	cframe->curr_recons_pitch[0] = pitch[0];
	cframe->curr_recons_pitch[1] = pitch[1];
	cframe->curr_recons_pitch[2] = pitch[2];

	cudaMemset(cframe->residuals->Ydct, 0, cm->yph * cm->ypw * sizeof(int16_t));
	cudaMemset(cframe->residuals->Udct, 0, cm->yph * cm->ypw * sizeof(int16_t));
	cudaMemset(cframe->residuals->Vdct, 0, cm->yph * cm->ypw * sizeof(int16_t));

	cudaMemset2D(cframe->curr_recons->Y, cframe->curr_recons_pitch[0], 0, cm->ypw, cm->yph);
	cudaMemset2D(cframe->curr_recons->U, cframe->curr_recons_pitch[1], 0, cm->ypw, cm->yph);
	cudaMemset2D(cframe->curr_recons->V, cframe->curr_recons_pitch[2], 0, cm->ypw, cm->yph);

	cudaMemset2D(cframe->predicted->Y, cframe->predicted_pitch[0], 0, cm->ypw, cm->yph);
	cudaMemset2D(cframe->predicted->U, cframe->predicted_pitch[1], 0, cm->upw, cm->uph);
	cudaMemset2D(cframe->predicted->V, cframe->predicted_pitch[2], 0, cm->vpw, cm->vph);

	cudaMemset(cframe->mbs[0], 0, cframe->mb_width_Y * cframe->mb_height_Y * sizeof(macroblock));
	cudaMemset(cframe->mbs[1], 0, cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock));
	cudaMemset(cframe->mbs[2], 0, cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock));

	cudaMemcpy2D(cframe->image->Y, cframe->image_pitch[0], cm->curframe->orig->Y, cm->ypw, cm->ypw, cm->yph, cudaMemcpyHostToDevice);
	cudaMemcpy2D(cframe->image->U, cframe->image_pitch[1], cm->curframe->orig->U, cm->upw, cm->upw, cm->uph, cudaMemcpyHostToDevice);
	cudaMemcpy2D(cframe->image->V, cframe->image_pitch[2], cm->curframe->orig->V, cm->vpw, cm->vpw, cm->vph, cudaMemcpyHostToDevice);

	catchCudaError("CUDA_NEW_FRAME");
}

void cuda_store_values(struct c63_common *cm) {
	cudaMemcpy(cm->curframe->mbs[0], cframe->mbs[0], cframe->mb_width_Y * cframe->mb_height_Y * sizeof(macroblock), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->mbs[1], cframe->mbs[1], cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->mbs[2], cframe->mbs[2], cframe->mb_width_UV * cframe->mb_height_UV * sizeof(macroblock), cudaMemcpyDeviceToHost);

	cudaMemcpy(cm->curframe->residuals->Ydct, cframe->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->residuals->Udct, cframe->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(cm->curframe->residuals->Vdct, cframe->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t), cudaMemcpyDeviceToHost);
	catchCudaError("CUDA_STORE_VALUES");

}
extern "C" void cuda_run(struct c63_common *cm) {
	cuda_new_frame(cm);

	if (!cm->curframe->keyframe) {
		/* Motion Estimation */
		c63_motion_estimate(cm, cframe);

		/* Motion Compensation */
		//motion_compensate_cuda(cm, cframe);

	}

	//cm->curframe->keyframe = 1;
	/* DCT and Quantization */
	dct_quantize_frame(cm, cframe);
	idct_dequantize_frame(cm, cframe);

	cuda_store_values(cm);
}

extern "C" void cuda_stop() {
	cudaFree(cframe->image->Y);
	cudaFree(cframe->image->U);
	cudaFree(cframe->image->V);
	cudaFree(cframe->curr_recons->Y);
	cudaFree(cframe->curr_recons->U);
	cudaFree(cframe->curr_recons->V);
	cudaFree(cframe->last_recons->Y);
	cudaFree(cframe->last_recons->U);
	cudaFree(cframe->last_recons->V);
	cudaFree(cframe->predicted->Y);
	cudaFree(cframe->predicted->U);
	cudaFree(cframe->predicted->V);
	cudaFree(cframe->residuals->Ydct);
	cudaFree(cframe->residuals->Udct);
	cudaFree(cframe->residuals->Vdct);

	free(cframe->image);
	free(cframe->predicted);
	free(cframe->curr_recons);
	free(cframe->last_recons);
	free(cframe->residuals);
	free(cframe);
}
