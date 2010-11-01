#include <math.h>
#include "c63.h"
#include "cuda_util.hcu"

__shared__ uint32_t predicted_s[256];

/* Motion compensation for 8x8 block */
__global__
void mc_block_cuda(int mb_width, int width, int height,size_t pitch, uint8_t *predicted, macroblock* mbs) {

	macroblock mb = mbs[BY * mb_width + BX * DZ + TZ];
	//write to global
	int y = (DY * BY + TY);
	int x = BX * DX * DZ + 8 * TZ + TX;
	if (x < width && y < height)
		predicted[y * pitch + x] = tex2D(tex_recons, x + mb.mv_x, y + mb.mv_y);
}

extern "C" void motion_compensate_cuda(struct c63_common *cm, struct cuda_frame *cframe) {

	cudaBindTexture2D(0, &tex_recons, cframe->last_recons->Y, &tex_recons.channelDesc, cm->ypw, cm->yph, cframe->last_recons_pitch[0]);
	mc_block_cuda<<<cframe->mc_blockDim_Y, cframe->mc_threadDim>>>(cframe->mb_width_Y, cm->ypw,cm->yph, cframe->predicted_pitch[0], cframe->predicted->Y, cframe->mbs[0]);

	cudaBindTexture2D(0, &tex_recons, cframe->last_recons->U, &tex_recons.channelDesc, cm->upw, cm->uph, cframe->last_recons_pitch[1]);
	mc_block_cuda<<<cframe->mc_blockDim_UV, cframe->mc_threadDim>>>(cframe->mb_width_UV, cm->upw,cm->uph,cframe->predicted_pitch[1], cframe->predicted->U, cframe->mbs[1]);

	cudaBindTexture2D(0, &tex_recons, cframe->last_recons->V, &tex_recons.channelDesc, cm->vpw, cm->vph, cframe->last_recons_pitch[2]);
	mc_block_cuda<<<cframe->mc_blockDim_UV, cframe->mc_threadDim>>>(cframe->mb_width_UV, cm->vpw,cm->vph,cframe->predicted_pitch[2], cframe->predicted->V, cframe->mbs[2]);
}
