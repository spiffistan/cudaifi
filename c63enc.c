#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "c63.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;
static uint32_t yph;
static uint32_t ypw;
static uint32_t uph;
static uint32_t upw;
static uint32_t vph;
static uint32_t vpw;

/* getopt */
extern int optind;
extern char *optarg;

//thread args
struct r_args
{
	uint32_t max_frames;
	struct c63_common *cm;
	FILE *infile;
	queue_t *input;
	queue_t *available;
} r_args;

struct w_args
{
	queue_t *output;
	queue_t *available;
	struct c63_common *cm;
} w_args;

struct c_args
{
	queue_t *input;
	queue_t *output;
	struct c63_common *cm;
} c_args;

// QUEUE //////////////////////////////////////////////////////////////////////


#define CATCH(p, str) if(p == NULL) { perror(str); exit(EXIT_FAILURE); }

// READ FILE //////////////////////////////////////////////////////////////////

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file, yuv_t* image)
{
	size_t len = 0;

	/* Read Y' */
	len += fread(image->Y, 1, width * height, file);
	if (ferror(file))
	{
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	/* Read U */
	len += fread(image->U, 1, (width * height) / 4, file);
	if (ferror(file))
	{
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	/* Read V */
	len += fread(image->V, 1, (width * height) / 4, file);
	if (ferror(file))
	{
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	if (len != width * height * 1.5)
	{
		fprintf(stderr, "Reached end of file.\n");
		return NULL;
	}

	return image;
}

// ENCODE /////////////////////////////////////////////////////////////////////


struct c63_common* init_c63_enc(int width, int height)
{
	int i;
	struct c63_common *cm = calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;
	cm->padw[0] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
	cm->padh[0] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
	cm->padw[1] = cm->upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[1] = cm->uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[2] = cm->vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[2] = cm->vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

	cm->mb_cols = cm->ypw / 8;
	cm->mb_rows = cm->yph / 8;

	/* Quality parameters */
	cm->qp = 25; // Constant quantization factor. Range: [1..50]
	cm->me_search_range = 16; // Pixels in every direction
	cm->keyframe_interval = 100; // Distance between keyframes


	/* Initialize quantization tables */
	for (i = 0; i < 64; ++i)
	{
		cm->quanttbl[0][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[1][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[2][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	return cm;
}

static void print_help()
{
	fprintf(stderr, "Usage: ./c63enc [options] input_file\n");
	fprintf(stderr, "Commandline options:\n");
	fprintf(stderr, "  -N                             choose naive motion estimation\n");
	fprintf(stderr, "  -h                             height of images to compress\n");
	fprintf(stderr, "  -w                             width of images to compress\n");
	fprintf(stderr, "  -o                             Output file (.mjpg)\n");
	fprintf(stderr, "  [-f]                           Limit number of frames to encode\n");
	fprintf(stderr, "\n");

	exit(EXIT_FAILURE);
}

void reset_workitem(workitem_t *w, struct c63_common *cm)
{
	memset(w->image->Y, 0, cm->width * cm->height);
	memset(w->image->U, 0, cm->width * cm->height / 4);
	memset(w->image->V, 0, cm->width * cm->height / 4);
	memset(w->mbs[0], 0, cm->mb_cols * cm->mb_rows * sizeof(struct macroblock));
	memset(w->mbs[1], 0, cm->mb_cols * cm->mb_rows * sizeof(struct macroblock));
	memset(w->mbs[2], 0, cm->mb_cols * cm->mb_rows * sizeof(struct macroblock));
	memset(w->residuals->Ydct, 0, cm->ypw * cm->yph * sizeof(int16_t));
	memset(w->residuals->Udct, 0, cm->upw * cm->uph * sizeof(int16_t));
	memset(w->residuals->Vdct, 0, cm->vpw * cm->vph * sizeof(int16_t));

}

queue_t* init_workitems(struct c63_common *cm)
{
	queue_t* q = init_queue();
	int i;
	for (i = 0; i < MAX_FRAMES; i++)
	{
		workitem_t *w;
		cudaMallocHost((void**) &w, sizeof(workitem_t));

		cudaMallocHost((void**) &w->image, sizeof(yuv_t));
		cudaMallocHost((void**) &w->image->Y, cm->width * cm->height);
		cudaMallocHost((void**) &w->image->U, cm->width * cm->height / 4);
		cudaMallocHost((void**) &w->image->V, cm->width * cm->height / 4);

		cudaMallocHost((void**) &w->residuals, sizeof(dct_t));
		cudaMallocHost((void**) &w->residuals->Ydct, cm->ypw * cm->yph * sizeof(int16_t));
		cudaMallocHost((void**) &w->residuals->Udct, cm->upw * cm->uph * sizeof(int16_t));
		cudaMallocHost((void**) &w->residuals->Vdct, cm->vpw * cm->vph * sizeof(int16_t));

		cudaMallocHost((void**) &w->mbs[0], cm->mb_cols * cm->mb_cols * sizeof(struct macroblock));
		cudaMallocHost((void**) &w->mbs[1], cm->mb_cols * cm->mb_cols * sizeof(struct macroblock));
		cudaMallocHost((void**) &w->mbs[2], cm->mb_cols * cm->mb_cols * sizeof(struct macroblock));
		reset_workitem(w, cm);
		queue_push(q, w);
	}
	return q;
}

int main(int argc, char **argv)
{
	int c;
	yuv_t *image;
	uint8_t naive = 0;
	if (argc == 1)
	{
		print_help();
	}

	while ((c = getopt(argc, argv, "h:w:o:f:N")) != -1)
	{
		switch (c)
		{
		case 'h':
			height = atoi(optarg);
			break;
		case 'w':
			width = atoi(optarg);
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'f':
			limit_numframes = atoi(optarg);
			break;
		case 'N':
			naive = 1;
			break;
		default:
			print_help();
			break;
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit(EXIT_FAILURE);
	}

	outfile = fopen(output_file, "wb");
	if (outfile == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	struct c63_common *cm = init_c63_enc(width, height);

	cm->use_naive = naive;

	cm->e_ctx.fp = outfile;
	/* Calculate the padded width and height */
	ypw = (uint32_t)(ceil(width / 8.0f) * 8);
	yph = (uint32_t)(ceil(height / 8.0f) * 8);
	upw = (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
	uph = (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
	vpw = (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
	vph = (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

	input_file = argv[optind];

	if (limit_numframes)
		fprintf(stderr, "Limited to %d frames.\n", limit_numframes);

	FILE *infile = fopen(input_file, "rb");

	if (infile == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	cuda_init(cm);
	queue_t *available = init_workitems(cm);
	queue_t *input_queue = init_queue();
	queue_t *output_queue = init_queue();
	input_queue->done = 0;
	output_queue->done = 0;
	pthread_t reader, writer, cuda_thread;

	struct w_args *writer_args = malloc(sizeof(struct w_args));
	struct r_args *reader_args = malloc(sizeof(struct r_args));
	struct c_args *cuda_args = malloc(sizeof(struct c_args));

	reader_args->max_frames = limit_numframes;
	reader_args->available = available;
	reader_args->input = input_queue;
	reader_args->infile = infile;
	reader_args->cm = cm;

	writer_args->available = available;
	writer_args->output = output_queue;
	writer_args->cm = cm;

	pthread_create(&reader, NULL, reader_thread, reader_args);
	pthread_create(&writer, NULL, writer_thread, writer_args);

	encoder_thread(input_queue, output_queue, cm);

	pthread_join(reader, NULL);
	pthread_join(writer, NULL);

	cuda_stop();
	destroy_queue(input_queue);
	destroy_queue(output_queue);

	while (available->size)
	{
		workitem_t* w = queue_pop(available);
		cudaFree(w->image->Y);
		cudaFree(w->image->U);
		cudaFree(w->image->V);
		cudaFree(w->image);
		cudaFree(w->residuals->Ydct);
		cudaFree(w->residuals->Udct);
		cudaFree(w->residuals->Vdct);
		cudaFree(w->residuals);
		cudaFree(w->mbs[0]);
		cudaFree(w->mbs[1]);
		cudaFree(w->mbs[2]);
		cudaFree(w);
	}

	destroy_queue(available);
	free(writer_args);
	free(reader_args);

	fclose(outfile);
	fclose(infile);

	return EXIT_SUCCESS;
}

// PTHREADS ///////////////////////////////////////////////////////////////////

void *reader_thread(void *a)
{
	struct r_args *args = (struct r_args *) a;
	int framecounter = 0;
	while (!feof(args->infile) && (framecounter < args->max_frames || args->max_frames == 0))
	{

		workitem_t *w = queue_pop(args->available);
		read_yuv(args->infile, w->image);
		if (!w->image || feof(args->infile))
			break;

		w->framenum = framecounter++;
		queue_push(args->input, w);
	}
	queue_stop(args->input);
	return (NULL);
}

void *writer_thread(void *a)
{
	struct w_args *args = (struct w_args *) a;

	while (1)
	{
		workitem_t *w = queue_pop(args->output);

		if (!w)
		{
			break;
		}
		write_frame(args->cm, w);

		++args->cm->framenum;
		++args->cm->frames_since_keyframe;
		reset_workitem(w, args->cm);
		queue_push(args->available, w);
	}
	return (NULL);
}

void encoder_thread(queue_t* input, queue_t* output, struct c63_common *cm)
{
	while (1)
	{
		/* Check if keyframe */

		workitem_t *w = queue_pop(input);
		if (!w)
			break;
		fprintf(stderr, "Encoding frame %d, ", w->framenum);

		if (w->framenum % cm->keyframe_interval == 0)
		{
			w->keyframe = 1;
			fprintf(stderr, " (keyframe) ");
		} else
			w->keyframe = 0;
		cuda_run(cm, w);

		fprintf(stderr, "Done!\n");

		queue_push(output, w);
	}

	queue_stop(output);
}
