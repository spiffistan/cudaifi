#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
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
struct r_args {
	uint32_t max_frames;
	struct c63_common *cm;
	FILE *infile;
	queue_t *fifo;
} r_args;

struct w_args {
	queue_t *output;
	struct c63_common *cm;
} w_args;

struct c_args {
	queue_t *input;
	queue_t *output;
	struct c63_common *cm;
} c_args;

// QUEUE //////////////////////////////////////////////////////////////////////


#define CATCH(p, str) if(p == NULL) { perror(str); exit(EXIT_FAILURE); }

// READ FILE //////////////////////////////////////////////////////////////////

/* Read YUV frames */
static yuv_t* read_yuv(FILE *file) {
	size_t len = 0;
	yuv_t *image = malloc(sizeof(yuv_t));
	/* Read Y' */
	image->Y = malloc(width * height);
	len += fread(image->Y, 1, width * height, file);
	if (ferror(file)) {
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	/* Read U */
	image->U = malloc(width * height);
	len += fread(image->U, 1, (width * height) / 4, file);
	if (ferror(file)) {
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	/* Read V */
	image->V = malloc(width * height);
	len += fread(image->V, 1, (width * height) / 4, file);
	if (ferror(file)) {
		perror("ferror");
		exit(EXIT_FAILURE);
	}

	if (len != width * height * 1.5) {
		fprintf(stderr, "Reached end of file.\n");
		return NULL;
	}

	return image;
}

// ENCODE /////////////////////////////////////////////////////////////////////

//static void c63_encode_image(struct c63_common *cm, yuv_t *image) {
//	/* Advance to next frame */
//	destroy_frame(cm->refframe);
//	cm->refframe = cm->curframe;
//	cm->curframe = create_frame(cm, image);
//
//	/* Check if keyframe */
//	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
//		cm->curframe->keyframe = 1;
//		cm->frames_since_keyframe = 0;
//
//		fprintf(stderr, " (keyframe) ");
//	} else
//		cm->curframe->keyframe = 0;
//
//	cuda_run(cm);
//
//	write_frame(cm);
//
//	++cm->framenum;
//	++cm->frames_since_keyframe;
//}

struct c63_common* init_c63_enc(int width, int height) {
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
	for (i = 0; i < 64; ++i) {
		cm->quanttbl[0][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[1][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[2][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
	}

	return cm;
}

static void print_help() {
	fprintf(stderr, "Usage: ./c63enc [options] input_file\n");
	fprintf(stderr, "Commandline options:\n");
	fprintf(stderr, "  -h                             height of images to compress\n");
	fprintf(stderr, "  -w                             width of images to compress\n");
	fprintf(stderr, "  -o                             Output file (.mjpg)\n");
	fprintf(stderr, "  [-f]                           Limit number of frames to encode\n");
	fprintf(stderr, "\n");

	exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
	int c;
	yuv_t *image;

	if (argc == 1) {
		print_help();
	}

	while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1) {
		switch (c) {
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
		default:
			print_help();
			break;
		}
	}

	if (optind >= argc) {
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit(EXIT_FAILURE);
	}

	outfile = fopen(output_file, "wb");
	if (outfile == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	struct c63_common *cm = init_c63_enc(width, height);
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

	if (infile == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	// WIP ////////////////////////////////////////////////////////////////////

	queue_t *input_queue = init_queue();
	queue_t *output_queue = init_queue();
	input_queue->done = 0;
	output_queue->done = 0;
	pthread_t reader, writer, cuda_thread;

	struct w_args *writer_args = malloc(sizeof(struct w_args));
	struct r_args *reader_args = malloc(sizeof(struct r_args));
	struct c_args *cuda_args = malloc(sizeof(struct c_args));

	reader_args->max_frames = limit_numframes;
	reader_args->fifo = input_queue;
	reader_args->infile = infile;
	reader_args->cm = cm;

	writer_args->output = output_queue;
	writer_args->cm = cm;

	cuda_args->input = input_queue;
	cuda_args->output = output_queue;
	cuda_args->cm = cm;

	pthread_create(&reader, NULL, reader_thread, reader_args);
	pthread_create(&cuda_thread, NULL, encoder_thread, cuda_args);
	pthread_create(&writer, NULL, writer_thread, writer_args);

	pthread_join(reader, NULL);
	pthread_join(cuda_thread, NULL);
	pthread_join(writer, NULL);


	destroy_queue(input_queue);
	free(writer_args);
	free(reader_args);

	fclose(outfile);
	fclose(infile);

	return EXIT_SUCCESS;
}

// PTHREADS ///////////////////////////////////////////////////////////////////

void *reader_thread(void *a) {
	struct r_args *args = (struct r_args *) a;
	int framecounter = 0;
	while (!feof(args->infile) && framecounter < args->max_frames) {

		workitem_t *w = malloc(sizeof(workitem_t));
		w->image = read_yuv(args->infile);
		if (!w->image)
			break;
		w->residuals = malloc(sizeof(dct_t));
		w->residuals->Ydct = malloc(args->cm->ypw * args->cm->yph * sizeof(int16_t));
		w->residuals->Udct = malloc(args->cm->upw * args->cm->uph * sizeof(int16_t));
		w->residuals->Vdct = malloc(args->cm->vpw * args->cm->vph * sizeof(int16_t));
		w->mbs[0] = malloc(args->cm->mb_cols * args->cm->mb_cols * sizeof(struct macroblock));
		w->mbs[1] = malloc(args->cm->mb_cols * args->cm->mb_cols * sizeof(struct macroblock));
		w->mbs[2] = malloc(args->cm->mb_cols * args->cm->mb_cols * sizeof(struct macroblock));

		w->framenum = framecounter++;
		queue_push(args->fifo, w);
	}
	queue_stop(args->fifo);
	return (NULL);
}

void *writer_thread(void *a) {
	struct w_args *args = (struct w_args *) a;

	while (1) {

		workitem_t *w = queue_pop(args->output);
		if(!w) {
			break;
		}
		write_frame(args->cm, w);
		++args->cm->framenum;
		++args->cm->frames_since_keyframe;

		free(w->image->Y);
		free(w->image->U);
		free(w->image->V);
		free(w->image);
		free(w->residuals->Ydct);
		free(w->residuals->Udct);
		free(w->residuals->Vdct);
		free(w->residuals);
		free(w->mbs[0]);
		free(w->mbs[1]);
		free(w->mbs[2]);
		free(w);
	}
	return (NULL);
}

void *encoder_thread(void *cuda_args) {
	/* Advance to next frame */
	struct c_args *args = (struct c_args*) cuda_args;
	cuda_init(args->cm);

	while (1) {
		/* Check if keyframe */
		workitem_t *w = queue_pop(args->input);
		if(!w)
			break;
		fprintf(stderr, "Encoding frame %d, ", w->framenum);

		if (w->framenum % args->cm->keyframe_interval == 0) {
			w->keyframe = 1;
			fprintf(stderr, " (keyframe) ");
		} else
			w->keyframe = 0;

		cuda_run(args->cm, w);
		fprintf(stderr, "Done!\n");

		queue_push(args->output, w);
	}

	queue_stop(args->output);
	cuda_stop();
	return (NULL);
}
