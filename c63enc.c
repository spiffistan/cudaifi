#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <pthread.h>

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


// QUEUE //////////////////////////////////////////////////////////////////////

#define NFRAMES 3
typedef struct 
{
	yuv_t framebuf[NFRAMES];
	uint32_t head, tail;
	uint8_t full, empty;
	uint8_t done; 
	pthread_mutex_t *mutex;
	pthread_cond_t *notFull, *notEmpty;
} queue_t;

#define CATCH(p, str) if(p == NULL) { perror(str); exit(EXIT_FAILURE); }
queue_t *init_queue(void) 
{
	queue_t *fifo = (queue_t *) malloc(sizeof(queue_t)); CATCH(fifo, "malloc");
	
	fifo->empty = 1;
	fifo->full  = 0;
	fifo->head  = 0;
	fifo->tail  = 0;
	
	fifo->mutex = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t)); CATCH(fifo->mutex, "malloc");
	fifo->notFull = (pthread_cond_t *) malloc (sizeof (pthread_cond_t)); CATCH(fifo->notFull, "malloc");
	fifo->notEmpty = (pthread_cond_t *) malloc (sizeof (pthread_cond_t)); CATCH(fifo->notEmpty, "malloc");
	
	pthread_mutex_init(fifo->mutex, NULL);
	pthread_cond_init(fifo->notFull, NULL);
	pthread_cond_init(fifo->notEmpty, NULL);
	
	return(fifo);
}

void destroy_queue (queue_t *fifo)
{
	pthread_mutex_destroy(fifo->mutex);
	free(fifo->mutex);	
	pthread_cond_destroy(fifo->notFull);
	free(fifo->notFull);
	pthread_cond_destroy(fifo->notEmpty);
	free(fifo->notEmpty);
	free(fifo);
}

void queue_add(queue_t *fifo, yuv_t *in)
{
	int *p = memcpy(&(fifo->framebuf[fifo->tail]), in, sizeof(yuv_t)); CATCH(p, "memcpy");
	
	fifo->tail += sizeof(yuv_t);
	if(fifo->tail == NFRAMES*sizeof(yuv_t))
		fifo->tail = 0;
	if(fifo->tail == fifo->head)
		fifo->full = 1;
	fifo->empty = 0;
}

void queue_del(queue_t *fifo, yuv_t *out)
{
	int *p = memcpy(out, &(fifo->framebuf[fifo->head]), sizeof(yuv_t)); CATCH(p, "memcpy");
	
	fifo->head += NFRAMES*sizeof(yuv_t);
	if(fifo->head == NFRAMES*sizeof(yuv_t))
		fifo->head = 0;
	if(fifo->head == fifo->tail)
		fifo->empty = 1;
	fifo->full = 0;
}

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

static void c63_encode_image(struct c63_common *cm, yuv_t *image) {
	/* Advance to next frame */
	destroy_frame(cm->refframe);
	cm->refframe = cm->curframe;
	cm->curframe = create_frame(cm, image);

	/* Check if keyframe */
	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval) {
		cm->curframe->keyframe = 1;
		cm->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");
	} else cm->curframe->keyframe = 0;

	cuda_run(cm);

	write_frame(cm);

	++cm->framenum;
	++cm->frames_since_keyframe;
}

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

struct r_args 
{
	FILE *infile;
	queue_t *fifo;
} r_args;

struct w_args
{
	queue_t *fifo;
	struct c63_common *cm;
	int numframes;
	int limit_numframes;
} w_args;

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

	if (limit_numframes) fprintf(stderr, "Limited to %d frames.\n", limit_numframes);

	FILE *infile = fopen(input_file, "rb");

	if (infile == NULL) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}
	cuda_init(cm);

	/* Encode input frames */
	int numframes = 0;
	
	// WIP ////////////////////////////////////////////////////////////////////
		
	queue_t *fifo = init_queue();
	
	fifo->done = 0;
	
	pthread_t reader, writer;
		
	struct w_args *writer_args = malloc(sizeof(struct w_args));
	struct r_args *reader_args = malloc(sizeof(struct r_args));
	
	reader_args->fifo = fifo;
	reader_args->infile = infile;
	
	writer_args->fifo = fifo;
	writer_args->cm = cm;
	writer_args->numframes = numframes;
	writer_args->limit_numframes = limit_numframes;
	
	pthread_create(&reader, NULL, reader_thread, reader_args);
	pthread_create(&writer, NULL, writer_thread, writer_args);
	
	pthread_join(reader, NULL);
	pthread_join(writer, NULL);
	
	/*
	while (!feof(infile)) 
	{
		image = read_yuv(infile); if(!image) break;
		queue_add(image);
		

		c63_encode_image(cm, image);

		free(image->Y);
		free(image->U);
		free(image->V);
		free(image);

		fprintf(stderr, "Done!\n");

		++numframes;
		if (limit_numframes && numframes >= limit_numframes) break;
	}
	*/
	
	
	destroy_queue(fifo);
	free(writer_args);
	free(reader_args);
		
	cuda_stop();
	fclose(outfile);
	fclose(infile);
	
	return EXIT_SUCCESS;
}

// PTHREADS ///////////////////////////////////////////////////////////////////

void *reader_thread(void *a) 
{
	struct r_args *args = (struct r_args *) a;
	
	while (!feof(args->infile)) 
	{
		pthread_mutex_lock(args->fifo->mutex);
		
		while(args->fifo->full) // WAIT WHILE FULL
			pthread_cond_wait(args->fifo->notFull, args->fifo->mutex);
		
		yuv_t *image = read_yuv(args->infile); 
		
		if(!image) break;
		
		queue_add(args->fifo, image);
		
		pthread_mutex_unlock(args->fifo->mutex);
		pthread_cond_signal(args->fifo->notEmpty);
	}
	
	pthread_mutex_lock(args->fifo->mutex);
	args->fifo->done = 1;
	pthread_mutex_unlock(args->fifo->mutex);
	
	return(NULL);
}

void *writer_thread(void *a)
{
	struct w_args *args = (struct w_args *) a;
		
	while(!args->fifo->done && !args->fifo->empty) 
	{
		pthread_mutex_lock(args->fifo->mutex);
		
		while(args->fifo->empty) // WAIT WHILE EMPTY
			pthread_cond_wait(args->fifo->notEmpty, args->fifo->mutex);
		
		yuv_t *image;
		
		queue_del(args->fifo, image);
		
		fprintf(stderr, "Encoding frame %d, ", args->numframes);
		
		c63_encode_image(args->cm, image);
		
		fprintf(stderr, "Done!\n");		
    	
		free(image->Y);
		free(image->U);
		free(image->V);
		free(image);
		
		++args->numframes;
		if (args->limit_numframes && args->numframes >= args->limit_numframes) break;
		
		pthread_mutex_unlock(args->fifo->mutex);
		pthread_cond_signal (args->fifo->notFull);
	}	
	return(NULL);
}

