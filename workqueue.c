#include <pthread.h>
#include <stdlib.h>

#include "c63.h"
queue_t* init_queue(void) {
	queue_t *fifo = (queue_t *) malloc(sizeof(queue_t));

	fifo->size = 0;
	fifo->head = 0;
	fifo->tail = 0;

	fifo->mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
	fifo->notFull = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));
	fifo->notEmpty = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));

	pthread_mutex_init(fifo->mutex, NULL);
	pthread_cond_init(fifo->notFull, NULL);
	pthread_cond_init(fifo->notEmpty, NULL);

	return fifo;
}

void destroy_queue(queue_t *fifo) {
	pthread_mutex_destroy(fifo->mutex);
	free(fifo->mutex);
	pthread_cond_destroy(fifo->notFull);
	free(fifo->notFull);
	pthread_cond_destroy(fifo->notEmpty);
	free(fifo->notEmpty);
	free(fifo);
}
void queue_push(queue_t *fifo, workitem_t *in) {

	pthread_mutex_lock(fifo->mutex);

	if (fifo->size >= MAX_FRAMES) { // WAIT WHILE FULL
		pthread_cond_wait(fifo->notFull, fifo->mutex);
	}
	node_t *new = malloc(sizeof(node_t));
	new->data = in;
	if (fifo->size != 0) {
		fifo->tail->next = new;
		fifo->tail = new;
	} else {
		fifo->tail = new;
		fifo->head = new;
		fifo->head->next = new;
	}
	fifo->size++;

	pthread_mutex_unlock(fifo->mutex);
	pthread_cond_signal(fifo->notEmpty);

}
workitem_t* queue_pop(queue_t *fifo) {

	pthread_mutex_lock(fifo->mutex);
	if (fifo->size == 0) {
		if (fifo->done > 0) {
			pthread_mutex_unlock(fifo->mutex);
			return 0;
		}
		pthread_cond_wait(fifo->notEmpty, fifo->mutex); // WAIT WHILE EMPTY

		if (fifo->done > 0) {
			pthread_mutex_unlock(fifo->mutex);
			return 0;
		}
	}
	node_t* this = fifo->head;
	fifo->head = fifo->head->next;
	workitem_t *out = this->data;

	free(this);
	fifo->size--;

	pthread_mutex_unlock(fifo->mutex);
	pthread_cond_signal(fifo->notFull);
	return out;
}

void queue_stop(queue_t *fifo) {
	fifo->done = 1;
	pthread_cond_broadcast(fifo->notEmpty);
}
