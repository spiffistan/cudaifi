#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <assert.h>

#include "c63.h"

// XXX: Should be moved to a struct with FILE*

void put_byte(FILE *fp, int byte)
{
    int status = fputc(byte, fp);

    if (status == EOF) {
        fprintf(stderr, "Error writing byte\n");
        exit(EXIT_FAILURE);
    }
}

void put_bytes(FILE *fp, const void* data, unsigned int len)
{
    int n = fwrite(data, 1, len, fp);

    if(n != len) {
        fprintf(stderr, "Error writing bytes\n");
        exit(-1);
    }
}

uint8_t get_byte(FILE *fp)
{
    int status = fgetc(fp);

    if (status == EOF) {
        fprintf(stderr, "End of file.\n");
        exit(EXIT_FAILURE);
    }

    return status;
}

int read_bytes(FILE *fp, void *data, unsigned int sz)
{
    int status = fread(data, 1, sz, fp);

    if (status == EOF)
    {
        fprintf(stderr, "End of file.\n");
        exit(0);
    }
    else if (status != sz) {
        fprintf(stderr, "Error reading bytes\n");
        exit(EXIT_FAILURE);
    }

    return status;
}

/**
 * Adds a bit to the bitBuffer. A call to bit_flush() is needed
 * in order to write any remainding bits in the buffer before
 * writing using another function.
 */
void put_bits(struct entropy_ctx *c, uint16_t bits, uint8_t n)
{
    assert(n <= 24  && "Error writing bit");

    if(n == 0)
        return;

    c->bit_buffer <<= n;
    c->bit_buffer |= bits & ((1 << n) - 1);
    c->bit_buffer_width += n;

    while(c->bit_buffer_width >= 8) {
        uint8_t b = (uint8_t)(c->bit_buffer >> (c->bit_buffer_width - 8));
        put_byte(c->fp, b);
        if(b == 0xff)
            put_byte(c->fp, 0);
        c->bit_buffer_width -= 8;
    }
}

uint16_t get_bits(struct entropy_ctx *c, uint8_t n)
{
    uint16_t ret = 0;

    while(c->bit_buffer_width < n)
    {
        uint8_t b = get_byte(c->fp);
        if (b == 0xff)
            get_byte(c->fp); // Discard stuffed byte

        c->bit_buffer <<= 8;
        c->bit_buffer |= b;
        c->bit_buffer_width += 8;
    }

    ret = c->bit_buffer >> (c->bit_buffer_width - n);
    c->bit_buffer_width -= n;

    /* Clear grabbed bits */
    c->bit_buffer &= (1 << c->bit_buffer_width) - 1;

    return ret;
}


/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
void flush_bits(struct entropy_ctx *c)
{
    if(c->bit_buffer > 0) {
        uint8_t b = c->bit_buffer << (8 - c->bit_buffer_width);
        put_byte(c->fp, b);
        if(b == 0xff)
            put_byte(c->fp, 0);
    }

    c->bit_buffer = 0;
    c->bit_buffer_width = 0;
}
