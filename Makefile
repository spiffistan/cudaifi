# Makefile 
 
CUDA_INSTALL_PATH := /usr/local/cuda
CUDA_SDK_PATH     := /Developer/GPU\ Computing/C
INCLUDES          += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_PATH)/common/inc
LIBS              := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_PATH)/lib
CFLAGS            := 
NVCFLAGS          := -G -g
LDFLAGS           := -lm -lcuda 
NVCC              := nvcc  
CC                := /usr/bin/gcc-4.0 

all: c63enc c63dec

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o common.o me.o
	$(NVCC) $(NVCFLAGS) $^ -o $@ $(LIBS) 

c63dec: c63dec.o dsp.o tables.o io.o common.o me.o
	$(NVCC) $(NVCFLAGS) $^ -o $@ $(LIBS) 

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< 

%.o: %.cu 
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -c $<

clean:
	rm -f *.o c63enc c63dec
