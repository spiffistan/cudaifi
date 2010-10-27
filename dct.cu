static void dct_quantize(uint8_t *in_data, uint32_t width, uint32_t height,
        int16_t *out_data, uint32_t padwidth,
        uint32_t padheight, uint8_t *quantization)
{	
    int y,x,u,v,j,i;

	__m128 coeff, cos4, cos1; 
	__m128i tmp128i;
	__m64 tmp64;
	const __m128 SUB128 = { 128.0f, 128.0f, 128.0f, 128.0f };
	
	float dctpart[4] __attribute((aligned(16)));
	
    /* Perform the DCT and quantization */
    for(y = 0; y < height; y += 8)
    {
        int jj = height - y;
        jj = MIN(jj, 8); // For the border-pixels, we might have a part of an 8x8 block

        for(x = 0; x < width; x += 8)
        {
            int ii = width - x;
            ii = MIN(ii, 8); // For the border-pixels, we might have a part of an 8x8 block

            //Loop through all elements of the block
            for(u = 0; u < 8; ++u)
            {	
                for(v = 0; v < 8; ++v)
                {
                    /* Compute the DCT */
					__m128 dct_;
					
					dct_ = _mm_setzero_ps();
					
                    for(j = 0; j < jj; ++j) {
	
						const int c = (y+j)*width+x;
						
						cos1 = _mm_load_ps1(&cos_table[(v*8)+j]);

                        for(i = 0; i < ii; i+=4)
                        { 	
							tmp128i = _mm_lddqu_si128((__m128i *) &in_data[(c+i)]); // SSE3
							tmp64 = _mm_movepi64_pi64(tmp128i);	
							coeff = _mm_cvtpu8_ps(tmp64);
							coeff = _mm_sub_ps(coeff, SUB128);
							cos4 = _mm_load_ps(&cos_table[(u*8)+i]);
							coeff = _mm_mul_ps(coeff, cos4);
							coeff = _mm_mul_ps(coeff, cos1);							                
							dct_ = _mm_add_ps(coeff, dct_);
                        }
					}
					
                    float a1 = !u ? ISQRT2 : 1.0f;
                    float a2 = !v ? ISQRT2 : 1.0f; 

					_mm_store_ps(dctpart, dct_);
					
					float dct = dctpart[0] + dctpart[1] + dctpart[2] + dctpart[3];

                    /* Scale according to normalizing function */
                    dct *= a1*a2/4.0f;

                    /* Quantize */
                    out_data[(y+v)*width+(x+u)] = (int16_t)(floor(0.5f + dct / (float)(quantization[v*8+u])));
                }
            }
        }
    }
}