static void dct_quantize(uint8_t *in_data, uint32_t width, uint32_t height,
        int16_t *out_data, uint32_t padwidth,
        uint32_t padheight, uint8_t *quantization)
{
    int y,x,u,v,j,i;

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
                    float dct = 0;
                    for(j = 0; j < jj; ++j)
                        for(i = 0; i < ii; ++i)
                        {
                            float coeff = in_data[(y+j)*width+(x+i)] - 128.0f;
                            dct += coeff * (float) (cos((2*i+1)*u*PI/16.0f) * cos((2*j+1)*v*PI/16.0f));
                        }

                    float a1 = !u ? ISQRT2 : 1.0f;
                    float a2 = !v ? ISQRT2 : 1.0f;

                    /* Scale according to normalizing function */
                    dct *= a1*a2/4.0f;

                    /* Quantize */
                    out_data[(y+v)*width+(x+u)] = (int16_t)(floor(0.5f + dct / (float)(quantization[v*8+u])));
                }
            }
        }
    }
}
