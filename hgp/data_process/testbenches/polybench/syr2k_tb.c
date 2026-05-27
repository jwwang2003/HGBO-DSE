#include "syr2k.h"
#include <string.h>

int main(void) {
    static float A[64][64], B[64][64], C[64][64], D_out[64][64];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < 64; i++)
        for (j = 0; j < 64; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C[i][j] = (float)(i == j ? 1 : 0);
            D_out[i][j] = 0.0f;
        }
    syr2k(alpha, beta, A, B, C, D_out);
    return 0;
}
