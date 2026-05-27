#include "gemm.h"
#include <string.h>

int main(void) {
    static float A[64][64], B[64][64], C[64][64], D[64][64];
    float alpha = 1.5f, beta = 0.5f;
    int i, j;
    for (i = 0; i < 64; i++)
        for (j = 0; j < 64; j++) {
            A[i][j] = (float)((i * 64 + j) % 13) / 13.0f;
            B[i][j] = (float)((i + j * 3) % 17) / 17.0f;
            C[i][j] = (float)(i % 7) / 7.0f;
            D[i][j] = 0.0f;
        }
    gemm(alpha, beta, A, B, C, D);
    return 0;
}
