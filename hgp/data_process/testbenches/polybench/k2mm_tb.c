#include "k2mm.h"
#include <string.h>

int main(void) {
    static float A[64][64], B[64][64], C[64][64], D[64][64], E_out[64][64];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < 64; i++)
        for (j = 0; j < 64; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C[i][j] = (float)((i + j * 3) % 7) / 7.0f;
            D[i][j] = (float)(i % 5) / 5.0f;
            E_out[i][j] = 0.0f;
        }
    k2mm(alpha, beta, A, B, C, D, E_out);
    return 0;
}
