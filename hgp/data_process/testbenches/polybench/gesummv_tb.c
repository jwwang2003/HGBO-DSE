#include "gesummv.h"
#include <string.h>

int main(void) {
    static float A[64][64], B[64][64], x[64], y_out[64];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < 64; i++) {
        x[i] = (float)(i + 1) / 64.0f;
        for (j = 0; j < 64; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 3 + j * 2) % 13) / 13.0f;
        }
    }
    memset(y_out, 0, sizeof(y_out));
    gesummv(alpha, beta, A, B, x, y_out);
    return 0;
}
