#include "atax.h"
#include <string.h>

int main(void) {
    static float A[64][64], x[64], y_out[64];
    int i, j;
    for (i = 0; i < 64; i++) {
        x[i] = (float)(i + 1) / 64.0f;
        for (j = 0; j < 64; j++)
            A[i][j] = (float)((i * 64 + j) % 64) / 64.0f + 0.1f;
    }
    memset(y_out, 0, sizeof(y_out));
    atax(A, x, y_out);
    return 0;
}
