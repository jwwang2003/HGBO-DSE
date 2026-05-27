#include "mvt.h"
#include <string.h>

int main(void) {
    static float A[64][64], x1[64], x2[64], y1[64], y2[64], x1_out[64], x2_out[64];
    int i, j;
    for (i = 0; i < 64; i++) {
        x1[i] = (float)(i + 1) / 64.0f;
        x2[i] = (float)(64 - i) / 64.0f;
        y1[i] = (float)(i % 7) / 7.0f;
        y2[i] = (float)(i % 11) / 11.0f;
        for (j = 0; j < 64; j++)
            A[i][j] = (float)((i + j * 2) % 13) / 13.0f;
    }
    memset(x1_out, 0, sizeof(x1_out));
    memset(x2_out, 0, sizeof(x2_out));
    mvt(A, x1, x2, y1, y2, x1_out, x2_out);
    return 0;
}
