#include "bicg.h"
#include <string.h>

int main(void) {
    static float A[64][64], p[64], r[64], s_out[64], q_out[64];
    int i, j;
    for (i = 0; i < 64; i++) {
        p[i] = (float)(i + 1) / 64.0f;
        r[i] = (float)(64 - i) / 64.0f;
        for (j = 0; j < 64; j++)
            A[i][j] = (float)((i + j + 1) % 64) / 64.0f;
    }
    memset(s_out, 0, sizeof(s_out));
    memset(q_out, 0, sizeof(q_out));
    bicg(A, p, r, s_out, q_out);
    return 0;
}
