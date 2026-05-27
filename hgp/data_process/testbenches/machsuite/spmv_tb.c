#include "spmv.h"
#include <string.h>

int main(void) {
    /* Simple diagonal matrix: NNZ = N, one entry per row */
    static TYPE val[NNZ];
    static int32_t cols[NNZ];
    static int32_t rowDelimiters[N + 1];
    static TYPE vec[N], out[N];
    int i;

    for (i = 0; i < N; i++) {
        val[i]  = (TYPE)(i + 1) / (TYPE)N;
        cols[i] = i;
        vec[i]  = (TYPE)(N - i) / (TYPE)N;
        rowDelimiters[i] = i;
    }
    /* Fill remaining NNZ-N entries with zeros on last row */
    for (i = N; i < NNZ; i++) {
        val[i]  = 0;
        cols[i] = 0;
    }
    rowDelimiters[N] = NNZ;
    memset(out, 0, sizeof(out));
    spmv(val, cols, rowDelimiters, vec, out);
    return 0;
}
