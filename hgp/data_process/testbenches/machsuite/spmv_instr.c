/* Instrumented spmv (CRS) for switching activity extraction */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define NNZ 1666
#define N 494
#define TYPE double

static inline double __logd(int id, double v) {
    uint64_t u; memcpy(&u, &v, 8);
    printf("SA %d 64 %llu\n", id, (unsigned long long)u);
    return v;
}
static inline int __logi(int id, int v) { printf("SA %d 32 %u\n", id, (unsigned)v); return v; }

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        TYPE sum = __logd(0, 0.0);
        int start = __logi(1, rowDelimiters[i]);
        int end   = __logi(2, rowDelimiters[i + 1]);
        for (j = start; j < end; j++)
            sum = __logd(3, sum + __logd(4, val[j] * vec[cols[j]]));
        out[i] = __logd(5, sum);
    }
}

int main(void) {
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
    for (i = N; i < NNZ; i++) { val[i] = 0; cols[i] = 0; }
    rowDelimiters[N] = NNZ;
    memset(out, 0, sizeof(out));
    spmv(val, cols, rowDelimiters, vec, out);
    return 0;
}
