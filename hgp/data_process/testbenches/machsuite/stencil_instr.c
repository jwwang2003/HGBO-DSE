/* Instrumented stencil2d for switching activity extraction */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define col_size 64
#define row_size 128
#define f_size 9
#define TYPE int32_t
#define MAX 1000
#define MIN 1
#define MAX_ITERATION 1

static inline int32_t __logi(int id, int32_t v) { printf("SA %d 32 %u\n", id, (unsigned)v); return v; }

void stencil(TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]) {
    int i, j, k;
    for (i = 1; i < row_size - 1; i++)
        for (j = 1; j < col_size - 1; j++) {
            int32_t sum = __logi(0, 0);
            for (k = 0; k < f_size; k++) {
                int row_off = k / 3 - 1;
                int col_off = k % 3 - 1;
                sum = __logi(1, sum + __logi(2, filter[k] * orig[(i + row_off) * col_size + (j + col_off)]));
            }
            sol[i * col_size + j] = __logi(3, sum);
        }
}

int main(void) {
    static TYPE orig[row_size * col_size], sol[row_size * col_size], filter[f_size];
    int i;
    for (i = 0; i < row_size * col_size; i++) orig[i] = (TYPE)(i % MAX) + MIN;
    memset(sol, 0, sizeof(sol));
    for (i = 0; i < f_size; i++) filter[i] = (TYPE)(i + 1);
    stencil(orig, sol, filter);
    return 0;
}
