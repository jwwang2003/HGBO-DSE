/* Instrumented stencil3d for switching activity extraction */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define height_size 32
#define col_size 32
#define row_size 16
#define TYPE int32_t
#define MAX 1000
#define MIN 1
#define SIZE (row_size * col_size * height_size)
#define INDX(_rs,_cs,_i,_j,_k) ((_i)+_rs*((_j)+_cs*(_k)))

static inline int32_t __logi(int id, int32_t v) { printf("SA %d 32 %u\n", id, (unsigned)v); return v; }

void stencil3d(TYPE C[2], TYPE orig[SIZE], TYPE sol[SIZE]) {
    int i, j, k;
    TYPE sum0, sum1, mul0, mul1;
    for (k = 1; k < height_size - 1; k++)
        for (j = 1; j < col_size - 1; j++)
            for (i = 1; i < row_size - 1; i++) {
                sum0 = __logi(0, orig[INDX(row_size,col_size,i,j,k)] * C[0]);
                sum1 = __logi(1, orig[INDX(row_size,col_size,i+1,j,k)] + orig[INDX(row_size,col_size,i-1,j,k)]
                               + orig[INDX(row_size,col_size,i,j+1,k)] + orig[INDX(row_size,col_size,i,j-1,k)]
                               + orig[INDX(row_size,col_size,i,j,k+1)] + orig[INDX(row_size,col_size,i,j,k-1)]);
                mul1 = __logi(2, sum1 * C[1]);
                sol[INDX(row_size,col_size,i,j,k)] = __logi(3, sum0 + mul1);
            }
}

int main(void) {
    static TYPE C[2], orig[SIZE], sol[SIZE];
    int i;
    C[0] = 2; C[1] = 1;
    for (i = 0; i < SIZE; i++) orig[i] = (TYPE)(i % MAX) + MIN;
    memset(sol, 0, sizeof(sol));
    stencil3d(C, orig, sol);
    return 0;
}
