/* Instrumented stencil3d kernel.
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["stencil3d"]):
 *   0: load orig
 *   1: add
 *   2: mul
 *   3: store sol
 */
#include "../sa_trace.h"
#include <stdint.h>

#define height_size 32
#define col_size 32
#define row_size 16
#define SIZE (row_size * col_size * height_size)
typedef int32_t TYPE;

static TYPE C[2];
static TYPE orig[SIZE];
static TYPE sol[SIZE];

#define INDX(rs, cs, i, j, k) ((i) + rs * ((j) + cs * (k)))

static void stencil3d_instr(void) {
    int i, j, k;
    for (k = 1; k < height_size - 1; k++)
        for (j = 1; j < col_size - 1; j++)
            for (i = 1; i < row_size - 1; i++) {
                TYPE v0 = __sa_log_i32(0, orig[INDX(row_size, col_size, i, j, k)]);
                TYPE v1 = orig[INDX(row_size, col_size, i + 1, j, k)];
                TYPE v2 = orig[INDX(row_size, col_size, i - 1, j, k)];
                TYPE v3 = orig[INDX(row_size, col_size, i, j + 1, k)];
                TYPE v4 = orig[INDX(row_size, col_size, i, j - 1, k)];
                TYPE neigh_sum = __sa_log_i32(1, v1 + v2 + v3 + v4);
                TYPE t = __sa_log_i32(2, C[1] * neigh_sum);
                TYPE r = __sa_log_i32(1, C[0] * v0 + t);
                sol[INDX(row_size, col_size, i, j, k)] = r;
                __sa_log_i32(3, r);
            }
}

int main(void) {
    int i;
    C[0] = 2;
    C[1] = 1;
    for (i = 0; i < SIZE; i++)
        orig[i] = (i % 1000) + 1;
    stencil3d_instr();
    return 0;
}
