/* Instrumented stencil (2D) kernel.
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["stencil"]):
 *   0: load orig
 *   1: add
 *   2: mul
 *   3: store sol
 */
#include "../sa_trace.h"
#include <stdint.h>

#define row_size 128
#define col_size 64
#define f_size 9
typedef int32_t TYPE;

static TYPE orig[row_size * col_size];
static TYPE sol[row_size * col_size];
static TYPE filter[f_size];

static void stencil_instr(void) {
    int r, c, k1, k2;
    for (r = 0; r < row_size - 2; r++)
        for (c = 0; c < col_size - 2; c++) {
            TYPE acc = 0;
            for (k1 = 0; k1 < 3; k1++)
                for (k2 = 0; k2 < 3; k2++) {
                    TYPE v = __sa_log_i32(0, orig[(r + k1) * col_size + c + k2]);
                    TYPE f = filter[k1 * 3 + k2];
                    TYPE p = __sa_log_i32(2, v * f);
                    acc = __sa_log_i32(1, acc + p);
                }
            sol[r * col_size + c] = acc;
            __sa_log_i32(3, sol[r * col_size + c]);
        }
}

int main(void) {
    int i;
    for (i = 0; i < row_size * col_size; i++)
        orig[i] = (i % 1000) + 1;
    for (i = 0; i < f_size; i++) filter[i] = i + 1;
    stencil_instr();
    return 0;
}
