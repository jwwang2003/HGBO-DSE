/* Instrumented spmv kernel (CRS).
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["spmv"]):
 *   1,2: load val/vec
 *   3:   add (accumulator)
 *   4:   mul
 *   5:   store out
 */
#include "../sa_trace.h"
#include <stdint.h>

#define NNZ 1666
#define N 494
typedef double TYPE;

static TYPE val[NNZ];
static int32_t cols[NNZ];
static int32_t rowDelimiters[N + 1];
static TYPE vec[N], out[N];

static void spmv_instr(void) {
    for (int i = 0; i < N; i++) {
        TYPE acc = 0.0;
        int32_t lo = rowDelimiters[i];
        int32_t hi = rowDelimiters[i + 1];
        for (int k = lo; k < hi; k++) {
            TYPE v = __sa_log_f64(1, val[k]);
            TYPE x = __sa_log_f64(2, vec[cols[k]]);
            TYPE p = v * x;
            acc = __sa_log_f64(3, acc + __sa_log_f64(4, p));
        }
        out[i] = acc;
        __sa_log_f64(5, out[i]);
    }
}

int main(void) {
    int i;
    for (i = 0; i < N; i++) {
        val[i] = (TYPE)(i + 1) / (TYPE)N;
        cols[i] = i;
        vec[i] = (TYPE)(N - i) / (TYPE)N;
        rowDelimiters[i] = i;
    }
    for (i = N; i < NNZ; i++) { val[i] = 0; cols[i] = 0; }
    rowDelimiters[N] = NNZ;
    spmv_instr();
    return 0;
}
