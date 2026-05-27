/* Instrumented atax kernel for ATAPP-style switching activity extraction.
 *
 * Op-id assignments (must match _KERNEL_OPCODE_MAP["atax"] in switching_activity.py):
 *   0: load A[i][j]            (load)
 *   1: load x[j]               (load)
 *   2: load tmp1[i] (running)  (load)
 *   3: load buff_y_out[j]      (load)
 *   4: fmul A[i][j] * x[j]     (fmul, used in lp1 inner loop)
 *   5: fadd tmp1[i] + product  (fadd, accumulator for lp1)
 *   6: fmul A[i][j] * tmp1[i]  (fmul, used in lp3 inner loop)
 *   7: fadd y_out[j] + product (fadd, accumulator for lp3)
 *   8: store y_out[i]          (store)
 */
#include "../sa_trace.h"

#define N 64

static void atax_instr(float A[N][N], float x[N], float y_out[N]) {
    float tmp[N];
    int i, j;
    for (i = 0; i < N; i++) {
        tmp[i] = 0.0f;
        y_out[i] = 0.0f;
    }
    /* lp1: tmp[i] += A[i][j] * x[j] */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float a = __sa_log_f32(0, A[i][j]);
            float xj = __sa_log_f32(1, x[j]);
            float t  = __sa_log_f32(2, tmp[i]);
            float prod = __sa_log_f32(4, a * xj);
            tmp[i] = __sa_log_f32(5, t + prod);
        }
    }
    /* lp3: y_out[j] += A[i][j] * tmp[i] */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float a = __sa_log_f32(0, A[i][j]);
            float ti = __sa_log_f32(2, tmp[i]);
            float yj = __sa_log_f32(3, y_out[j]);
            float prod = __sa_log_f32(6, a * ti);
            y_out[j] = __sa_log_f32(7, yj + prod);
        }
    }
    /* writeback */
    for (i = 0; i < N; i++) {
        __sa_log_f32(8, y_out[i]);
    }
}

int main(void) {
    static float A[N][N], x[N], y_out[N];
    int i, j;
    for (i = 0; i < N; i++) {
        x[i] = (float)(i + 1) / (float)N;
        for (j = 0; j < N; j++)
            A[i][j] = (float)((i * N + j) % 13) / 13.0f + 0.1f;
    }
    atax_instr(A, x, y_out);
    return 0;
}
