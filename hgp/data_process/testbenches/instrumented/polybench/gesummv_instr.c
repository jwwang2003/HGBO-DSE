/* Instrumented gesummv kernel.
 * y = alpha * A * x + beta * B * x
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["gesummv"]):
 *   0,3,4: load A, B, x  (load)
 *   5,7:   fadd
 *   6,8:   fmul
 *   9:     store y_out
 */
#include "../sa_trace.h"

#define N 64

static void gesummv_instr(float alpha, float beta,
                          float A[N][N], float B[N][N],
                          float x[N], float y_out[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        float acc1 = 0.0f, acc2 = 0.0f;
        for (j = 0; j < N; j++) {
            float a  = __sa_log_f32(0, A[i][j]);
            float b  = __sa_log_f32(3, B[i][j]);
            float xj = __sa_log_f32(4, x[j]);
            float p1 = __sa_log_f32(6, a * xj);
            acc1 = __sa_log_f32(5, acc1 + p1);
            float p2 = __sa_log_f32(8, b * xj);
            acc2 = __sa_log_f32(7, acc2 + p2);
        }
        y_out[i] = alpha * acc1 + beta * acc2;
        __sa_log_f32(9, y_out[i]);
    }
}

int main(void) {
    static float A[N][N], B[N][N], x[N], y_out[N];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < N; i++) {
        x[i] = (float)(i + 1) / (float)N;
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 3 + j * 2) % 13) / 13.0f;
        }
    }
    gesummv_instr(alpha, beta, A, B, x, y_out);
    return 0;
}
