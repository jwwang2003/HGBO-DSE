/* Instrumented syrk kernel.
 * C = alpha * A * A^T + beta * C
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["syrk"]):
 *   0:  load A
 *   2:  fadd
 *   3:  fmul
 *   4:  store C_out
 */
#include "../sa_trace.h"

#define N 64

static void syrk_instr(float alpha, float beta,
                       float A[N][N], float B[N][N], float C_out[N][N]) {
    int i, j, k;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float acc = beta * C_out[i][j];
            for (k = 0; k < N; k++) {
                float a1 = __sa_log_f32(0, A[i][k]);
                float a2 = __sa_log_f32(0, A[j][k]);
                float p  = __sa_log_f32(3, a1 * a2);
                acc = __sa_log_f32(2, acc + alpha * p);
            }
            C_out[i][j] = acc;
            __sa_log_f32(4, C_out[i][j]);
        }
}

int main(void) {
    static float A[N][N], B[N][N], C_out[N][N];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C_out[i][j] = 0.0f;
        }
    syrk_instr(alpha, beta, A, B, C_out);
    return 0;
}
