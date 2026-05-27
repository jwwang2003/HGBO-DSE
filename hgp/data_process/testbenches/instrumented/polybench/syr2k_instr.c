/* Instrumented syr2k kernel.
 * D = alpha * (A*B^T + B*A^T) + beta * C
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["syr2k"]):
 *   0,1,2: load A, B, C
 *   4:     fadd
 *   5,6:   fmul
 *   7:     store D
 */
#include "../sa_trace.h"

#define N 64

static void syr2k_instr(float alpha, float beta,
                        float A[N][N], float B[N][N],
                        float C[N][N], float D_out[N][N]) {
    int i, j, k;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float acc = __sa_log_f32(2, C[i][j]) * beta;
            for (k = 0; k < N; k++) {
                float aik = __sa_log_f32(0, A[i][k]);
                float bjk = __sa_log_f32(1, B[j][k]);
                float ajk = __sa_log_f32(0, A[j][k]);
                float bik = __sa_log_f32(1, B[i][k]);
                float p1 = __sa_log_f32(5, aik * bjk);
                float p2 = __sa_log_f32(6, bik * ajk);
                acc = __sa_log_f32(4, acc + alpha * (p1 + p2));
            }
            D_out[i][j] = acc;
            __sa_log_f32(7, D_out[i][j]);
        }
}

int main(void) {
    static float A[N][N], B[N][N], C[N][N], D_out[N][N];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C[i][j] = (float)(i == j ? 1 : 0);
        }
    syr2k_instr(alpha, beta, A, B, C, D_out);
    return 0;
}
