/* Instrumented k3mm kernel.
 * E = A * B * C * D
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["k3mm"]):
 *   0,1,2: load A, B, C  (also covers D loads via op 2)
 *   5,7:   fadd
 *   6,8:   fmul
 *   9:     store E
 */
#include "../sa_trace.h"

#define N 64

static void k3mm_instr(float A[N][N], float B[N][N],
                       float C[N][N], float D[N][N], float E_out[N][N]) {
    static float tmp1[N][N], tmp2[N][N];
    int i, j, k;
    /* tmp1 = A * B */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float a = __sa_log_f32(0, A[i][k]);
                float b = __sa_log_f32(1, B[k][j]);
                float p = __sa_log_f32(6, a * b);
                acc = __sa_log_f32(5, acc + p);
            }
            tmp1[i][j] = acc;
        }
    /* tmp2 = tmp1 * C */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float c = __sa_log_f32(2, C[k][j]);
                float p = __sa_log_f32(8, tmp1[i][k] * c);
                acc = __sa_log_f32(7, acc + p);
            }
            tmp2[i][j] = acc;
        }
    /* E = tmp2 * D */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float d = __sa_log_f32(2, D[k][j]);
                float p = __sa_log_f32(8, tmp2[i][k] * d);
                acc = __sa_log_f32(7, acc + p);
            }
            E_out[i][j] = acc;
            __sa_log_f32(9, E_out[i][j]);
        }
}

int main(void) {
    static float A[N][N], B[N][N], C[N][N], D[N][N], E_out[N][N];
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C[i][j] = (float)((i + j * 3) % 7) / 7.0f;
            D[i][j] = (float)(i % 5) / 5.0f;
        }
    k3mm_instr(A, B, C, D, E_out);
    return 0;
}
