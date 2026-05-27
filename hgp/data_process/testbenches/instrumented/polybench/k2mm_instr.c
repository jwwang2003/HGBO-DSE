/* Instrumented k2mm kernel.
 * E = alpha * A * B * C + beta * D
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["k2mm"]):
 *   0,1,2,3: load A, B, C, D  (load)
 *   6,10:    fadd
 *   7,11:    fmul
 *   12:      store E
 */
#include "../sa_trace.h"

#define N 64

static void k2mm_instr(float alpha, float beta,
                       float A[N][N], float B[N][N],
                       float C[N][N], float D[N][N], float E_out[N][N]) {
    static float tmp[N][N];
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float a = __sa_log_f32(0, A[i][k]);
                float b = __sa_log_f32(1, B[k][j]);
                float p = __sa_log_f32(7, a * b);
                acc = __sa_log_f32(6, acc + p);
            }
            tmp[i][j] = alpha * acc;
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float t = tmp[i][k];
                float c = __sa_log_f32(2, C[k][j]);
                float p = __sa_log_f32(11, t * c);
                acc = __sa_log_f32(10, acc + p);
            }
            float d = __sa_log_f32(3, D[i][j]);
            E_out[i][j] = acc + beta * d;
            __sa_log_f32(12, E_out[i][j]);
        }
    }
}

int main(void) {
    static float A[N][N], B[N][N], C[N][N], D[N][N], E_out[N][N];
    float alpha = 1.5f, beta = 1.2f;
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i + j) % 11) / 11.0f;
            B[i][j] = (float)((i * 2 + j) % 13) / 13.0f;
            C[i][j] = (float)((i + j * 3) % 7) / 7.0f;
            D[i][j] = (float)(i % 5) / 5.0f;
        }
    k2mm_instr(alpha, beta, A, B, C, D, E_out);
    return 0;
}
