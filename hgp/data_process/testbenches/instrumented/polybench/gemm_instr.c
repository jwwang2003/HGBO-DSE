/* Instrumented gemm kernel.
 * D = alpha * A * B + beta * C
 *
 * Op-id assignments (must match _KERNEL_OPCODE_MAP["gemm"]):
 *   0,1,2: load A, B, C   (load)
 *   5,6:   fadd            (fadd)
 *   7:     fmul            (fmul)
 *   8:     store D         (store)
 */
#include "../sa_trace.h"

#define N 64

static void gemm_instr(float alpha, float beta,
                       float A[N][N], float B[N][N],
                       float C[N][N], float D[N][N]) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float acc = 0.0f;
            for (k = 0; k < N; k++) {
                float a = __sa_log_f32(0, A[i][k]);
                float b = __sa_log_f32(1, B[k][j]);
                float prod = __sa_log_f32(7, a * b);
                acc = __sa_log_f32(5, acc + prod);
            }
            float c = __sa_log_f32(2, C[i][j]);
            /* D = alpha * acc + beta * C  -> use op 6 for the second fadd */
            float scaled = alpha * acc + beta * c;
            D[i][j] = __sa_log_f32(6, scaled);
            __sa_log_f32(8, D[i][j]);
        }
    }
}

int main(void) {
    static float A[N][N], B[N][N], C[N][N], D[N][N];
    float alpha = 1.5f, beta = 0.5f;
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (float)((i * N + j) % 13) / 13.0f;
            B[i][j] = (float)((i + j * 3) % 17) / 17.0f;
            C[i][j] = (float)(i % 7) / 7.0f;
            D[i][j] = 0.0f;
        }
    }
    gemm_instr(alpha, beta, A, B, C, D);
    return 0;
}
