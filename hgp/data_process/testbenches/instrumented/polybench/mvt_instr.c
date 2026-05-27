/* Instrumented mvt kernel.
 * x1_out = A * y1 + x1
 * x2_out = A^T * y2 + x2
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["mvt"]):
 *   0,1,2,3,6: load A, x1, x2, y1, y2
 *   8,11:      fadd
 *   9,12:      fmul
 *   13,14:     store x1_out, x2_out
 */
#include "../sa_trace.h"

#define N 64

static void mvt_instr(float A[N][N], float x1[N], float x2[N],
                      float y1[N], float y2[N],
                      float x1_out[N], float x2_out[N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        float acc1 = __sa_log_f32(1, x1[i]);
        for (j = 0; j < N; j++) {
            float a  = __sa_log_f32(0, A[i][j]);
            float yj = __sa_log_f32(3, y1[j]);
            float p  = __sa_log_f32(9, a * yj);
            acc1 = __sa_log_f32(8, acc1 + p);
        }
        x1_out[i] = acc1;
        __sa_log_f32(13, x1_out[i]);
    }
    for (i = 0; i < N; i++) {
        float acc2 = __sa_log_f32(2, x2[i]);
        for (j = 0; j < N; j++) {
            float a  = __sa_log_f32(0, A[j][i]);
            float yj = __sa_log_f32(6, y2[j]);
            float p  = __sa_log_f32(12, a * yj);
            acc2 = __sa_log_f32(11, acc2 + p);
        }
        x2_out[i] = acc2;
        __sa_log_f32(14, x2_out[i]);
    }
}

int main(void) {
    static float A[N][N], x1[N], x2[N], y1[N], y2[N], x1_out[N], x2_out[N];
    int i, j;
    for (i = 0; i < N; i++) {
        x1[i] = (float)(i + 1) / (float)N;
        x2[i] = (float)(N - i) / (float)N;
        y1[i] = (float)(i % 7) / 7.0f;
        y2[i] = (float)(i % 11) / 11.0f;
        for (j = 0; j < N; j++)
            A[i][j] = (float)((i + j * 2) % 13) / 13.0f;
    }
    mvt_instr(A, x1, x2, y1, y2, x1_out, x2_out);
    return 0;
}
