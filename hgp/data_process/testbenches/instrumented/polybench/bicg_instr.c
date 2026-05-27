/* Instrumented bicg kernel for ATAPP-style switching activity extraction.
 *
 * Op-id assignments (must match _KERNEL_OPCODE_MAP["bicg"]):
 *   0,1,4: load A, p, r       (load)
 *   5,7:   fadd                (fadd)
 *   6,8:   fmul                (fmul)
 *   9,10:  store s_out, q_out  (store)
 */
#include "../sa_trace.h"

#define N 64

static void bicg_instr(float A[N][N], float p[N], float r[N],
                       float s_out[N], float q_out[N]) {
    int i, j;
    for (i = 0; i < N; i++) { s_out[i] = 0.0f; q_out[i] = 0.0f; }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float a  = __sa_log_f32(0, A[i][j]);
            float pj = __sa_log_f32(1, p[j]);
            float ri = __sa_log_f32(4, r[i]);
            /* s += r[i] * A[i][j] */
            float prod1 = __sa_log_f32(6, ri * a);
            s_out[j] = __sa_log_f32(5, s_out[j] + prod1);
            /* q += A[i][j] * p[j] */
            float prod2 = __sa_log_f32(8, a * pj);
            q_out[i] = __sa_log_f32(7, q_out[i] + prod2);
        }
    }
    for (i = 0; i < N; i++) {
        __sa_log_f32(9, s_out[i]);
        __sa_log_f32(10, q_out[i]);
    }
}

int main(void) {
    static float A[N][N], p[N], r[N], s_out[N], q_out[N];
    int i, j;
    for (i = 0; i < N; i++) {
        p[i] = (float)(i + 1) / (float)N;
        r[i] = (float)(N - i) / (float)N;
        for (j = 0; j < N; j++)
            A[i][j] = (float)((i + j + 1) % 13) / 13.0f;
    }
    bicg_instr(A, p, r, s_out, q_out);
    return 0;
}
