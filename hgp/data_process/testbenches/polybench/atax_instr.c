/* Instrumented atax kernel for ATAPP switching activity extraction.
 * Each arithmetic/load result is logged via __log32 so SA/AR can be computed.
 * op_ids correspond (in order) to the CDFG nodes as ordered in cdfg_node_dict.csv.
 *
 * Trace format: SA <op_id> <bits> <uint_value>
 * SA  = switching activity op log
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define N 64
typedef float DATA_TYPE;

static inline float __log32(int id, float v) {
    uint32_t u; memcpy(&u, &v, 4);
    printf("SA %d 32 %u\n", id, u);
    return v;
}
static inline int __log_int(int id, int v) {
    printf("SA %d 32 %u\n", id, (unsigned)v);
    return v;
}

static int __op = 0;

void atax(DATA_TYPE A[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
    int i, j;
    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_x[N];
    DATA_TYPE buff_y_out[N];
    DATA_TYPE tmp1[N];
    __op = 0;

    for (i = 0; i < N; i++) {
        buff_x[i]      = __log32(0, x[i]);
        buff_y_out[i]  = __log32(1, 0.0f);
        tmp1[i]        = __log32(2, 0.0f);
        for (j = 0; j < N; j++)
            buff_A[i][j] = __log32(3, A[i][j]);
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            DATA_TYPE mul = __log32(4, buff_A[i][j] * buff_x[j]);
            tmp1[i]       = __log32(5, tmp1[i] + mul);
        }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            DATA_TYPE mul2   = __log32(6, buff_A[i][j] * tmp1[i]);
            buff_y_out[j]    = __log32(7, buff_y_out[j] + mul2);
        }

    for (i = 0; i < N; i++)
        y_out[i] = __log32(8, buff_y_out[i]);
}

int main(void) {
    static DATA_TYPE A[N][N], x[N], y_out[N];
    int i, j;
    for (i = 0; i < N; i++) {
        x[i] = (DATA_TYPE)(i + 1) / (DATA_TYPE)N;
        for (j = 0; j < N; j++)
            A[i][j] = (DATA_TYPE)((i * N + j) % N) / (DATA_TYPE)N + 0.1f;
    }
    memset(y_out, 0, sizeof(y_out));
    atax(A, x, y_out);
    return 0;
}
