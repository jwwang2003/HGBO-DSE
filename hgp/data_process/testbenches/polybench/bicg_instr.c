#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void bicg(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N]) {
    int i, j;
    DATA_TYPE buff_A[N][N], buff_p[N], buff_r[N], buff_s[N], buff_q[N];
    for (i = 0; i < N; i++) {
        buff_p[i] = __log32(0, p[i]);
        buff_r[i] = __log32(1, r[i]);
        buff_s[i] = __log32(2, 0.0f);
        buff_q[i] = __log32(3, 0.0f);
        for (j = 0; j < N; j++) buff_A[i][j] = __log32(4, A[i][j]);
    }
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            buff_s[j] = __log32(5, buff_s[j] + __log32(6, buff_A[i][j] * buff_r[i]));
            buff_q[i] = __log32(7, buff_q[i] + __log32(8, buff_A[i][j] * buff_p[j]));
        }
    for (i = 0; i < N; i++) { s_out[i] = __log32(9, buff_s[i]); q_out[i] = __log32(10, buff_q[i]); }
}

int main(void) {
    static DATA_TYPE A[N][N], p[N], r[N], s[N], q[N];
    int i, j;
    for (i=0;i<N;i++){p[i]=(i+1.0f)/N;r[i]=(N-i+0.0f)/N;for(j=0;j<N;j++)A[i][j]=(float)((i+j+1)%N)/(float)N;}
    bicg(A,p,r,s,q);
    return 0;
}
