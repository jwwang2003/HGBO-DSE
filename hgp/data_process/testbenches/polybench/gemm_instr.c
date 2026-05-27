#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void gemm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N]) {
    int i,j,k;
    DATA_TYPE bA[N][N],bB[N][N],bC[N][N],bD[N][N];
    for(i=0;i<N;i++) for(j=0;j<N;j++){bA[i][j]=__log32(0,A[i][j]);bB[i][j]=__log32(1,B[i][j]);bC[i][j]=__log32(2,C[i][j]);bD[i][j]=__log32(3,0.0f);}
    for(i=0;i<N;i++) for(j=0;j<N;j++) {
        bD[i][j]=__log32(4,beta*bC[i][j]);
        for(k=0;k<N;k++) bD[i][j]=__log32(5,bD[i][j]+__log32(6,alpha*__log32(7,bA[i][k]*bB[k][j])));
    }
    for(i=0;i<N;i++) for(j=0;j<N;j++) D[i][j]=__log32(8,bD[i][j]);
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],C[N][N],D[N][N];
    int i,j;
    for(i=0;i<N;i++) for(j=0;j<N;j++){A[i][j]=(float)((i*N+j)%13)/13.0f;B[i][j]=(float)((i+j*3)%17)/17.0f;C[i][j]=(float)(i%7)/7.0f;}
    gemm(1.5f,0.5f,A,B,C,D);
    return 0;
}
