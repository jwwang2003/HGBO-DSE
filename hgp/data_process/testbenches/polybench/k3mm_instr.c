#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void k3mm(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N]) {
    int i,j,k;
    DATA_TYPE bA[N][N],bB[N][N],bC[N][N],bAB[N][N],bE[N][N];
    for(i=0;i<N;i++) for(j=0;j<N;j++){bA[i][j]=__log32(0,A[i][j]);bB[i][j]=__log32(1,B[i][j]);bC[i][j]=__log32(2,C[i][j]);bAB[i][j]=__log32(3,0.0f);bE[i][j]=__log32(4,0.0f);}
    for(i=0;i<N;i++) for(j=0;j<N;j++) for(k=0;k<N;k++) bAB[i][j]=__log32(5,bAB[i][j]+__log32(6,bA[i][k]*bB[k][j]));
    for(i=0;i<N;i++) for(j=0;j<N;j++) for(k=0;k<N;k++) bE[i][j]=__log32(7,bE[i][j]+__log32(8,bAB[i][k]*bC[k][j]));
    for(i=0;i<N;i++) for(j=0;j<N;j++) E_out[i][j]=__log32(9,bE[i][j]);
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],C[N][N],D[N][N],E[N][N];
    int i,j;
    for(i=0;i<N;i++) for(j=0;j<N;j++){A[i][j]=(float)((i+j)%11)/11.0f;B[i][j]=(float)((i*2+j)%13)/13.0f;C[i][j]=(float)((i+j*3)%7)/7.0f;}
    k3mm(A,B,C,D,E);
    return 0;
}
