#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void syr2k(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N]) {
    int i,j,k;
    DATA_TYPE bA[N][N],bB[N][N],bC[N][N],bD[N][N];
    for(i=0;i<N;i++) for(j=0;j<N;j++){bA[i][j]=__log32(0,A[i][j]);bB[i][j]=__log32(1,B[i][j]);bC[i][j]=__log32(2,C[i][j]);bD[i][j]=__log32(3,beta*C[i][j]);}
    for(i=0;i<N;i++) for(j=0;j<=i;j++) for(k=0;k<N;k++) {
        bD[i][j]=__log32(4,bD[i][j]+__log32(5,alpha*bA[i][k]*bB[j][k])+__log32(6,alpha*bB[i][k]*bA[j][k]));
    }
    for(i=0;i<N;i++) for(j=0;j<N;j++) D_out[i][j]=__log32(7,bD[i][j]);
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],C[N][N],D[N][N];
    int i,j;
    for(i=0;i<N;i++) for(j=0;j<N;j++){A[i][j]=(float)((i+j)%11)/11.0f;B[i][j]=(float)((i*2+j)%13)/13.0f;C[i][j]=(float)(i==j?1:0);}
    syr2k(1.5f,1.2f,A,B,C,D);
    return 0;
}
