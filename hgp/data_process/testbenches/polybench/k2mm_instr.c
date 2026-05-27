#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void k2mm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N]) {
    int i,j,k;
    DATA_TYPE bA[N][N],bB[N][N],bC[N][N],bD[N][N],bE[N][N];
    for(i=0;i<N;i++) for(j=0;j<N;j++){bA[i][j]=__log32(0,A[i][j]);bB[i][j]=__log32(1,B[i][j]);bC[i][j]=__log32(2,C[i][j]);bD[i][j]=__log32(3,D[i][j]);bE[i][j]=__log32(4,0.0f);}
    for(i=0;i<N;i++) for(j=0;j<N;j++){
        DATA_TYPE tmp=__log32(5,0.0f);
        for(k=0;k<N;k++) tmp=__log32(6,tmp+__log32(7,alpha*bA[i][k]*bB[k][j]));
        bE[i][j]=__log32(8,tmp);
    }
    for(i=0;i<N;i++) for(j=0;j<N;j++){
        DATA_TYPE tmp2=__log32(9,0.0f);
        for(k=0;k<N;k++) tmp2=__log32(10,tmp2+__log32(11,beta*bE[i][k]*bC[k][j]));
        E_out[i][j]=__log32(12,tmp2+bD[i][j]);
    }
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],C[N][N],D[N][N],E[N][N];
    int i,j;
    for(i=0;i<N;i++) for(j=0;j<N;j++){A[i][j]=(float)((i+j)%11)/11.0f;B[i][j]=(float)((i*2+j)%13)/13.0f;C[i][j]=(float)((i+j*3)%7)/7.0f;D[i][j]=(float)(i%5)/5.0f;}
    k2mm(1.5f,1.2f,A,B,C,D,E);
    return 0;
}
