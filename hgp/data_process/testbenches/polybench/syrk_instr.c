#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void syrk(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N]) {
    int i,j,k;
    DATA_TYPE bA[N][N],bC[N][N];
    for(i=0;i<N;i++) for(j=0;j<N;j++){bA[i][j]=__log32(0,A[i][j]);bC[i][j]=__log32(1,beta*B[i][j]);}
    for(i=0;i<N;i++) for(j=0;j<=i;j++) for(k=0;k<N;k++)
        bC[i][j]=__log32(2,bC[i][j]+__log32(3,alpha*bA[i][k]*bA[j][k]));
    for(i=0;i<N;i++) for(j=0;j<N;j++) C_out[i][j]=__log32(4,bC[i][j]);
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],C[N][N];
    int i,j;
    for(i=0;i<N;i++) for(j=0;j<N;j++){A[i][j]=(float)((i+j)%11)/11.0f;B[i][j]=(float)(i==j?1:0);}
    syrk(1.5f,1.2f,A,B,C);
    return 0;
}
