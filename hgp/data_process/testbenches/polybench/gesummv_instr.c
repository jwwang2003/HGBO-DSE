#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void gesummv(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N]) {
    int i,j;
    DATA_TYPE bA[N][N],bB[N][N],bx[N],tmp[N],by[N];
    for(i=0;i<N;i++){bx[i]=__log32(0,x[i]);tmp[i]=__log32(1,0.0f);by[i]=__log32(2,0.0f);for(j=0;j<N;j++){bA[i][j]=__log32(3,A[i][j]);bB[i][j]=__log32(4,B[i][j]);}}
    for(i=0;i<N;i++) for(j=0;j<N;j++){
        tmp[i]=__log32(5,tmp[i]+__log32(6,bA[i][j]*bx[j]));
        by[i]=__log32(7,by[i]+__log32(8,bB[i][j]*bx[j]));
    }
    for(i=0;i<N;i++) y_out[i]=__log32(9,alpha*tmp[i]+beta*by[i]);
}

int main(void) {
    static DATA_TYPE A[N][N],B[N][N],x[N],y[N];
    int i,j;
    for(i=0;i<N;i++){x[i]=(i+1.0f)/N;for(j=0;j<N;j++){A[i][j]=(float)((i+j)%11)/11.0f;B[i][j]=(float)((i*3+j*2)%13)/13.0f;}}
    gesummv(1.5f,1.2f,A,B,x,y);
    return 0;
}
