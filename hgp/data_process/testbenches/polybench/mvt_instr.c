#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define N 64
typedef float DATA_TYPE;
static inline float __log32(int id, float v) { uint32_t u; memcpy(&u,&v,4); printf("SA %d 32 %u\n",id,u); return v; }

void mvt(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N]) {
    int i,j;
    DATA_TYPE bA[N][N],bx1[N],bx2[N],by1[N],by2[N],bo1[N],bo2[N];
    for(i=0;i<N;i++){bx1[i]=__log32(0,x1[i]);bx2[i]=__log32(1,x2[i]);by1[i]=__log32(2,y1[i]);by2[i]=__log32(3,y2[i]);bo1[i]=__log32(4,0.0f);bo2[i]=__log32(5,0.0f);for(j=0;j<N;j++)bA[i][j]=__log32(6,A[i][j]);}
    for(i=0;i<N;i++){
        bo1[i]=__log32(7,bx1[i]);
        for(j=0;j<N;j++) bo1[i]=__log32(8,bo1[i]+__log32(9,bA[i][j]*by1[j]));
    }
    for(i=0;i<N;i++){
        bo2[i]=__log32(10,bx2[i]);
        for(j=0;j<N;j++) bo2[i]=__log32(11,bo2[i]+__log32(12,bA[j][i]*by2[j]));
    }
    for(i=0;i<N;i++){x1_out[i]=__log32(13,bo1[i]);x2_out[i]=__log32(14,bo2[i]);}
}

int main(void) {
    static DATA_TYPE A[N][N],x1[N],x2[N],y1[N],y2[N],o1[N],o2[N];
    int i,j;
    for(i=0;i<N;i++){x1[i]=(i+1.0f)/N;x2[i]=(N-i+0.0f)/N;y1[i]=(float)(i%7)/7.0f;y2[i]=(float)(i%11)/11.0f;for(j=0;j<N;j++)A[i][j]=(float)((i+j*2)%13)/13.0f;}
    mvt(A,x1,x2,y1,y2,o1,o2);
    return 0;
}
