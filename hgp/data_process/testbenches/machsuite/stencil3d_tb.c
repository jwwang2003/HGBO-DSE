#include "stencil.h"
#include <string.h>

int main(void) {
    static TYPE C[2];
    static TYPE orig[SIZE];
    static TYPE sol[SIZE];
    int i;

    C[0] = 2;
    C[1] = 1;
    for (i = 0; i < SIZE; i++)
        orig[i] = (TYPE)(i % MAX) + MIN;
    memset(sol, 0, sizeof(sol));

    stencil3d(C, orig, sol);
    return 0;
}
