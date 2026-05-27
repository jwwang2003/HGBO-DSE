#include "stencil.h"
#include <string.h>

int main(void) {
    static TYPE orig[row_size * col_size];
    static TYPE sol[row_size * col_size];
    static TYPE filter[f_size];
    int i;

    for (i = 0; i < row_size * col_size; i++)
        orig[i] = (TYPE)(i % MAX) + MIN;
    memset(sol, 0, sizeof(sol));
    for (i = 0; i < f_size; i++)
        filter[i] = (TYPE)(i + 1);

    stencil(orig, sol, filter);
    return 0;
}
