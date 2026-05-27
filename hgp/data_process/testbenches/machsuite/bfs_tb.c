#include "bfs.h"
#include <string.h>

int main(void) {
    static node_t nodes[N_NODES];
    static edge_t edges[N_EDGES];
    static level_t level[N_NODES];
    static edge_index_t level_counts[N_LEVELS];
    int i;

    /* Build a simple chain graph: node i -> node i+1 */
    for (i = 0; i < N_NODES; i++) {
        nodes[i].edge_begin = (edge_index_t)(i * EDGE_FACTOR);
        nodes[i].edge_end   = (edge_index_t)((i + 1) * EDGE_FACTOR);
    }
    for (i = 0; i < N_EDGES; i++)
        edges[i].dst = (node_index_t)((i / EDGE_FACTOR + 1) % N_NODES);

    memset(level, -1, sizeof(level));
    memset(level_counts, 0, sizeof(level_counts));

    node_index_t start = 0;
    bfs(nodes, edges, start, level, level_counts);
    return 0;
}
