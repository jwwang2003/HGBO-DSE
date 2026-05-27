/* Instrumented bfs kernel.
 *
 * Op-id map (matches _KERNEL_OPCODE_MAP["bfs"]):
 *   0:    icmp (frontier check)
 *   2,3,4: load nodes/edges/level
 *   5,6:  store level/level_counts
 *   7:    add (edge index increment)
 */
#include "../sa_trace.h"
#include <stdint.h>

#define SCALE 8
#define EDGE_FACTOR 16
#define N_NODES (1<<SCALE)
#define N_EDGES (N_NODES*EDGE_FACTOR)
#define N_LEVELS 10

typedef int32_t node_index_t;
typedef int32_t edge_index_t;
typedef int8_t  level_t;

typedef struct { edge_index_t edge_begin, edge_end; } node_t;
typedef struct { node_index_t dst; } edge_t;

static node_t  nodes[N_NODES];
static edge_t  edges[N_EDGES];
static level_t level[N_NODES];
static edge_index_t level_counts[N_LEVELS];

static void bfs_instr(node_index_t start) {
    int i;
    for (i = 0; i < N_NODES; i++) level[i] = -1;
    for (i = 0; i < N_LEVELS; i++) level_counts[i] = 0;
    level[start] = 0;
    level_counts[0] = 1;

    for (level_t lvl = 0; lvl < N_LEVELS - 1; lvl++) {
        for (i = 0; i < N_NODES; i++) {
            level_t li = __sa_log_i8(2, level[i]);
            int8_t check = (int8_t)__sa_log_i32(0, (int32_t)(li == lvl));
            if (!check) continue;
            edge_index_t b = __sa_log_i32(3, nodes[i].edge_begin);
            edge_index_t e = nodes[i].edge_end;
            for (edge_index_t k = b; k < e; k++) {
                node_index_t dst = __sa_log_i32(4, edges[k].dst);
                if (level[dst] < 0) {
                    level[dst] = lvl + 1;
                    __sa_log_i8(5, level[dst]);
                    level_counts[lvl + 1] = __sa_log_i32(7, level_counts[lvl + 1] + 1);
                    __sa_log_i32(6, level_counts[lvl + 1]);
                }
            }
        }
    }
}

int main(void) {
    int i;
    for (i = 0; i < N_NODES; i++) {
        nodes[i].edge_begin = i * EDGE_FACTOR;
        nodes[i].edge_end   = (i + 1) * EDGE_FACTOR;
    }
    for (i = 0; i < N_EDGES; i++)
        edges[i].dst = (i / EDGE_FACTOR + 1) % N_NODES;
    bfs_instr(0);
    return 0;
}
