/* Instrumented bfs for switching activity extraction */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>

#define SCALE 8
#define EDGE_FACTOR 16
#define N_NODES (1<<SCALE)
#define N_EDGES (N_NODES*EDGE_FACTOR)
#define N_LEVELS 10
#define SECTION_TERMINATED -1
#define MAX_N_EDGES 158

typedef uint64_t edge_index_t;
typedef uint64_t node_index_t;
typedef int8_t level_t;
#define MAX_LEVEL INT8_MAX

typedef struct { node_index_t dst; } edge_t;
typedef struct { edge_index_t edge_begin; edge_index_t edge_end; } node_t;

static inline uint64_t __log64(int id, uint64_t v) { printf("SA %d 64 %llu\n", id, (unsigned long long)v); return v; }
static inline int8_t __log8(int id, int8_t v) { printf("SA %d 8 %d\n", id, (int)v); return v; }

#define Q_PUSH(node) { queue[q_in==0?N_NODES-1:q_in-1]=node; q_in=(q_in+1)%N_NODES; }
#define Q_PEEK() (queue[q_out])
#define Q_POP() { q_out = (q_out+1)%N_NODES; }
#define Q_EMPTY() (q_in>q_out ? q_in==q_out+1 : (q_in==0)&&(q_out==N_NODES-1))

void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES], node_index_t starting_node,
         level_t level[N_NODES], edge_index_t level_counts[N_LEVELS]) {
    node_index_t queue[N_NODES];
    node_index_t q_in, q_out, dummy, n;
    edge_index_t e;
    int i;
    q_in = 1; q_out = 0;
    level[starting_node] = __log8(0, 0);
    level_counts[0] = __log64(1, 1);
    Q_PUSH(starting_node);

    for (dummy = 0; dummy < N_NODES; dummy++) {
        if (Q_EMPTY()) break;
        n = Q_PEEK(); Q_POP();
        e = __log64(2, nodes[n].edge_begin);
        edge_index_t tmp_end = __log64(3, nodes[n].edge_end);
        for (i = 0; i < MAX_N_EDGES; i++) {
            if (e < tmp_end) {
                node_index_t dst = __log64(4, edges[e].dst);
                if (level[dst] == MAX_LEVEL) {
                    level[dst] = __log8(5, level[n] + 1);
                    level_counts[(int)level[dst]] = __log64(6, level_counts[(int)level[dst]] + 1);
                    Q_PUSH(dst);
                }
                e = __log64(7, e + 1);
            }
        }
    }
}

int main(void) {
    static node_t nodes[N_NODES];
    static edge_t edges[N_EDGES];
    static level_t level[N_NODES];
    static edge_index_t level_counts[N_LEVELS];
    int i;
    for (i = 0; i < N_NODES; i++) {
        nodes[i].edge_begin = (edge_index_t)(i * EDGE_FACTOR);
        nodes[i].edge_end   = (edge_index_t)((i + 1) * EDGE_FACTOR);
    }
    for (i = 0; i < N_EDGES; i++)
        edges[i].dst = (node_index_t)((i / EDGE_FACTOR + 1) % N_NODES);
    memset(level, -1, sizeof(level));
    memset(level_counts, 0, sizeof(level_counts));
    bfs(nodes, edges, 0, level, level_counts);
    return 0;
}
