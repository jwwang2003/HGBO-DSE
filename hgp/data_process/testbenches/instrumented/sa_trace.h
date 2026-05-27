#ifndef SA_TRACE_H
#define SA_TRACE_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* SA tracing macros: log <op_id, bits, value-as-uint> per invocation. */
static void __sa_trace(int op_id, int bits, uint64_t val) {
    printf("SA %d %d %llu\n", op_id, bits, (unsigned long long)val);
}
static inline float __sa_log_f32(int op_id, float v) {
    uint32_t u; memcpy(&u, &v, 4);
    __sa_trace(op_id, 32, (uint64_t)u);
    return v;
}
static inline double __sa_log_f64(int op_id, double v) {
    uint64_t u; memcpy(&u, &v, 8);
    __sa_trace(op_id, 64, u);
    return v;
}
static inline int32_t __sa_log_i32(int op_id, int32_t v) {
    __sa_trace(op_id, 32, (uint64_t)(uint32_t)v);
    return v;
}
static inline uint64_t __sa_log_u64(int op_id, uint64_t v) {
    __sa_trace(op_id, 64, v);
    return v;
}
static inline int8_t __sa_log_i8(int op_id, int8_t v) {
    __sa_trace(op_id, 8, (uint64_t)(uint8_t)v);
    return v;
}

#endif /* SA_TRACE_H */
