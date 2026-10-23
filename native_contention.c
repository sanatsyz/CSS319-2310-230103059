/* Native kernels keep the increment loops outside Python's GIL. */
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

void contention_increment(uint64_t *slot, uint64_t iterations) {
    /* Volatile forces all requested loads/stores; each thread owns its slot. */
    volatile uint64_t *count = slot;
    for (uint64_t i = 0; i < iterations; ++i) {
        *count = *count + 1;
    }
}

typedef struct {
    _Atomic uint64_t value;
    pthread_mutex_t mutex;
} contention_counter;

void *contention_create_counter(void) {
    contention_counter *counter = malloc(sizeof(*counter));
    if (counter == NULL) {
        return NULL;
    }
    atomic_init(&counter->value, 0);
    if (pthread_mutex_init(&counter->mutex, NULL) != 0) {
        free(counter);
        return NULL;
    }
    return counter;
}

void contention_free_counter(void *address) {
    contention_counter *counter = address;
    pthread_mutex_destroy(&counter->mutex);
    free(counter);
}

void contention_reset_counter(void *address) {
    contention_counter *counter = address;
    atomic_store_explicit(&counter->value, 0, memory_order_relaxed);
}

uint64_t contention_get_counter(void *address) {
    contention_counter *counter = address;
    return atomic_load_explicit(&counter->value, memory_order_relaxed);
}

void contention_racy_increment(void *address, uint64_t iterations) {
    contention_counter *counter = address;
    for (uint64_t i = 0; i < iterations; ++i) {
        /* The compound operation is NOT an atomic read-modify-write.
         * Relaxed individual accesses avoid undefined C data-race behavior,
         * but provide no ordering or protection against lost updates.
         */
        uint64_t old = atomic_load_explicit(&counter->value, memory_order_relaxed);
        atomic_store_explicit(&counter->value, old + 1, memory_order_relaxed);
    }
}

void contention_locked_increment(void *address, uint64_t iterations) {
    contention_counter *counter = address;
    for (uint64_t i = 0; i < iterations; ++i) {
        pthread_mutex_lock(&counter->mutex);
        uint64_t old = atomic_load_explicit(&counter->value, memory_order_relaxed);
        atomic_store_explicit(&counter->value, old + 1, memory_order_relaxed);
        pthread_mutex_unlock(&counter->mutex);
    }
}
