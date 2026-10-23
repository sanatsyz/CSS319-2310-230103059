/* Elementwise vector kernel, called through ctypes (which releases the GIL). */
#include <stddef.h>

void vector_affine(const double *restrict x, double *restrict y,
                   size_t begin, size_t end, double a, double b) {
    for (size_t i = begin; i < end; ++i) {
        y[i] = a * x[i] + b;
    }
}
