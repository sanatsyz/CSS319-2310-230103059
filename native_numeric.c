#include <math.h>
#include <stddef.h>

/* Ordinary serial float64 reductions. No OpenMP, BLAS or hidden workers. */
double numeric_sum(const double *values, size_t begin, size_t end) {
    double result = 0.0;
    for (size_t i = begin; i < end; ++i) result += values[i];
    return result;
}

double numeric_squared_deviation(const double *values, size_t n, double mean) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = values[i] - mean;
        result += d * d;
    }
    return result;
}

void numeric_copy(const double *restrict source, double *restrict dest,
                  size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) dest[i] = source[i];
}

void numeric_trig(const double *restrict source, double *restrict dest,
                  size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        double v = source[i];
        dest[i] = sin(v) * cos(v) + tan(v);
    }
}
