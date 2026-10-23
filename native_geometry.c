#include <stddef.h>

/* Each worker owns complete rows of C, so no locks are needed. */
void geometry_naive(const double *restrict a, const double *restrict b,
                    double *restrict c, size_t n, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double value = 0.0;
            for (size_t k = 0; k < n; ++k)
                value += a[i * n + k] * b[k * n + j];
            c[i * n + j] = value;
        }
    }
}

static size_t minimum(size_t a, size_t b) { return a < b ? a : b; }

void geometry_tiled(const double *restrict a, const double *restrict b,
                    double *restrict c, size_t n, size_t begin, size_t end,
                    size_t tile) {
    for (size_t i = begin; i < end; ++i)
        for (size_t j = 0; j < n; ++j)
            c[i * n + j] = 0.0;

    for (size_t ii = begin; ii < end; ii += tile) {
        const size_t i_end = minimum(ii + tile, end);
        for (size_t kk = 0; kk < n; kk += tile) {
            const size_t k_end = minimum(kk + tile, n);
            for (size_t jj = 0; jj < n; jj += tile) {
                const size_t j_end = minimum(jj + tile, n);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const double value = a[i * n + k];
                        for (size_t j = jj; j < j_end; ++j)
                            c[i * n + j] += value * b[k * n + j];
                    }
                }
            }
        }
    }
}
