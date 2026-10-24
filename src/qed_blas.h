#ifndef QED_BLAS_H
#define QED_BLAS_H

/* LP64 (32-bit BLAS integers), matching the existing ctypes/C interface.
 * Intel MKL remains the default for existing Linux build commands. */
#ifdef QED_USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#ifdef OPENBLAS_USE64BITINT
#error "QED-CI requires LP64 OpenBLAS, not INTERFACE64/ILP64"
#endif
static inline int qed_blas_get_max_threads(void) {
    return openblas_get_num_threads();
}
static inline void qed_blas_set_num_threads(int n) {
    openblas_set_num_threads(n);
}
#else
#include <mkl.h>
#include <mkl_lapacke.h>
static inline int qed_blas_get_max_threads(void) {
    return mkl_get_max_threads();
}
static inline void qed_blas_set_num_threads(int n) {
    mkl_set_num_threads(n);
}
#endif

_Static_assert(sizeof(lapack_int) == sizeof(int), "QED-CI requires LP64 LAPACK");
#endif
