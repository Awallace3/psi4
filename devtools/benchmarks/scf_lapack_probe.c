/* ELF-only test interposer, never linked into Psi4.
 * Exercise vendor workspace differences and recoverable LAPACK failures.
 * Compile: cc -shared -fPIC scf_lapack_probe.c -ldl -o probe.so
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w,
             double* work, int* lwork, int* iwork, int* liwork, int* info) {
    typedef void (*fn)(char*, char*, int*, double*, int*, double*, double*, int*, int*, int*, int*);
    if (*jobz == 'V' && getenv("PSI_TEST_DSYEVD_MINIMUM")) {
        const int minimum = *n <= 1 ? 1 : 1 + 6 * *n + 2 * *n * *n;
        const int iminimum = *n <= 1 ? 1 : 3 + 5 * *n;
        if (*lwork == -1 || *liwork == -1) {
            /* Legal vendor optimum larger than the documented minimum. */
            work[0] = 2.0 * minimum + 128;
            iwork[0] = iminimum;
            *info = 0;
            return;
        }
        if (*lwork != minimum || *liwork != iminimum) {
            fprintf(stderr, "DSYEVD exceeded reserved minimum: %d/%d versus %d/%d\n",
                    *lwork, *liwork, minimum, iminimum);
            abort();
        }
    }
    if (*lwork != -1 && *liwork != -1 && getenv("PSI_TEST_DSYEVD_FAIL")) {
        /* Corrupt A too: the fallback must restore the original input. */
        if (*n) a[0] = -99999.0;
        *info = 1;
        return;
    }
    fn real = (fn)dlsym(RTLD_NEXT, "dsyevd_");
    if (!real) {
        fputs("LAPACK probe: load Psi4 with RTLD_GLOBAL before interposition\n", stderr);
        abort();
    }
    real(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info);
}

void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info) {
    typedef void (*fn)(char*, int*, double*, int*, int*);
    if (getenv("PSI_TEST_DPOTRF_FAIL")) {
        *info = 1;
        return;
    }
    fn real = (fn)dlsym(RTLD_NEXT, "dpotrf_");
    if (!real) abort();
    real(uplo, n, a, lda, info);
}

void dpocon_(char* uplo, int* n, double* a, int* lda, double* anorm, double* rcond,
             double* work, int* iwork, int* info) {
    typedef void (*fn)(char*, int*, double*, int*, double*, double*, double*, int*, int*);
    fn real = (fn)dlsym(RTLD_NEXT, "dpocon_");
    if (!real) abort();
    real(uplo, n, a, lda, anorm, rcond, work, iwork, info);
    if (getenv("PSI_TEST_DPOCON_SMALL_RCOND")) *rcond = 1e-15;
}
