#include <stdlib.h>
#include <stdio.h>
#include "orbital.h"
#include <string.h>
#include <math.h>
#include <mkl.h>
#include<omp.h>
#include<time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>


/* --- QED_JK_PROFILE ---------------------------------------------------------
   Env-gated per-section wall-clock breakdown of
   full_transformation_macroiteration (`full JK transformation` in the Python
   log, helper_PFCI.py:5410).  OFF unless QED_JK_PROFILE is set to something
   other than 0/empty; when off it costs one predictable branch per section and
   prints nothing.  Purely diagnostic -- it changes no arithmetic. */
static int jk_prof_enabled(void) {
    static int v = -1;
    if (v < 0) {                       /* called from the serial region only */
        const char* e = getenv("QED_JK_PROFILE");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v;
}
#define JKP(label)                                                            \
    do {                                                                      \
        if (jkp) {                                                            \
            double _t1 = omp_get_wtime();                                     \
            printf("  [JKP] %-24s %9.4f\n", (label), _t1 - _t0);              \
            fflush(stdout);                                                   \
            _t0 = _t1;                                                        \
        }                                                                     \
    } while (0)


/* --- QED_JK_BLOCK ----------------------------------------------------------
   Row-block size for the S2 quarter transform, in PAIRS (rows of the packed
   h2e).  Default 0 = one block = exactly the pre-2026-09-02 behaviour.
   Step 1 of the SLW99 restructure: S2 is the only consumer of `h2e` in
   full_transformation_macroiteration, so blocking it is what lets `h2e` be
   streamed rather than held whole (15.60 GiB at nmo 302).
   `jk_h2e_block` is the seam: today it just returns a pointer into the
   resident array, which is why this commit is a BITWISE no-op.  A file-backed
   source plugs in here without touching the loop. */
/* --- QED_JK_Q2K -----------------------------------------------------------
   Step 2 of the SLW99 restructure: Appendix B's Q2(K).  See the long note at
   the K-side quarter transform for what it changes and what it costs.

   DEFAULT 1 SINCE 2026-09-08 (was 0).  It was measured default-off through
   items 358/359 and then shipped off, so none of the -68.8% peak-RSS chain
   reached a real run.  At nmo 302 this knob is 13.65 -> 13.14 GiB on its own,
   and it is the PRECONDITION for QED_JK_T1BLOCK (13.14 -> 9.74).  K moves by
   ~1 ULP (a different dgemm blocking of the same terms); J stays bitwise.
   QED_JK_Q2K=0 restores the gather form exactly.
   Value 2 is the debug variant (write packed temp3, keep the original S10
   scatter) and is NOT a default. */
static int jk_q2k_enabled(void) {
    const char* e = getenv("QED_JK_Q2K");
    if (!e || !*e) return 1;                     /* default ON */
    if (*e == '0') return 0;
    return atoi(e);
}

/* --- QED_JK_T1BLOCK -------------------------------------------------------
   Step 2 phase 2 (option A).  Number of bra pairs per block for the BLOCKED
   PRODUCER: S2, S3, S4 and the Q2K bra-row pass run inside one loop over
   blocks of pq_up, so `temp1` never exists whole (4.01 GiB at nmo 302) and
   `temp3` shrinks to one block.  0 = off = the resident producer; the default
   is 1024 whenever QED_JK_Q2K is 1 (see jk_t1block_rows below).

   It REQUIRES QED_JK_Q2K=1 and says so rather than silently doing nothing: the
   gather form of S8+S9 reads slabs pidx(p,r) for every r, which no block can
   satisfy.  The price of the route is a SECOND (n_occupied_pair x nmo^2)
   accumulator for the K side -- +0.57 GiB at nmo 302 -- because `temp5` is
   used by the J chain first and cannot also hold K's partial sums.  Net at
   nmo 302: -4.01 (temp1) -0.52 (temp3) +0.57 (temp5K) + one block. */
/* --- QED_JK_FUSE_KL -------------------------------------------------------
   The J chain S5 -> S6 -> S7 and the K chain S11 -> S12 -> S13 are ALREADY
   independent per kl_up: each does two nmo^3 dgemms on one (nmo x nmo) slab
   and scatters that slab into J (or K).  Nothing crosses kl_up.  So `temp6`
   and `temp8` -- one (n_occupied_pair x nmo^2) buffer each, 569 MB apiece at
   nmo 302 -- exist only because the three steps are written as three separate
   loops over the whole range.

   With this on they become PER-THREAD (nmo x nmo) scratch, 0.73 MB each, and
   the three steps run as one fused loop.  BITWISE identical on BOTH J and K:
   the same dgemms with the same arguments in the same per-kl_up order, and an
   omp schedule over an independent index changes none of them.  Default 0. */
static int jk_fuse_kl(void) {
    const char* e = getenv("QED_JK_FUSE_KL");
    return (e && *e && *e != '0') ? 1 : 0;
}

/* DEFAULT 1024 SINCE 2026-09-08 (was 0), but ONLY when q2k == 1.  The default
   has to be conditional: the guard in full_transformation_macroiteration exits
   the process when a block size is asked for without Q2K=1, so an
   unconditional default would turn the documented `QED_JK_Q2K=0` escape hatch
   into a crash.  Under an explicit QED_JK_Q2K=0 (or the =2 debug variant) the
   blocked producer stays off unless the user asks for it by name -- and asking
   for it by name still hits the guard, which is the intended refusal. */
static size_t jk_t1block_rows(int q2k) {
    const char* e = getenv("QED_JK_T1BLOCK");
    if (!e || !*e) return (q2k == 1) ? 1024 : 0; /* default ON with Q2K */
    long long v = atoll(e);
    return (v > 0) ? (size_t) v : 0;
}

static size_t jk_block_size(size_t npair) {
    const char* e = getenv("QED_JK_BLOCK");
    if (!e || !*e) return npair;                 /* one block */
    long long v = atoll(e);
    if (v <= 0) return npair;
    return ((size_t) v > npair) ? npair : (size_t) v;
}

static const double* jk_h2e_block(const double* h2e, size_t blk0,
                                  size_t blk1, size_t npair) {
    (void) blk1; (void) npair;                   /* used by the file source */
    return h2e + blk0 * npair;
}

/* --- step 1(b): the FILE-BACKED h2e source ---------------------------------
   When a path is set, S2 reads its row-block straight off disc instead of out
   of a resident (npair x npair) array -- 15.60 GiB at nmo 302, which is the
   single largest allocation in the whole run.  The file holds exactly what the
   resident array holds: row-major doubles, row `pq_up` at byte offset
   pq_up * npair * 8, both triangles stored (the packer writes it square).

   NO SIGNATURE CHANGE, deliberately.  `full_transformation_macroiteration` is
   also called from the C++ port (cpp_casscf/src/integral_transformer.cpp:69,
   declared in ci_orbital_backend.hpp:85), which compiles THIS file from source;
   adding a parameter would break that build.  The path is instead set once per
   process through `qed_jk_set_h2e_file` (or the QED_JK_H2E_FILE env var, which
   is what a standalone C test uses).  A caller that never sets it -- the C++
   port, every existing python path -- keeps the resident source and is BITWISE
   unchanged.

   The env var and the setter are both honoured; the setter wins when non-empty
   so a python caller cannot be overridden by a stale environment. */
static char jk_h2e_path[4096] = {0};

void qed_jk_set_h2e_file(const char* path) {
    if (!path || !*path) { jk_h2e_path[0] = '\0'; return; }
    size_t n = strlen(path);
    if (n >= sizeof(jk_h2e_path)) n = sizeof(jk_h2e_path) - 1;
    memcpy(jk_h2e_path, path, n);
    jk_h2e_path[n] = '\0';
}

static const char* jk_h2e_file(void) {
    if (jk_h2e_path[0]) return jk_h2e_path;
    const char* e = getenv("QED_JK_H2E_FILE");
    return (e && *e) ? e : NULL;
}

/* Budget for the streaming buffer, in MiB (default 512).  Only consulted when
   the file source is active AND QED_JK_BLOCK was not set explicitly: a file
   source with the default "one block" would allocate the whole tensor, which
   is the very thing this route exists to avoid. */
static size_t jk_block_size_for_file(size_t npair) {
    const char* e = getenv("QED_JK_BLOCK");
    if (e && *e && atoll(e) > 0) return jk_block_size(npair);
    long long mb = 512;
    const char* m = getenv("QED_JK_BLOCK_MB");
    if (m && *m && atoll(m) > 0) mb = atoll(m);
    size_t rows = (size_t) ((mb * 1024LL * 1024LL) / (long long) (npair * sizeof(double)));
    if (rows < 1) rows = 1;
    return (rows > npair) ? npair : rows;
}

/* Read rows [blk0, blk1) into `buf`.  Returns 0 on success.  pread is used so
   the call is position-independent and needs no shared file offset. */
static int jk_h2e_read(int fd, double* buf, size_t blk0, size_t blk1,
                       size_t npair) {
    size_t want = (blk1 - blk0) * npair * sizeof(double);
    off_t off = (off_t) (blk0 * npair * sizeof(double));
    char* dst = (char*) buf;
    while (want > 0) {
        ssize_t got = pread(fd, dst, want, off);
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        if (got == 0) return -1;                 /* short file */
        dst += got; off += got; want -= (size_t) got;
    }
    return 0;
}

void full_transformation_macroiteration(double* U, double* h2e, double* J, double *K, int* index_map_pq, int* index_map_kl, int nmo, int n_occupied) {
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;
    int jkp = jk_prof_enabled();
    const int q2k = jk_q2k_enabled();
    size_t t1blk = jk_t1block_rows(q2k);
    const int fusekl = jk_fuse_kl();
    if (t1blk && q2k != 1) {
        fprintf(stderr, "QED_JK_T1BLOCK needs QED_JK_Q2K=1 -- the gather "
                "form of S8+S9 reads every slab pidx(p,r), which a block "
                "cannot supply.  Refusing rather than silently ignoring "
                "the request.\n");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    double _t0 = jkp ? omp_get_wtime() : 0.0;
    /* h2e_half REMOVED (2026-08-21).  It was nmo(nmo+1)/2 * nmo * nmo doubles
       -- 4.22 GB at c10h8 dims -- holding (p r | q s) purely so the K-side
       quarter transform below could run as a contiguous dgemm.  But that dgemm
       recomputed a tensor the J side had already produced; see the fused
       gather at "S8+S9 K quarter1+2" for the derivation.  Deleting it removes
       the 4.22 GB buffer, the regrouping pass that filled it (30.4% of this
       routine's wall time, and zero flops), and the 41.2 GFlop dgemm it fed. */
    /* With QED_JK_T1BLOCK the producer below fills one BLOCK of bra pairs at
       a time, so temp1 is sized to a block instead of to npair.  temp5/temp6
       and the K accumulator have to exist before that loop runs, so they are
       declared here and allocated at their usual site only if still NULL. */
    const size_t npair_t = nmo_t * (nmo_t + 1) / 2;
    if (t1blk > npair_t) t1blk = npair_t;
    const size_t t1rows = t1blk ? t1blk : npair_t;
    double* temp5 = NULL;
    double* temp6 = NULL;
    double* temp5K = NULL;
    double* temp3 = NULL;
    double* temp8 = NULL;
    double* temp1 = (double*) malloc(t1rows * nmo_t * n_occupied_t * sizeof(double));
    memset(temp1, 0, t1rows * nmo_t * n_occupied_t * sizeof(double));
    JKP("alloc temp1");
    // temp2 removed: it was malloc'd + memset here but every use of it is
    // commented out (the reference loops at ~66 and ~390 that checked temp1
    // against a hand-written contraction), and free(temp2) below was commented
    // out too -- so it was nmo*(nmo+1)/2 * nmo * n_occupied doubles of leak per
    // call, 915 MB at c10h8 dims (nmo=180, n_occupied=39), x10 macroiterations
    // = 9.2 GB of the 33.1 GB peak RSS, plus a 915 MB memset per call.
    // The commented-out reference loops are kept as-is; they declare their own
    // temp2 inside the comment, so reviving one means reviving its alloc.
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    print("%4d%4d%4d\n", pq_up, p , q);
    //}


    /* PACKED h2e.  h2e is (npair, npair), npair = nmo(nmo+1)/2, with BOTH the
       bra and the ket pair folded onto p >= q by the 8-fold symmetry --
       2.12 GB instead of 8.40 at c10h8 dims.  Row `pq_up` is exactly the pair
       index_map_pq[pq_up] names, because the Python packer uses that same
       ordering (smaller index in the OUTER loop), so there is no bra-side
       lookup here at all.
       pidx[a*nmo + b] inverts it for the ket and for step 1's regrouping. */
    size_t npair = nmo_t * (nmo_t + 1) / 2;
    /* File-backed source (step 1b) or resident array.  Everything below this
       point is identical either way -- only where S2's rows come from moves. */
    const char* h2e_file = jk_h2e_file();
    /* With the blocked producer the h2e block and the temp1 block are the
       SAME block -- one loop, one row range -- so the streaming buffer is
       sized to it and QED_JK_BLOCK/_MB do not apply. */
    const size_t jk_block = t1blk ? t1blk
                                  : (h2e_file ? jk_block_size_for_file(npair)
                                              : jk_block_size(npair));
    int    h2e_fd  = -1;
    double* h2e_buf = NULL;
    int     h2e_err = 0;
    if (h2e_file) {
        h2e_fd = open(h2e_file, O_RDONLY);
        if (h2e_fd < 0) {
            fprintf(stderr, "QED_JK_H2E_FILE: cannot open '%s' (%s)\n",
                    h2e_file, strerror(errno));
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
        h2e_buf = (double*) malloc(jk_block * npair * sizeof(double));
        if (!h2e_buf) {
            fprintf(stderr, "QED_JK_H2E_FILE: cannot allocate %zu rows x %zu "
                    "(%.2f GiB) of stream buffer\n", jk_block, npair,
                    (double) (jk_block * npair * sizeof(double)) / 1073741824.0);
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
        if (jkp) printf("  [JKP] %-24s %s (%.2f GiB buffer)\n", "S2 h2e source",
                        h2e_file,
                        (double) (jk_block * npair * sizeof(double)) / 1073741824.0);
    }
    if (jkp) printf("  [JKP] %-24s %9zu rows x %zu blocks\n", "S2 h2e block",
                    jk_block, (npair + jk_block - 1) / jk_block);
    int* pidx = (int*) malloc(nmo_t * nmo_t * sizeof(int));
    for (size_t i = 0; i < npair; i++) {
        size_t p = (size_t) index_map_pq[2*i];
        size_t q = (size_t) index_map_pq[2*i+1];
        pidx[p*nmo_t + q] = (int) i;
        pidx[q*nmo_t + p] = (int) i;
    }
    JKP("pidx build");

    if (t1blk) {
        /* --- QED_JK_T1BLOCK: THE BLOCKED PRODUCER (option A) --------------
           S2, S3, S4 and the Q2K bra-row pass run inside ONE loop over blocks
           of bra pairs, so `temp1` and `temp3` are block-sized instead of
           npair-sized.  What makes this possible is Q2K: its bra-row form
           touches each slab exactly once, where the old gather read slabs
           pidx(p,r) for every r and so needed all of temp1 at once.

           The K side gets its OWN (n_occupied_pair x nmo^2) accumulator.  It
           cannot share `temp5`: the J chain (S5-S7) consumes temp5 only after
           every block has contributed, and the K partial sums have to survive
           across blocks at the same time.  That is the price of option A --
           +0.57 GiB at nmo 302 -- and it buys back 4.01 + 0.52.

           The K pass ACCUMULATES (`+=`), so temp5K is zeroed once above and
           each block adds the r's it happens to contain.  Threads partition by
           p, and each p owns distinct rows of temp5K, so the accumulation
           needs no atomics. */
        const size_t ql   = nmo_t * n_occupied_t;
        const size_t no2  = n_occupied_t * n_occupied_t;
        const size_t nklp = n_occupied_t * (n_occupied_t + 1) / 2;
        const size_t t5sz = nklp * nmo_t * nmo_t;
        temp3  = (double*) malloc(t1rows * no2 * sizeof(double));
        temp5  = (double*) malloc(t5sz * sizeof(double));
        temp6  = fusekl ? NULL : (double*) malloc(t5sz * sizeof(double));
        temp5K = (double*) malloc(t5sz * sizeof(double));
        if (!temp3 || !temp5 || !temp5K || (!fusekl && !temp6)) {
            fprintf(stderr, "QED_JK_T1BLOCK: allocation failed\n");
            fflush(stderr); exit(EXIT_FAILURE);
        }
        memset(temp3,  0, t1rows * no2 * sizeof(double));
        memset(temp5,  0, t5sz * sizeof(double));
        if (temp6) memset(temp6,  0, t5sz * sizeof(double));
        memset(temp5K, 0, t5sz * sizeof(double));
        JKP("alloc blocked producer");
        const size_t gmax = (t1blk < nmo_t) ? t1blk : nmo_t;
        for (size_t blk0 = 0; blk0 < npair; blk0 += t1blk) {
            size_t blk1 = blk0 + t1blk; if (blk1 > npair) blk1 = npair;
            const double* h2e_blk;
            if (h2e_fd >= 0) {
                if (jk_h2e_read(h2e_fd, h2e_buf, blk0, blk1, npair) != 0)
                    h2e_err = 1;
                h2e_blk = h2e_buf;
            } else {
                h2e_blk = jk_h2e_block(h2e, blk0, blk1, npair);
            }
            /* S2' -- the ket unpack + quarter transform, block-local */
            #pragma omp parallel
            {
                double* scr = (double*) malloc(nmo_t * nmo_t * sizeof(double));
                #pragma omp for
                for (size_t pq_up = blk0; pq_up < blk1; pq_up++) {
                    const double* row = h2e_blk + (pq_up - blk0) * npair;
                    for (size_t s2 = 0, kk = 0; s2 < nmo_t; s2++) {
                        for (size_t r2 = s2; r2 < nmo_t; r2++, kk++) {
                            scr[r2*nmo_t + s2] = row[kk];
                            scr[s2*nmo_t + r2] = row[kk];
                        }
                    }
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                nmo_t, n_occupied_t, nmo_t, 1.0, scr, nmo_t,
                                U, nmo_t, 0.0,
                                temp1 + (pq_up - blk0) * ql, n_occupied_t);
                }
                free(scr);
            }
            /* S3' -- J's second quarter transform, block-local */
            #pragma omp parallel for
            for (size_t pq_up = blk0; pq_up < blk1; pq_up++) {
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            n_occupied_t, n_occupied_t, nmo_t, 1.0, U, nmo_t,
                            temp1 + (pq_up - blk0) * ql, n_occupied_t, 0.0,
                            temp3 + (pq_up - blk0) * no2, n_occupied_t);
            }
            /* S4' -- scatter this block's pairs into temp5 (J).  Disjoint
               (p,q) entries, so blocks simply fill different parts. */
            #pragma omp parallel for
            for (size_t kl_up = 0; kl_up < nklp; kl_up++) {
                size_t k = (size_t) index_map_kl[2*kl_up];
                size_t l = (size_t) index_map_kl[2*kl_up+1];
                size_t kl = k * n_occupied_t + l;
                for (size_t pq_up = blk0; pq_up < blk1; pq_up++) {
                    size_t p = (size_t) index_map_pq[2*pq_up];
                    size_t q = (size_t) index_map_pq[2*pq_up+1];
                    double v = temp3[(pq_up - blk0) * no2 + kl];
                    temp5[kl_up * nmo_t * nmo_t + p * nmo_t + q] = v;
                    temp5[kl_up * nmo_t * nmo_t + q * nmo_t + p] = v;
                }
            }
            /* K' -- Q2K's bra-row contraction, restricted to the r's this
               block happens to hold, accumulated into temp5K. */
            #pragma omp parallel
            {
                size_t* rl = (size_t*) malloc(gmax * sizeof(size_t));
                double* Ug = (double*) malloc(gmax * n_occupied_t * sizeof(double));
                double* G  = (double*) malloc(gmax * ql * sizeof(double));
                double* R  = (double*) malloc(n_occupied_t * ql * sizeof(double));
                #pragma omp for
                for (size_t p = 0; p < nmo_t; p++) {
                    size_t nr = 0;
                    for (size_t r = 0; r < nmo_t; r++) {
                        size_t idx = (size_t) pidx[p*nmo_t + r];
                        if (idx >= blk0 && idx < blk1) {
                            rl[nr] = idx - blk0;
                            memcpy(Ug + nr * n_occupied_t, U + r * nmo_t,
                                   n_occupied_t * sizeof(double));
                            nr++;
                        }
                    }
                    if (!nr) continue;
                    for (size_t j = 0; j < nr; j++)
                        memcpy(G + j * ql, temp1 + rl[j] * ql,
                               ql * sizeof(double));
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                n_occupied_t, ql, nr, 1.0,
                                Ug, n_occupied_t, G, ql, 0.0, R, ql);
                    for (size_t kl_up = 0; kl_up < nklp; kl_up++) {
                        size_t k = (size_t) index_map_kl[2*kl_up];
                        size_t l = (size_t) index_map_kl[2*kl_up+1];
                        double* dst = temp5K + kl_up * nmo_t * nmo_t + p * nmo_t;
                        const double* src = R + k * ql + l;
                        for (size_t q = 0; q < nmo_t; q++)
                            dst[q] += src[q * n_occupied_t];
                    }
                }
                free(rl); free(Ug); free(G); free(R);
            }
        }
        if (h2e_fd >= 0) {
            close(h2e_fd); free(h2e_buf); h2e_fd = -1; h2e_buf = NULL;
            if (h2e_err) {
                fprintf(stderr, "QED_JK_H2E_FILE: short or failed read -- "
                        "refusing to continue\n");
                fflush(stderr); exit(EXIT_FAILURE);
            }
        }
        JKP("blocked producer S2-S4+K");
    }

    /* Step 1 (the regrouping pass that built h2e_half) REMOVED -- see the
       h2e_half note above and the fused K-side gather below. */

    if (!t1blk) {
    /* Step 2.  Unpack the ket triangle of row pq_up into a thread-local
       scratch and run the same dgemm.  Measured ~1.4x faster than the
       contiguous full-row form (unpack_probe.c at real shapes): this step is
       bandwidth-bound, the packed read is half the bytes, and the unpack
       stays in L2. */
    #pragma omp parallel
    {
        double* scr = (double*) malloc(nmo_t * nmo_t * sizeof(double));
        /* BLOCKED OVER pq_up (2026-09-02).  Arithmetically identical to the
           flat loop it replaces -- each pq_up still does exactly the same
           unpack and dgemm into the same temp1 offset, and blocks partition
           the range -- so this is a BITWISE no-op with the resident source.
           It exists so `h2e` can be delivered a row-block at a time instead of
           whole (QED_JK_BLOCK / step 1 of the SLW99 work): S2 is the ONLY
           consumer of `h2e` in this routine, verified by grep.
           The block loop is OUTSIDE `#pragma omp for` on purpose: the inner
           loop keeps the same static schedule over its rows, so thread->row
           assignment within a block is unchanged and dgemm blocking is
           untouched.  Block size 0 (default) means "one block", i.e. exactly
           the previous behaviour. */
        for (size_t blk0 = 0; blk0 < npair; blk0 += jk_block) {
            size_t blk1 = blk0 + jk_block; if (blk1 > npair) blk1 = npair;
            const double* h2e_blk;
            if (h2e_fd >= 0) {
                /* ONE thread reads the block; `single` carries an implicit
                   barrier on exit, so every thread sees the filled buffer.
                   The `omp for` below carries one too, so the next round's
                   read cannot start while a thread is still reading the
                   buffer.  No extra synchronisation is needed. */
                #pragma omp single
                {
                    if (jk_h2e_read(h2e_fd, h2e_buf, blk0, blk1, npair) != 0)
                        h2e_err = 1;
                }
                h2e_blk = h2e_buf;
            } else {
                h2e_blk = jk_h2e_block(h2e, blk0, blk1, npair);
            }
            #pragma omp for
            for (size_t pq_up = blk0; pq_up < blk1; pq_up++) {
                const double* row = h2e_blk + (pq_up - blk0) * npair;
                for (size_t s2 = 0, k = 0; s2 < nmo_t; s2++) {
                    for (size_t r2 = s2; r2 < nmo_t; r2++, k++) {
                        scr[r2*nmo_t + s2] = row[k];
                        scr[s2*nmo_t + r2] = row[k];
                    }
                }
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0,
                          scr, nmo_t, U, nmo_t, 0.0,
                          temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t);
            }
        }
        free(scr);
    }
    if (h2e_fd >= 0) {
        close(h2e_fd);
        free(h2e_buf);
        if (h2e_err) {
            fprintf(stderr, "QED_JK_H2E_FILE: short or failed read from '%s' "
                    "-- J and K would be garbage, refusing to continue\n",
                    h2e_file);
            fflush(stderr);
            exit(EXIT_FAILURE);
        }
    }
    JKP("S2 J quarter1");
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //    	    a += h2e[pq * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //            }
    //    	temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //        }
    //    }
    //}
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l], temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //            print("%20.12lf \n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] - temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //        }
    //    }
    //}


    if (!temp3) {
        temp3 = (double*) malloc(t1rows * n_occupied_t * n_occupied_t * sizeof(double));
        memset(temp3, 0, t1rows * n_occupied_t * n_occupied_t * sizeof(double));
    }
    JKP("alloc temp3");
    //double* temp4 = (double*) malloc(nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    //memset(temp4, 0, nmo_t *(nmo_t+1)/2 * n_occupied_t * n_occupied_t * sizeof(double));
    // when matrix is transposed, in cblas_dgemm, m,n,k are the physical size of transposed matrix
    // but lda,ldb are leading dimensions (strides to next row) of original matrices (number of columns)
    if (!t1blk) {
    #pragma omp parallel for
    for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
        //size_t stride = pq_up * 2;
        //size_t p = index_map_pq[stride];
        //size_t q = index_map_pq[stride+1];
	//size_t pq = p * nmo_t + q;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, nmo_t, 1.0, U,
                  nmo_t, temp1+pq_up*nmo_t*n_occupied_t, n_occupied_t, 0.0,
                  temp3+pq_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    JKP("S3 J quarter2");
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t r = 0; r < nmo_t; r++) {
    //                size_t rl = r * n_occupied_t + l;
    //    	    a += temp1[pq_up * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //            }
    //    	temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //        }
    //    }
    //}

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l], temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //            print("%20.12lf \n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]- temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //        }	
    //    }
    //}
    if (!temp5) {
        temp5 = (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
        memset(temp5, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    }
    if (!temp6 && !fusekl) {
        temp6 = (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
        memset(temp6, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    }
    JKP("alloc temp5+temp6");
    
    if (!t1blk) {
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    temp5[kl_up * nmo_t * nmo_t + pq] = temp3[pq_up * n_occupied_t * n_occupied_t + kl];
	    temp5[kl_up * nmo_t * nmo_t + qp] = temp5[kl_up * nmo_t * nmo_t + pq];
	}
    }
    JKP("S4 J scatter t3->t5");
    if (q2k == 1) {
        /* Under Q2K the K side scatters straight into temp5 and never
           touches temp3, and S4 above was the J side's last read of it.
           So it is dead from here: nmo(nmo+1)/2 * n_occupied^2 doubles,
           0.52 GiB at nmo 302.  (Mode 2, the debug path, still writes
           temp3 and must keep it.)  The free() at the end is left as it
           is -- free(NULL) is a no-op. */
        free(temp3);
        temp3 = NULL;
        JKP("free temp3 (Q2K)");
    }
    }
    if (fusekl) {
        /* QED_JK_FUSE_KL: S5 + S6 + S7 in one pass over kl_up, temp6/temp8 as
           per-thread (nmo x nmo) scratch.  See the knob comment at the top. */
        #pragma omp parallel
        {
            double* t6 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
            double* t8 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
            #pragma omp for
            for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t,
                            nmo_t, nmo_t, 1.0, temp5+kl_up*nmo_t*nmo_t, nmo_t,
                            U, nmo_t, 0.0, t6, nmo_t);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t,
                            nmo_t, nmo_t, 1.0, U, nmo_t, t6, nmo_t, 0.0,
                            t8, nmo_t);
                size_t k = (size_t) index_map_kl[2*kl_up];
                size_t l = (size_t) index_map_kl[2*kl_up+1];
                size_t kl = k * n_occupied_t + l;
                size_t lk = l * n_occupied_t + k;
                for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
                    size_t p = (size_t) index_map_pq[2*pq_up];
                    size_t q = (size_t) index_map_pq[2*pq_up+1];
                    size_t pq = p * nmo_t + q;
                    size_t qp = q * nmo_t + p;
                    double v = t8[pq];
                    J[kl * nmo_t * nmo_t + pq] = v;
                    J[kl * nmo_t * nmo_t + qp] = v;
                    J[lk * nmo_t * nmo_t + pq] = v;
                    J[lk * nmo_t * nmo_t + qp] = v;
                }
            }
            free(t6); free(t8);
        }
        JKP("S5-S7 J fused (kl)");
    } else {
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, temp5+kl_up*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*nmo_t, nmo_t);
    }
    JKP("S5 J dgemm nmo^3");
    //double* temp7 = (double*) malloc(n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //memset(temp7, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t q = 0; q < nmo_t; q++) {
    //                size_t pq = p * nmo_t + q;
    //    	    a += temp5[kl_up * nmo_t * nmo_t + pq] * U[q * nmo_t + s];
    //            }
    //    	temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s], temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //            print("%20.12lf \n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] - temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //        }
    //    }
    //}
    temp8 = fusekl ? NULL : (double*) malloc((size_t)n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    if (temp8) memset(temp8, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    JKP("alloc temp8");
    //double* temp9 = (double*) malloc(n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    //memset(temp9, 0, n_occupied_t *(n_occupied_t+1)/2 * nmo_t * nmo_t * sizeof(double));
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp6+kl_up*nmo_t*nmo_t, nmo_t, 0.0,
                  temp8+kl_up*nmo_t*nmo_t, nmo_t);
    }
    JKP("S6 J dgemm nmo^3");
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t p = 0; p < nmo_t; p++) {
    //                size_t ps = p * nmo_t + s;
    //    	    a += temp6[kl_up * nmo_t * nmo_t + ps] * U[p * nmo_t + r];
    //            }
    //    	temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s], temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //            print("%20.12lf \n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] - temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //        }
    //    }
    //}
    //double* temp10 = (double*) malloc(nmo_t * nmo_t * nmo_t * n_occupied_t* sizeof(double));
    //memset(temp10, 0,  nmo_t * nmo_t * nmo_t * n_occupied_t * sizeof(double));
    //double* temp11 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp11, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //double* temp12 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp12, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //double* temp13 = (double*) malloc( nmo_t * nmo_t * n_occupied_t * n_occupied_t* sizeof(double));
    //memset(temp13, 0, nmo_t * nmo_t * n_occupied_t * n_occupied_t * sizeof(double));
    //fflush(stdout);
 

    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    J[kl * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + pq];
	    J[kl * nmo_t * nmo_t + qp] = J[kl * nmo_t * nmo_t + pq];
	    J[lk * nmo_t * nmo_t + pq] = J[kl * nmo_t * nmo_t + pq];
	    J[lk * nmo_t * nmo_t + qp] = J[kl * nmo_t * nmo_t + pq];
	}
    }
    JKP("S7 J scatter->J");
    }

    //print("test full transformation\n");
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t q = 0; q < nmo_t; q++) {
    //        size_t pq = p * nmo_t + q;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    double a = 0.0;
    //                for (size_t s = 0; s < nmo_t; s++) {
    //    	        size_t rs = r * nmo_t + s;
    //                    a += h2e[pq * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //    	    }	
    //                temp10[pq * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t q = 0; q < nmo_t; q++) {
    //        size_t pq = p * nmo_t + q;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    double a = 0.0;
    //                for (size_t r = 0; r < nmo_t; r++) {
    //    	        size_t rl = r * n_occupied_t + l;
    //                    a += temp10[pq * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //    	    }	
    //                temp11[pq * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //for (size_t p = 0; p < nmo_t; p++) {
    //    for (size_t s = 0; s < nmo_t; s++) {
    //        size_t ps = p * nmo_t + s;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    size_t kl = k * n_occupied_t + l;
    //    	    double a = 0.0;
    //                for (size_t q = 0; q < nmo_t; q++) {
    //    	        size_t pq = p * nmo_t + q;
    //                    a += temp11[pq * n_occupied_t * n_occupied_t + kl] * U[q * nmo_t + s];
    //    	    }	
    //                temp12[ps * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}

    //for (size_t r = 0; r < nmo_t; r++) {
    //    for (size_t s = 0; s < nmo_t; s++) {
    //        size_t rs = r * nmo_t + s;
    //        for (size_t k = 0; k < n_occupied_t; k++) {
    //            for (size_t l = 0; l < n_occupied_t; l++) {
    //    	    size_t kl = k * n_occupied_t + l;
    //    	    double a = 0.0;
    //                for (size_t p = 0; p < nmo_t; p++) {
    //    	        size_t ps = p * nmo_t + s;
    //                    a += temp12[ps * n_occupied_t * n_occupied_t + kl] * U[p * nmo_t + r];
    //    	    }	
    //                temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //    	}
    //        }
    //    }
    //}
    //
    //double* temp14 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp14, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp15, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
 


 
    //for (size_t k_p = 0; k_p < n_occupied_t; k_p++) {
    //    for (size_t l_p = 0; l_p < n_occupied_t; l_p++) {
    //        for (size_t r_p = 0; r_p < nmo_t; r_p++) {
    //            for (size_t s_p = 0; s_p < nmo_t; s_p++) {
    //                for (size_t p = 0; p < nmo_t; p++) {
    //                    for (size_t q = 0; q < nmo_t; q++) {
    //                        for (size_t r = 0; r < nmo_t; r++) {
    //                            for (size_t s = 0; s < nmo_t; s++) {
    //    			    temp14[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[p * nmo_t + k_p] * U[q * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * h2e[p * nmo_t * nmo_t * nmo_t + q * nmo_t * nmo_t + r * nmo_t +s];	   
    //                                temp15[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[p * nmo_t + k_p] * U[q * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * h2e[r * nmo_t * nmo_t * nmo_t + p * nmo_t * nmo_t + s * nmo_t +q];	  
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo_t * nmo_t + rs], temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l], J[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J[kl * nmo_t * nmo_t + rs]);
    //    	    //J[kl * nmo_t * nmo_t + rs] = temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l];
    //    	}
    //        }
    //    }
    //}
   



    //build K
    /* The K-side first quarter transform used to be a second 41.2 GFlop dgemm
       over h2e_half; it is gone, fused into the second one below.  NOTE: the
       commented-out reference loops that follow still read h2e_half, which no
       longer exists -- reviving one means reviving its buffer and step 1. */

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //    	    a += h2e_half[pq_up * nmo_t * nmo_t + rs] * U[s * nmo_t + l];
    //            }
    //    	temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] = a;
    //        }
    //    }
    //}
    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l], temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //            print("%20.12lf \n", temp2[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l] - temp1[pq_up * nmo_t * n_occupied_t + r * n_occupied_t + l]);
    //        }
    //    }
    //}

    /* K-side quarter transforms 1 and 2, FUSED (2026-08-21).

       Both quarter transforms in this routine compute the same tensor
           T[a][b][c][l] = sum_s (a b | c s) U[s][l],
       just laid out differently.  The J side (above) produced
           temp1[PQ][r][l] = T[p][q][r][l],  (p,q) = index_map_pq[PQ],
       and the K side wants
           sum_s (p r | q s) U[s][l] = T[p][r][q][l],
       which is ALREADY IN temp1, at temp1[pidx[p*nmo+r]][q][l].  (T is
       symmetric in its first two indices because (ab|cs) = (ba|cs), so pidx
       resolves p<r and p>r alike.)  So the K side needs no dgemm against the
       integrals at all -- only a gather.

       The gather is fused into the second quarter transform rather than
       materialised: for each bra pair it collects nmo runs of n_occupied
       contiguous doubles into a scratch of nmo x n_occupied (56 KB at c10h8
       dims, L2-resident) and hands that straight to the dgemm that was already
       here.  Traffic is ~915 MB of 312-byte gathered runs, replacing a 4.22 GB
       regrouping pass plus a 41.2 GFlop dgemm.

       EXACTNESS: this consumes the J side's numbers instead of recomputing
       them, so J and K can differ from the old path by ~1 ULP where the two
       dgemms blocked differently -- verified at nmo=9/n_occ=4 as 7.1e-15 on a
       scale of 39, i.e. 1.8e-16 relative.  That is far below the 1e-6 noise
       the trust-region solves inject anyway.

       The `#pragma omp parallel for` on the old second quarter transform was
       once missing while its J-side twin at ~:150 had it, which left this
       branch serial (0.325 s of a 1.35 s call); the parallel region below
       keeps it threaded. */
    if (!t1blk) {
    if (q2k) {
        /* --- QED_JK_Q2K: Appendix B's Q2(K), the BRA-ROW form -------------
           The gather below collects, for ONE output pair (p,q), a strided
           column q out of the nmo different temp1 slabs pidx(p,r) -- so every
           output pair touches the whole of temp1, which is exactly why temp1
           (4.01 GiB at nmo 302) has to be resident.

           Reorganised by the BRA index instead: for one p, gather the WHOLE
           slabs pidx(p,r) for every r and contract them all at once,
               R[k][(q,l)] = sum_r U[r][k] * temp1[pidx(p,r)][(q,l)],
           which is T3 for every ordered pair (p,q) at once, indexed [k][l].
           Three consequences:
             * each slab is read as one CONTIGUOUS nmo*n_occupied run rather
               than as nmo strided columns;
             * the result can be scattered STRAIGHT INTO temp5, so the K side
               stops needing temp3 at all (-0.52 GiB, freed below);
             * every slab is touched by exactly one (p, r) pair per pass, which
               is the property a blocked temp1 needs.  The blocking itself is
               NOT done here -- this commit is the algebra alone, so its cost
               can be measured before any I/O is written (the same discipline
               the step-4 note asks for).

           COST, MEASURED NOT ASSUMED: this computes T3 for ALL ordered (p,q)
           where the gather form computed only p>=q, so it does 2x the flops of
           S8+S9.  Whether that is worth -4.53 GiB is a measurement, and the
           knob exists so it can be A/B'd.

           EXACTNESS: same terms, same U, but summed by a different dgemm
           blocking, so K moves by ~1 ULP.  J is untouched and stays bitwise.
           (Item 253 set the same expectation for the last K-side change.) */
        const size_t ql = nmo_t * n_occupied_t;
        #pragma omp parallel
        {
            double* G = (double*) malloc(nmo_t * ql * sizeof(double));
            double* R = (double*) malloc(n_occupied_t * ql * sizeof(double));
            #pragma omp for
            for (size_t p = 0; p < nmo_t; p++) {
                for (size_t r = 0; r < nmo_t; r++) {
                    memcpy(G + r * ql,
                           temp1 + (size_t) pidx[p*nmo_t + r] * ql,
                           ql * sizeof(double));
                }
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            n_occupied_t, ql, nmo_t, 1.0,
                            U, nmo_t, G, ql, 0.0, R, ql);
                /* R[k][q][l] is T3 for the ORDERED pair (p,q); temp5 wants
                   temp5[kl_up][p*nmo + q] with kl_up the packed (k>=l). */
                if (q2k == 2) {
                    /* DEBUG MODE: write the packed temp3 instead and let the
                       original S10 scatter run, to isolate R from the fusion. */
                    for (size_t q = 0; q <= p; q++) {
                        double* d3 = temp3 + (size_t) pidx[p*nmo_t + q]
                                     * n_occupied_t * n_occupied_t;
                        for (size_t k = 0; k < n_occupied_t; k++)
                            for (size_t l = 0; l < n_occupied_t; l++)
                                d3[k*n_occupied_t + l] = R[k*ql + q*n_occupied_t + l];
                    }
                } else {
                for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2;
                     kl_up++) {
                    size_t k = (size_t) index_map_kl[2*kl_up];
                    size_t l = (size_t) index_map_kl[2*kl_up+1];
                    double* dst = temp5 + kl_up * nmo_t * nmo_t + p * nmo_t;
                    const double* src = R + k * ql + l;
                    for (size_t q = 0; q < nmo_t; q++)
                        dst[q] = src[q * n_occupied_t];
                }
                }
            }
            free(G);
            free(R);
        }
        JKP("S8-S10 K bra-row (Q2K)");
    } else {
    #pragma omp parallel
    {
        double* scr = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        #pragma omp for
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t p = (size_t) index_map_pq[2*pq_up];
            size_t q = (size_t) index_map_pq[2*pq_up+1];
            /* scr[r][l] = T[p][r][q][l] = temp1[pidx(p,r)][q][l] */
            for (size_t r = 0; r < nmo_t; r++) {
                memcpy(scr + r * n_occupied_t,
                       temp1 + (size_t) pidx[p*nmo_t + r] * nmo_t * n_occupied_t
                             + q * n_occupied_t,
                       n_occupied_t * sizeof(double));
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, nmo_t, 1.0, U,
                      nmo_t, scr, n_occupied_t, 0.0,
                      temp3+pq_up*n_occupied_t*n_occupied_t, n_occupied_t);
        }
        free(scr);
    }
    JKP("S8+S9 K quarter1+2");
    }
    }

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    size_t stride = pq_up * 2;
    //    size_t p = index_map_pq[stride];
    //    size_t q = index_map_pq[stride+1];
    //    size_t pq = p * nmo_t + q;
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //    	double a = 0.0;
    //            for (size_t r = 0; r < nmo_t; r++) {
    //                size_t rl = r * n_occupied_t + l;
    //    	    a += temp1[pq_up * nmo_t * n_occupied_t + rl] * U[r * nmo_t + k];
    //            }
    //    	temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l] = a;
    //        }
    //    }
    //}

    //for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
    //    for (size_t k = 0; k < n_occupied_t; k++) {
    //        for (size_t l = 0; l < n_occupied_t; l++) {
    //            //print("%20.12lf %20.12lf\n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l], temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //            print("%20.12lf \n", temp4[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]- temp3[pq_up * n_occupied_t * n_occupied_t + k * n_occupied_t + l]);
    //        }	
    //    }
    //}

    if (q2k != 1) {
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    temp5[kl_up * nmo_t * nmo_t + pq] = temp3[pq_up * n_occupied_t * n_occupied_t + kl];
	    temp5[kl_up * nmo_t * nmo_t + qp] = temp3[pq_up * n_occupied_t * n_occupied_t + lk];
	}
    }
    JKP("S10 K scatter t3->t5");
    }   /* with Q2K the scatter is fused into the bra-row pass above */
    if (fusekl) {
        /* QED_JK_FUSE_KL: S11 + S12 + S13, the K twin of the J fusion. */
        #pragma omp parallel
        {
            double* t6 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
            double* t8 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
            #pragma omp for
            for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t,
                            nmo_t, nmo_t, 1.0,
                            (temp5K ? temp5K : temp5)+kl_up*nmo_t*nmo_t, nmo_t,
                            U, nmo_t, 0.0, t6, nmo_t);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t,
                            nmo_t, nmo_t, 1.0, U, nmo_t, t6, nmo_t, 0.0,
                            t8, nmo_t);
                size_t k = (size_t) index_map_kl[2*kl_up];
                size_t l = (size_t) index_map_kl[2*kl_up+1];
                size_t kl = k * n_occupied_t + l;
                size_t lk = l * n_occupied_t + k;
                for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
                    size_t p = (size_t) index_map_pq[2*pq_up];
                    size_t q = (size_t) index_map_pq[2*pq_up+1];
                    size_t pq = p * nmo_t + q;
                    size_t qp = q * nmo_t + p;
                    K[kl * nmo_t * nmo_t + pq] = t8[pq];
                    K[lk * nmo_t * nmo_t + pq] = t8[qp];
                    K[kl * nmo_t * nmo_t + qp] = t8[qp];
                    K[lk * nmo_t * nmo_t + qp] = t8[pq];
                }
            }
            free(t6); free(t8);
        }
        JKP("S11-S13 K fused (kl)");
    } else {
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0,
                  (temp5K ? temp5K : temp5)+kl_up*nmo_t*nmo_t,
                  nmo_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*nmo_t, nmo_t);
    }
    JKP("S11 K dgemm nmo^3");
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t q = 0; q < nmo_t; q++) {
    //                size_t pq = p * nmo_t + q;
    //    	    a += temp5[kl_up * nmo_t * nmo_t + pq] * U[q * nmo_t + s];
    //            }
    //    	temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] = a;
    //        }
    //    }
    //}
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t p = 0; p < nmo_t; p++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s], temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //            print("%20.12lf \n", temp7[kl_up * nmo_t * nmo_t + p * nmo_t + s] - temp6[kl_up * nmo_t * nmo_t + p * nmo_t + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	//size_t stride = kl_up * 2;
        //size_t k = index_map_kl[stride];
        //size_t l = index_map_kl[stride+1];
	//size_t kl = k * n_occupied_t + l;

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp6+kl_up*nmo_t*nmo_t, nmo_t, 0.0,
                  temp8+kl_up*nmo_t*nmo_t, nmo_t);
    }
    JKP("S12 K dgemm nmo^3");
    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //    	double a = 0.0;
    //            for (size_t p = 0; p < nmo_t; p++) {
    //                size_t ps = p * nmo_t + s;
    //    	    a += temp6[kl_up * nmo_t * nmo_t + ps] * U[p * nmo_t + r];
    //            }
    //    	temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] = a;
    //        }
    //    }
    //}

    //for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
    //    size_t stride = kl_up * 2;
    //    size_t k = index_map_kl[stride];
    //    size_t l = index_map_kl[stride+1];
    //    size_t kl = k * n_occupied_t + l;
    //    for (size_t r = 0; r < nmo_t; r++) {
    //        for (size_t s = 0; s < nmo_t; s++) {
    //            //print("%20.12lf %20.12lf\n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s], temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //            print("%20.12lf \n", temp9[kl_up * nmo_t * nmo_t + r * nmo_t + s] - temp8[kl_up * nmo_t * nmo_t + r * nmo_t + s]);
    //        }
    //    }
    //}
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
	size_t stride = kl_up * 2;
        size_t k = index_map_kl[stride];
        size_t l = index_map_kl[stride+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t pq_up = 0; pq_up < nmo_t*(nmo_t+1)/2; pq_up++) {
            size_t stride1 = pq_up * 2;
            size_t p = index_map_pq[stride1];
            size_t q = index_map_pq[stride1+1];
	    size_t pq = p * nmo_t + q;
	    size_t qp = q * nmo_t + p;
	    K[kl * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + pq];
	    K[lk * nmo_t * nmo_t + pq] = temp8[kl_up * nmo_t * nmo_t + qp];
	    K[kl * nmo_t * nmo_t + qp] = K[lk * nmo_t * nmo_t + pq];
	    K[lk * nmo_t * nmo_t + qp] = K[kl * nmo_t * nmo_t + pq];
	}
    }
    JKP("S13 K scatter->K");
    }

    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf %20.12lf %20.12lf\n", temp14[kl * nmo_t * nmo_t + rs], temp13[rs * n_occupied_t * n_occupied_t + k * n_occupied_t + l], J[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf\n", temp15[kl * nmo_t * nmo_t + rs] -  K[kl * nmo_t * nmo_t + rs]);
    //    	}
    //        }
    //    }
    //}
   

    //fflush(stdout);
    free(pidx);
    free(temp1);
    free(temp3);
    //free(temp4);
    free(temp5);
    free(temp5K);
    free(temp6);
    //free(temp7);
    free(temp8);
    JKP("free");
    //free(temp9);
    //free(temp10);
    //free(temp11);
    //free(temp12);
    //free(temp13);
    //free(temp14);
    //free(temp15);
}

void full_transformation_internal_optimization(double* U, double* J, double *K, double* h, double *d_cmo, 
		double* J1, double* K1, double* h1, double *d_cmo1, int* index_map_ab, int* index_map_kl, int nmo, int n_occupied) {
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;

    size_t n_virtual_t = nmo_t - n_occupied_t;
    double* J_half = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(J_half, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp1 = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp1, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp2 = (double*) malloc((size_t)n_virtual_t *(n_virtual_t+1)/2 * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp2, 0, n_virtual_t *(n_virtual_t+1)/2 *  n_occupied_t *  n_occupied_t * sizeof(double));

    //for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
    //    size_t stride = ab_up * 2;
    //    size_t a = index_map_ab[stride];
    //    size_t b = index_map_ab[stride+1];
    //    size_t ab = a * n_virtual_t + b;
    //    print("%4d%4d%4d\n", ab_up, a , b);
    //}


    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];
	//size_t pq = p * nmo_t + q;
	//print("%4d%4d%4d\n", pq_up, p , q);
        for (size_t k = 0; k < n_occupied_t; k++) {
            for (size_t l = 0; l < n_occupied_t; l++) {
		size_t kl = k * n_occupied_t + l;
	        J_half[ab_up * n_occupied_t * n_occupied_t + kl] = J[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t];
	    }
	}
    }
    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, J_half+ab_up*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp2+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride1 = ab_up * 2;
        size_t a = index_map_ab[stride1];
        size_t b = index_map_ab[stride1+1];
        for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
            size_t stride2 = kl_up * 2;
            size_t k = index_map_kl[stride2];
            size_t l = index_map_kl[stride2+1];
	    size_t kl = k * n_occupied_t + l;
	    size_t lk = l * n_occupied_t + k;
	    J1[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[lk * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[kl * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    J1[lk * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	}

    }
    double* temp3 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp3, 0, nmo_t * n_occupied_t *  n_occupied_t *  n_occupied_t * sizeof(double));
    double* temp4 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t *  n_occupied_t * sizeof(double));
    memset(temp4, 0, nmo_t * n_occupied_t *  n_occupied_t *  n_occupied_t * sizeof(double));
    #pragma omp parallel for
    for (size_t p = 0; p < nmo_t; p++) {
        for (size_t m = 0; m < n_occupied_t; m++) {
            size_t pm = p * n_occupied_t + m;
            for (size_t k = 0; k < n_occupied_t; k++) {
                for (size_t l = 0; l < n_occupied_t; l++) {
                    size_t kl = k * n_occupied_t + l;
                    temp3[pm * n_occupied_t * n_occupied_t + kl] = J[kl * nmo_t * nmo_t + p * nmo_t + m];
		}
	    }
	}
    }
    #pragma omp parallel for
    for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, temp3+pm*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp4+pm*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for
    for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp4+pm*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp3+pm*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    
    double* temp5 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    memset(temp5, 0, nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    double* temp6 = (double*) malloc((size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));
    memset(temp6, 0, nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2 * sizeof(double));   
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        size_t stride2 = kl_up * 2;
        size_t k = index_map_kl[stride2];
        size_t l = index_map_kl[stride2+1];
        size_t kl = k * n_occupied_t + l;
        size_t lk = l * n_occupied_t + k;
        for (size_t pm = 0; pm < nmo_t * n_occupied_t; pm++) {
            temp5[kl_up * nmo_t * n_occupied_t + pm] = temp3[pm * n_occupied_t * n_occupied_t + kl];
	}
    }
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, n_occupied_t, 1.0, temp5+kl_up*nmo_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp6+kl_up*nmo_t*n_occupied_t, n_occupied_t);
    }
    cblas_dcopy( (size_t)nmo_t * n_occupied_t * n_occupied_t * (n_occupied_t+1)/2,temp6,1,temp5,1);
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp5+kl_up*nmo_t*n_occupied_t, n_occupied_t, 0.0,
                  temp6+kl_up*nmo_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for
    for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
        size_t stride2 = kl_up * 2;
        size_t k = index_map_kl[stride2];
        size_t l = index_map_kl[stride2+1];
	size_t kl = k * n_occupied_t + l;
	size_t lk = l * n_occupied_t + k;
        for (size_t p = 0; p < nmo_t; p++) {
            for (size_t m = 0; m < n_occupied_t; m++) {
		size_t lm = l * n_occupied_t + m;
		size_t ml = m * n_occupied_t + l;
		size_t km = k * n_occupied_t + m;
		size_t mk = m * n_occupied_t + k;
	        J1[kl * nmo_t * nmo_t + p * nmo_t + m] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[lk * nmo_t * nmo_t + p * nmo_t + m] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[kl * nmo_t * nmo_t + m * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        J1[lk * nmo_t * nmo_t + m * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[lm * nmo_t * nmo_t + k * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[km * nmo_t * nmo_t + l * nmo_t + p] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[ml * nmo_t * nmo_t + p * nmo_t + k] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	        K1[mk * nmo_t * nmo_t + p * nmo_t + l] = temp6[kl_up * nmo_t * n_occupied_t + p * n_occupied_t + m];
	    }
	}
    }
    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];
	//size_t pq = p * nmo_t + q;
	//print("%4d%4d%4d\n", pq_up, p , q);
        for (size_t k = 0; k < n_occupied_t; k++) {
            for (size_t l = 0; l < n_occupied_t; l++) {
		size_t kl = k * n_occupied_t + l;
	        J_half[ab_up * n_occupied_t * n_occupied_t + kl] = K[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t];
	    }
	}
    }
    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, J_half+ab_up*n_occupied_t*n_occupied_t,
                  n_occupied_t, U, nmo_t, 0.0,
                  temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }

    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride = ab_up * 2;
        size_t a = index_map_ab[stride];
        size_t b = index_map_ab[stride+1];

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_occupied_t, n_occupied_t, n_occupied_t, 1.0, U,
                  nmo_t, temp1+ab_up*n_occupied_t*n_occupied_t, n_occupied_t, 0.0,
                  temp2+ab_up*n_occupied_t*n_occupied_t, n_occupied_t);
    }
    #pragma omp parallel for
    for (size_t ab_up = 0; ab_up < n_virtual_t*(n_virtual_t+1)/2; ab_up++) {
	size_t stride1 = ab_up * 2;
        size_t a = index_map_ab[stride1];
        size_t b = index_map_ab[stride1+1];
        for (size_t kl_up = 0; kl_up < n_occupied_t*(n_occupied_t+1)/2; kl_up++) {
            size_t stride2 = kl_up * 2;
            size_t k = index_map_kl[stride2];
            size_t l = index_map_kl[stride2+1];
	    size_t kl = k * n_occupied_t + l;
	    size_t lk = l * n_occupied_t + k;
	    K1[kl * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	    K1[lk * nmo_t * nmo_t + (a + n_occupied_t) * nmo_t + b + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + lk];
	    K1[kl * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + lk];
	    K1[lk * nmo_t * nmo_t + (b + n_occupied_t) * nmo_t + a + n_occupied_t] = temp2[ab_up * n_occupied_t * n_occupied_t + kl];
	}
    }

    //double* temp14 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp14, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
    //double* temp15 = (double*) malloc(n_occupied_t * n_occupied_t * nmo_t * nmo_t * sizeof(double));
    //memset(temp15, 0,n_occupied_t * n_occupied_t *  nmo_t * nmo_t * sizeof(double));
 

 
 
    //for (size_t k_p = 0; k_p < n_occupied_t; k_p++) {
    //    for (size_t l_p = 0; l_p < n_occupied_t; l_p++) {
    //        for (size_t r_p = 0; r_p < nmo_t; r_p++) {
    //            for (size_t s_p = 0; s_p < nmo_t; s_p++) {
    //                for (size_t k = 0; k < n_occupied_t; k++) {
    //                    for (size_t l = 0; l < n_occupied_t; l++) {
    //                        for (size_t r = 0; r < nmo_t; r++) {
    //                            for (size_t s = 0; s < nmo_t; s++) {
    //    			    temp14[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[k * nmo_t + k_p] * U[l * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * J[k * n_occupied_t * nmo_t * nmo_t + l * nmo_t * nmo_t + r * nmo_t +s];	   
    //    			    temp15[k_p * n_occupied_t * nmo_t * nmo_t +l_p *nmo_t * nmo_t + r_p * nmo_t + s_p] +=
    //    			    U[k * nmo_t + k_p] * U[l * nmo_t + l_p] * U[r * nmo_t + r_p] * U[s * nmo_t + s_p] * K[k * n_occupied_t * nmo_t * nmo_t + l * nmo_t * nmo_t + r * nmo_t +s];	   
    //    			}
    //    		    }
    //    		}
    //    	    }
    //    	}
    //        }
    //    }
    //}
    ////for (size_t k = 0; k < n_occupied_t; k++) {
    ////    for (size_t l = 0; l < n_occupied_t; l++) {
    ////        size_t kl = k * n_occupied_t + l;
    ////        for (size_t r = 0; r < nmo_t; r++) {
    ////            for (size_t s = 0; s < nmo_t; s++) {
    ////                size_t rs = r * nmo_t + s;
    ////                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    ////                print("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs],  J1[kl * nmo_t * nmo_t + rs], temp14[kl * nmo_t * nmo_t + rs]-  J1[kl * nmo_t * nmo_t + rs]);
    ////    	}
    ////        }
    ////    }
    ////}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t r = 0; r < nmo_t; r++) {
    //            for (size_t s = 0; s < nmo_t; s++) {
    //                size_t rs = r * nmo_t + s;
    //                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf %20.12lf%20.12lf\n", temp15[kl * nmo_t * nmo_t + rs],  K1[kl * nmo_t * nmo_t + rs], temp15[kl * nmo_t * nmo_t + rs]-  K1[kl * nmo_t * nmo_t + rs]);
    //    	}
    //        }
    //    }
    //}
    //for (size_t k = 0; k < n_occupied_t; k++) {
    //    for (size_t l = 0; l < n_occupied_t; l++) {
    //        size_t kl = k * n_occupied_t + l;
    //        for (size_t a = 0; a < n_virtual_t; a++) {
    //            for (size_t b = 0; b < n_virtual_t; b++) {
    //                size_t ab = (a +n_occupied_t)* nmo_t + b+n_occupied_t;
    //                //print("%20.12lf\n", temp14[kl * nmo_t * nmo_t + rs] -  J1[kl * nmo_t * nmo_t + rs]);
    //                print("%20.12lf %20.12lf%20.12lf\n", temp14[kl * nmo_t * nmo_t + ab],  J1[kl * nmo_t * nmo_t + ab], temp14[kl * nmo_t * nmo_t + ab]-  J1[kl * nmo_t * nmo_t + ab]);
    //    	}
    //        }
    //    }
    //}
    double* temp7 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
    memset(temp7, 0, nmo_t * nmo_t * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, h,
                  nmo_t, U, nmo_t, 0.0,
                  temp7, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp7, nmo_t, 0.0,
                  h1, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, d_cmo,
                  nmo_t, U, nmo_t, 0.0,
                  temp7, nmo_t);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, nmo_t, nmo_t, 1.0, U,
                  nmo_t, temp7, nmo_t, 0.0,
                  d_cmo1, nmo_t);
    //fflush(stdout);
    free(temp1);  
    free(temp2);  
    free(temp3);  
    free(temp4);  
    free(temp5);  
    free(temp6);  
    free(temp7);  
    //free(temp14);  
    //free(temp15); 

}




void build_sigma_reduced(double* U, double* A_tilde, int* index_map, double* G, double* R_reduced, double* sigma_reduced, int    num_states, int    pointer, int    nmo, int    index_map_size, int    n_occupied){
    size_t nmo_t = (size_t) nmo;
    size_t n_occupied_t = (size_t) n_occupied;

    double* R_total = (double*) malloc(num_states * nmo_t * n_occupied_t * sizeof(double));
    memset(R_total, 0, num_states * nmo_t * n_occupied_t * sizeof(double));
    double* sigma_total = (double*) malloc(num_states * nmo_t * n_occupied_t * sizeof(double));
    memset(sigma_total, 0, num_states * nmo_t * n_occupied_t * sizeof(double));
    //print("num state %d\n",num_states); 
    //print("index_map_size %d\n",index_map_size); 
    double* A3 = (double*) malloc(nmo_t * nmo_t * sizeof(double));
    memset(A3, 0, nmo_t * nmo_t * sizeof(double));
    #pragma omp parallel for
    for (size_t p = 0; p < nmo_t; p++) {
        for (size_t q = 0; q < nmo_t; q++) {
	    A3[p * nmo_t + q] = A_tilde[p * nmo_t + q] + A_tilde[q * nmo_t + p];
	}
    }
    
    #pragma omp parallel for
    for (size_t j = 0; j < index_map_size; j++) {
        size_t r = index_map[j * 2 + 0]; 
        size_t k = index_map[j * 2 + 1];
        //print("%4d  %4d %4d\n",j,r,k);	
        for (size_t i = 0; i < num_states; i++) {
            R_total[i * nmo_t * n_occupied_t + r * n_occupied_t + k] = R_reduced[i * (index_map_size + pointer) + j+pointer];
        }
    } 
     
    
    #pragma omp parallel for
    for (size_t i = 0; i < num_states; i++) {
        double* R = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(R, 0, nmo_t * n_occupied_t * sizeof(double));
        double* sigma = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(sigma, 0, nmo_t * n_occupied_t * sizeof(double));
        double* temp1 = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        memset(temp1, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp2  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp2, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp3  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp3, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp4 = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp4 , 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp5  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp5, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp6  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp6, 0, nmo_t * n_occupied_t * sizeof(double));
        //double* temp7  = (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(temp7, 0, nmo_t * n_occupied_t * sizeof(double));
	//double* sigma2= (double*) malloc(nmo_t * n_occupied_t * sizeof(double));
        //memset(sigma2, 0, nmo_t * n_occupied_t * sizeof(double));



        cblas_dcopy(nmo_t * n_occupied_t, R_total + i*nmo_t*n_occupied_t,1,R,1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, U,
                 nmo_t, R, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, -1.0, U,
                 nmo_t, R, n_occupied_t, 1.0,
                 temp1, n_occupied_t);
        
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
	//	double a = 0.0;    
        //        for (size_t s = 0; s < nmo_t; s++) {
        //            a += U[q * nmo_t + s] * R[s * n_occupied_t +l]; 
	//	}
        //            temp2[q * n_occupied_t + l] = a; 
	//    }
	//}	
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t s = 0; s < n_occupied_t; s++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += U[q * nmo_t + l] * R[s * n_occupied_t +l]; 
	//	}
        //            temp2[q * n_occupied_t + s] -= a; 
	//    }
	//}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", temp2[q * n_occupied_t + l],  temp1[q * n_occupied_t + l], temp2[q * n_occupied_t + l]-  temp1[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        //for (size_t p = 0; p < nmo_t; p++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	size_t pk = p * n_occupied_t +k;
	//	double a = 0.0;    
        //        for (size_t q = 0; q < nmo_t; q++) {
        //            for (size_t l = 0; l < n_occupied_t; l++) {
	//	        size_t ql = q * n_occupied_t +l;
        //                a += temp1[q * n_occupied_t + l] * G[ql * nmo_t * n_occupied_t +pk]; 
	//	    }
	//	}
        //            temp3[p * n_occupied_t + k] = a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, nmo_t * n_occupied_t, nmo_t * n_occupied_t, 1.0, temp1,
                 nmo_t * n_occupied_t, G, nmo_t * n_occupied_t, 0.0,
                 sigma, nmo_t * n_occupied_t);
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", temp3[q * n_occupied_t + l],  sigma[q * n_occupied_t + l], temp3[q * n_occupied_t + l]-  sigma[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, 1.0, U,
                 nmo_t, sigma, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
	cblas_dcopy(nmo_t * n_occupied_t, temp1,1,sigma,1);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t p = 0; p < nmo_t; p++) {
        //            a += temp3[p * n_occupied_t + k] * U[p * nmo_t +r]; 
	//	}
        //            temp4[r * n_occupied_t + k] = a; 
	//    }
	//}
	//for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] = temp4[r * n_occupied_t + k]; 
	//    }
	//}
        //for (size_t r = 0; r < n_occupied_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] -= temp4[k * n_occupied_t + r]; 
	//    }
	//}
	for (size_t r = 0; r < n_occupied_t; r++) {
            for (size_t k = 0; k < n_occupied_t; k++) {
                sigma[r * n_occupied_t + k] -= temp1[k * n_occupied_t + r]; 
	    }
	}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nmo_t, n_occupied_t, nmo_t, -0.5, A3,
                 nmo_t, R, n_occupied_t, 0.0,
                 temp1, n_occupied_t);
        cblas_daxpy(nmo_t * n_occupied_t, 1.0, temp1, 1, sigma, 1);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t s = 0; s < nmo_t; s++) {
        //            a += R[s * n_occupied_t + k] * A3[r * nmo_t +s]; 
	//	}
        //            temp5[r * n_occupied_t + k] = a; 
	//    }
	//}
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] -= 0.5 * temp5[r * n_occupied_t + k]; 
	//    }
	//}
        //for (size_t r = 0; r < n_occupied_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
        //        sigma2[r * n_occupied_t + k] += 0.5 * temp5[k * n_occupied_t + r]; 
	//    }
	//}
        

	for (size_t r = 0; r < n_occupied_t; r++) {
            for (size_t k = 0; k < n_occupied_t; k++) {
                sigma[r * n_occupied_t + k] -= temp1[k * n_occupied_t + r]; 
	    }
	}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, 0.5, A3,
                 nmo_t, R, n_occupied_t, 1.0,
                 sigma, n_occupied_t);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += R[k * n_occupied_t + l] * A3[r * nmo_t +l]; 
	//	}
        //            sigma2[r * n_occupied_t + k] += 0.5 * a; 
	//    }
	//}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nmo_t, n_occupied_t, n_occupied_t, -0.5, R,
                 n_occupied_t, A3, nmo_t, 1.0,
                 sigma, n_occupied_t);
        //for (size_t r = 0; r < nmo_t; r++) {
        //    for (size_t k = 0; k < n_occupied_t; k++) {
	//	double a = 0.0;    
        //        for (size_t l = 0; l < n_occupied_t; l++) {
        //            a += R[r * n_occupied_t + l] * A3[k * nmo_t +l]; 
	//	}
        //            sigma2[r * n_occupied_t + k] -= 0.5 * a; 
	//    }
	//}
        //for (size_t q = 0; q < nmo_t; q++) {
        //    for (size_t l = 0; l < n_occupied_t; l++) {
        //        print("%20.12lf %20.12lf%20.12lf\n", sigma[q * n_occupied_t + l],  sigma2[q * n_occupied_t + l], sigma[q * n_occupied_t + l]-  sigma2[q * n_occupied_t + l]);
	//	    
	//    }
	//}
	cblas_dcopy(nmo_t * n_occupied_t, sigma,1,sigma_total + i*nmo_t*n_occupied_t,1);
        free(R);
        free(sigma);
        free(temp1);
        //free(temp2);
        //free(temp3);
        //free(temp4);
        //free(temp5);
        //free(temp6);
        //free(temp7);
        //free(sigma2);
    } 
   
    #pragma omp parallel for
    for (size_t j = 0; j < index_map_size; j++) {
        size_t r = index_map[j * 2 + 0]; 
        size_t k = index_map[j * 2 + 1]; 
        for (size_t i = 0; i < num_states; i++) {
            sigma_reduced[i * (index_map_size + pointer) + j+pointer] = sigma_total[i * nmo_t * n_occupied_t + r * n_occupied_t + k];
        }
    } 
     
free(sigma_total);
free(R_total);
free(A3);


}


