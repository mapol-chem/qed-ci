// Read psi4 libtrans's transformed two-electron integrals into a numpy array.
//
// WHY THIS FILE EXISTS.  Building self.twoeint via
// mints.mo_eri(Ca,Ca,Ca,Ca) costs 134.75 s at c10h8 dims (nmo=180, cc-pVDZ,
// 24 threads) and is the single largest item in the job.  Almost all of it is
// the AO integrals, which mo_eri rebuilds internally every call: measured
// separately, mints.ao_eri() is 125.65 s while mints.integrals() computes THE
// SAME integrals in 17.24 s.  The whole difference is that ao_eri materialises
// a dense nbf^4 array (8.40 GB, 8x redundant, no screening) whereas
// integrals() writes them to disk for libtrans.  Neither one threads at all --
// ao_eri is 125.32 s at 1 thread and 125.65 s at 24 -- so this is not
// recoverable by throwing cores at it.
//
// Going through libtrans instead costs, at the same dims and 24 threads:
//     mints.integrals()          17.24 s
//     presort_so_tei              3.64 s
//     transform_tei(n,n,n,n)     11.56 s
//     readback (this file)        1.23 s
//                              ------------
//                               33.67 s   vs 134.75 s   =>  -101 s
//
// psi4 exposes no way to do the readback from Python: core.dpdbuf4 has no
// constructor and only {axpy_matrix, coldim, rowdim, zero}, all write-side
// (psi4's own driver/procrouting/diis.py uses it purely as a DIIS storage
// target), and there are no IWL or PSIO-read bindings.  Hence C++.
//
// BUILD (needs only psi4's shipped headers; links nothing -- the symbols are
// resolved at dlopen time out of psi4's already-loaded core.so):
//     g++ -O2 -fPIC -shared -std=c++17 \
//         -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/psi4 \
//         psi4_dpd_shim.cpp -o psi4_dpd_shim.so
//
// THE CALLER MUST HAVE PROMOTED psi4's SYMBOLS TO GLOBAL FIRST.  psi4 ships as
// one monolithic core.cpython-*.so which Python dlopens RTLD_LOCAL, so its
// symbols are invisible to this library.  helper_PFCI does the promotion with
// ctypes.CDLL(psi4.core.__file__, mode=RTLD_GLOBAL); that works even after
// psi4 has been imported normally, which is what lets this be a library-side
// change with no driver script touched.
#include <cstdio>
#include <stdexcept>
#include <cstdlib>

#include "psi4/libdpd/dpd.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

using namespace psi;

extern "C" {

// Returns psi::dpd_default.  Used only to prove the symbols resolved.
int qed_dpd_probe(void) { return dpd_default; }

// IntegralTransform does NOT use DPD instance 0.  It allocates its own
// (it.get_dpd_id() == 1 in practice) while psi::dpd_default is 0, and
// global_dpd_ points at whichever instance is current -- so calling buf4_init
// without switching first reads a DPD that was never initialised for these
// buffers, which segfaults rather than returning an error.
int qed_dpd_set_default(int n) { return dpd_set_default(n); }

// Copy one DPD buf4 into `out`, permuting BOTH pair axes by `perm`:
//
//     out[i*npair + j] = B.matrix[0][perm[i]][perm[j]]
//
// The permutation is not optional.  libtrans stores (AA|AA) with both indices
// packed as [A>=A]+, which runs the LARGER index in the outer loop
// (for p: for q <= p), whereas helper_PFCI's index_map_pq (:3221-3224) runs the
// SMALLER one outer (for q: for p >= q).  Measured on h2o/6-31g: reading with
// the larger-outer convention agrees with mints.mo_eri to 6.2e-15, with the
// smaller-outer convention to 1.04.  Doing the permutation here rather than as
// a numpy fancy-index afterwards avoids a second 2.12 GB pass.
//
// Pass perm = identity to read DPD's own ordering unchanged.
//
// Returns 0 on success, -1 on a shape mismatch, -2 on a psi4 exception.
int qed_dpd_read_buf4(int filenum, int pqnum, int rsnum, const char *label,
                      double *out, long npair, const int *perm) {
    // libtrans closes the DPD file when transform_tei returns.  PSIO_OPEN_OLD
    // keeps the contents; the matching close(..., 1) keeps the file on disk.
    auto psio = _default_psio_lib_;
    const bool was_open = psio->open_check(filenum);
    if (!was_open) psio->open(filenum, PSIO_OPEN_OLD);

    int rc = 0;
    try {
        dpdbuf4 B;
        global_dpd_->buf4_init(&B, filenum, 0 /* irrep */, pqnum, rsnum, pqnum,
                               rsnum, 0 /* no antisymmetrisation */, label);
        global_dpd_->buf4_mat_irrep_init(&B, 0);
        global_dpd_->buf4_mat_irrep_rd(&B, 0);

        const long nrow = B.params->rowtot[0];
        const long ncol = B.params->coltot[0];
        if (nrow != npair || ncol != npair) {
            // This is the failure mode to watch for: asking for the UNPACKED
            // pair type returns rc=0 and an all-zero buffer, because buf4_init
            // happily creates an empty one for a TOC entry that does not
            // exist.  Checking the shape turns that silence into an error.
            std::printf("qed_dpd_read_buf4: shape mismatch -- dpd says "
                        "%ld x %ld, caller expected %ld x %ld.  Wrong pair "
                        "type (packed vs unpacked), wrong file number, or "
                        "wrong label.\n",
                        nrow, ncol, npair, npair);
            rc = -1;
        } else {
#pragma omp parallel for schedule(static)
            for (long i = 0; i < npair; ++i) {
                const double *src = B.matrix[0][perm[i]];
                double *dst = out + i * npair;
                for (long j = 0; j < npair; ++j) dst[j] = src[perm[j]];
            }
        }
        global_dpd_->buf4_mat_irrep_close(&B, 0);
        global_dpd_->buf4_close(&B);
    } catch (const std::exception &e) {
        std::printf("qed_dpd_read_buf4: psi4 threw: %s\n", e.what());
        rc = -2;
    }

    if (!was_open) psio->close(filenum, 1 /* keep */);
    return rc;
}

// Copy one RECTANGULAR DPD buf4 into `out`, row-major, with OPTIONAL
// permutations on each axis:
//
//     out[i*ncol + j] = B.matrix[0][rperm ? rperm[i] : i][cperm ? cperm[j] : j]
//
// This is qed_dpd_read_buf4 generalised for the restricted-space transforms
// that QED_LIBTRANS_OO needs, where the two pair axes have DIFFERENT lengths:
//   J side   transform_tei(occ, occ, all, all)  ->  [o>=o] x [A>=A]
//   K side   transform_tei(occ, all, occ, all)  ->  [o,A]  x [o,A]
// The square reader cannot express either, because it asserts nrow == ncol
// and applies one perm to both axes.
//
// Pass rperm/cperm = NULL to read DPD's own ordering on that axis unchanged.
// The shape check is kept and is the important part: asking for a pair type
// that was never built returns rc=0 and an all-zero buffer from buf4_init,
// so a silent wrong answer is the default failure mode here.  A zero row
// count in particular is what a mis-specified MOSpace produces -- see the
// QED_LIBTRANS_OO note in helper_PFCI.py.
int qed_dpd_read_buf4_rect(int filenum, int pqnum, int rsnum, const char *label,
                           double *out, long nrow_want, long ncol_want,
                           const int *rperm, const int *cperm) {
    auto psio = _default_psio_lib_;
    const bool was_open = psio->open_check(filenum);
    if (!was_open) psio->open(filenum, PSIO_OPEN_OLD);

    int rc = 0;
    try {
        dpdbuf4 B;
        global_dpd_->buf4_init(&B, filenum, 0 /* irrep */, pqnum, rsnum, pqnum,
                               rsnum, 0 /* no antisymmetrisation */, label);
        global_dpd_->buf4_mat_irrep_init(&B, 0);
        global_dpd_->buf4_mat_irrep_rd(&B, 0);

        const long nrow = B.params->rowtot[0];
        const long ncol = B.params->coltot[0];
        if (nrow != nrow_want || ncol != ncol_want) {
            std::printf("qed_dpd_read_buf4_rect: shape mismatch -- dpd says "
                        "%ld x %ld, caller expected %ld x %ld.  Wrong pair "
                        "type (packed vs unpacked), wrong file number, or "
                        "wrong label.%s\n",
                        nrow, ncol, nrow_want, ncol_want,
                        nrow == 0 ? "  (0 rows = the row MOSpace was never "
                                    "populated; see force_occpi)" : "");
            rc = -1;
        } else {
#pragma omp parallel for schedule(static)
            for (long i = 0; i < nrow_want; ++i) {
                const double *src = B.matrix[0][rperm ? rperm[i] : i];
                double *dst = out + i * ncol_want;
                if (cperm) {
                    for (long j = 0; j < ncol_want; ++j) dst[j] = src[cperm[j]];
                } else {
                    for (long j = 0; j < ncol_want; ++j) dst[j] = src[j];
                }
            }
        }
        global_dpd_->buf4_mat_irrep_close(&B, 0);
        global_dpd_->buf4_close(&B);
    } catch (const std::exception &e) {
        std::printf("qed_dpd_read_buf4_rect: psi4 threw: %s\n", e.what());
        rc = -2;
    }

    if (!was_open) psio->close(filenum, 1 /* keep */);
    return rc;
}

// ---------------------------------------------------------------------------
// STREAMING READER (item 354E).  The two readers above call
// buf4_mat_irrep_init + buf4_mat_irrep_rd, which materialise psi4's ENTIRE
// irrep matrix and then copy it into `out` -- so for the time of the copy TWO
// full tensors are alive.  Measured at nmo 302 (item 354B): RssAnon frees the
// old buffer to 0.22 GiB, then climbs linearly to 30.86 GiB over ~10 s, which
// is exactly the log's `readback 9.57 s`, and falls back to 15.82.  One copy is
// 15.6 GiB.  That transient, not the resident tensor, is what sets the run's
// memory requirement at that size.
//
// This trio reads the same buffer a ROW-BLOCK at a time through psi4's own
// buf4_mat_irrep_{init,rd}_block (dpd.h:420-422), so psi4's side is capped at
// one block and the caller can write each block straight out to disc.  The
// COLUMN permutation is applied here, for the same reason the whole-matrix
// reader applies it here: doing it as a numpy fancy-index afterwards would cost
// a second pass over the block.
//
// The ROW permutation is deliberately NOT applied.  `perm` maps OUR row i to a
// DPD row, and DPD can only deliver a CONTIGUOUS row range -- so a block of our
// rows is a scattered set of DPD rows.  The stream therefore hands back DPD
// rows in DPD order and the caller places them (it knows perm, and inverting it
// in numpy is free).  Trying to iterate our order here would turn one
// sequential pass into npair random reads.
//
// The handle is file-static: one stream at a time, which is all the caller
// needs, and it keeps the psio unit open across blocks instead of reopening
// per block.
namespace {
dpdbuf4 g_sb;
int  g_sb_file = -1;
bool g_sb_open = false;
bool g_sb_was_open = false;
long g_sb_nrow = 0, g_sb_ncol = 0, g_sb_cap = 0;
}  // namespace

// Open the buffer and allocate a block of `block_rows` rows.  Verifies the
// shape for the same reason the other readers do: a wrong pair type yields a
// silent all-zero buffer, not an error.  Returns 0, -1 (shape), -2 (threw),
// -3 (a stream is already open).
int qed_dpd_stream_open(int filenum, int pqnum, int rsnum, const char *label,
                        long nrow_want, long ncol_want, long block_rows) {
    if (g_sb_open) {
        std::printf("qed_dpd_stream_open: a stream is already open\n");
        return -3;
    }
    auto psio = _default_psio_lib_;
    g_sb_was_open = psio->open_check(filenum);
    if (!g_sb_was_open) psio->open(filenum, PSIO_OPEN_OLD);
    g_sb_file = filenum;
    try {
        global_dpd_->buf4_init(&g_sb, filenum, 0, pqnum, rsnum, pqnum, rsnum, 0,
                               label);
        g_sb_nrow = g_sb.params->rowtot[0];
        g_sb_ncol = g_sb.params->coltot[0];
        if (g_sb_nrow != nrow_want || g_sb_ncol != ncol_want) {
            std::printf("qed_dpd_stream_open: shape mismatch -- dpd says "
                        "%ld x %ld, caller expected %ld x %ld.\n",
                        g_sb_nrow, g_sb_ncol, nrow_want, ncol_want);
            global_dpd_->buf4_close(&g_sb);
            if (!g_sb_was_open) psio->close(filenum, 1);
            return -1;
        }
        if (block_rows < 1) block_rows = 1;
        if (block_rows > g_sb_nrow) block_rows = g_sb_nrow;
        g_sb_cap = block_rows;
        global_dpd_->buf4_mat_irrep_init_block(&g_sb, 0, (int) g_sb_cap);
    } catch (const std::exception &e) {
        std::printf("qed_dpd_stream_open: psi4 threw: %s\n", e.what());
        if (!g_sb_was_open) psio->close(filenum, 1);
        return -2;
    }
    g_sb_open = true;
    return 0;
}

// Read DPD rows [start_row, start_row + nrows) into `out` (nrows x ncol,
// row-major), applying the column gather:
//     out[k*ncol + j] = B.matrix[0][k][cperm ? cperm[j] : j]
// Returns 0, -2 (threw), -3 (no stream), -4 (bad range).
int qed_dpd_stream_read(double *out, long start_row, long nrows,
                        const int *cperm) {
    if (!g_sb_open) {
        std::printf("qed_dpd_stream_read: no stream open\n");
        return -3;
    }
    if (start_row < 0 || nrows < 1 || nrows > g_sb_cap ||
        start_row + nrows > g_sb_nrow) {
        std::printf("qed_dpd_stream_read: bad range %ld+%ld (nrow %ld, cap "
                    "%ld)\n", start_row, nrows, g_sb_nrow, g_sb_cap);
        return -4;
    }
    try {
        global_dpd_->buf4_mat_irrep_rd_block(&g_sb, 0, (int) start_row,
                                             (int) nrows);
        const long ncol = g_sb_ncol;
#pragma omp parallel for schedule(static)
        for (long k = 0; k < nrows; ++k) {
            const double *src = g_sb.matrix[0][k];
            double *dst = out + k * ncol;
            if (cperm) {
                for (long j = 0; j < ncol; ++j) dst[j] = src[cperm[j]];
            } else {
                for (long j = 0; j < ncol; ++j) dst[j] = src[j];
            }
        }
    } catch (const std::exception &e) {
        std::printf("qed_dpd_stream_read: psi4 threw: %s\n", e.what());
        return -2;
    }
    return 0;
}

int qed_dpd_stream_close(void) {
    if (!g_sb_open) return 0;
    int rc = 0;
    try {
        global_dpd_->buf4_mat_irrep_close_block(&g_sb, 0, (int) g_sb_cap);
        global_dpd_->buf4_close(&g_sb);
    } catch (const std::exception &e) {
        std::printf("qed_dpd_stream_close: psi4 threw: %s\n", e.what());
        rc = -2;
    }
    if (!g_sb_was_open) _default_psio_lib_->close(g_sb_file, 1 /* keep */);
    g_sb_open = false;
    g_sb_file = -1;
    g_sb_cap = g_sb_nrow = g_sb_ncol = 0;
    return rc;
}

// ---------------------------------------------------------------------------
// AO-FEED PROBE (step 3).  Our per-macro kernel is already SLW99 Appendix A;
// what it is NOT fed from is AO integrals -- it rotates a pre-built MO tensor
// instead, which is where the nmo^4/4 storage (53.9 GB at nmo 412) comes from.
// Molpro reads AO integrals per macro and never forms that object.
//
// Before writing any AO->MO arithmetic, measure the FLOOR: how long does it
// take merely to STREAM psi4's AO integral file?  If that already costs more
// than libtrans's presort+transform (251 s at nmo 302), the route is dead and
// no amount of clever contraction saves it.  This function reads the IWL file
// and touches every value, doing no transform at all.
//
// Returns the number of integrals read, or -1 on failure.  `sum_out` receives
// the plain sum of the values, which exists only so the read cannot be
// optimised away and as a cheap fingerprint of the file.
long qed_iwl_scan(int itap, double cutoff, double *sum_out) {
    auto psio = _default_psio_lib_;
    const bool was_open = psio->open_check(itap);
    long n = 0;
    double s = 0.0;
    try {
    // THE UNIT MUST BE CLOSED FIRST.  mints.integrals() leaves unit 33 open
    // at the END of its write, and IWL(oldfile=1) attaches to the already-open
    // unit at that position -- so the read starts PART WAY IN and silently
    // returns a TAIL of the file.  Measured on ch4/6-31G: 6569 of 9549
    // canonical integrals, beginning at (13,6|6,3) with (0,0|0,0) absent
    // entirely, every value it did return correct.  A partial AO list gives a
    // partial quarter transform -- the worst kind of wrong: smooth, plausible
    // and not obviously broken.
    if (psio->open_check(itap)) psio->close(itap, 1 /* keep */);
        IWL iwl(psio.get(), itap, cutoff, /*oldfile=*/1, /*readflag=*/1);
        // The CONSTRUCTOR already fetched buffer 1 (readflag=1), so process
        // first and fetch after -- a leading fetch() silently discards the
        // first buffer, which is where the low-index integrals live.
        while (true) {
            const Value *val = iwl.values();
            const int m = iwl.buffer_count();
            for (int i = 0; i < m; ++i) s += val[i];
            n += m;
            if (iwl.last_buffer()) break;
            iwl.fetch();
        }
        iwl.set_keep_flag(true);
        iwl.close();
    } catch (const std::exception &e) {
        std::printf("qed_iwl_scan: psi4 threw: %s\n", e.what());
        return -1;
    }
    if (!was_open) { /* IWL::close already released it */ }
    if (sum_out) *sum_out = s;
    return n;
}

// ---------------------------------------------------------------------------
// STEP 3, QUARTER 1: the AO-fed first quarter transform.
//
//     half[pair(mu,nu)][lam][i] = sum_sig (mu nu | lam sig) C[sig][i]
//
// with i over the OCCUPIED range (inactive+active, n_occupied) -- the AO
// analogue of orbital.c's S2, and the only step of an AO-fed route whose cost
// is not a dense dgemm.  Everything after it works on progressively smaller
// objects that this codebase already handles well, so THIS is the number that
// decides whether step 3 is worth building: it competes against libtrans's
// 209.2 s transform at nmo 302 (plus its 42.1 s presort, which an AO feed does
// not need at all).
//
// The 8-fold expansion, written out because it is where correctness is easy to
// lose.  One unique (pq|rs) from IWL contributes to FOUR distinct
// (pair, third, fourth) slots -- mu<->nu and lam<->sig are already absorbed by
// the pair packing, so only the r<->s and bra<->ket swaps survive:
//     half[pq][r] += v * C[s]        always
//     half[pq][s] += v * C[r]        unless r == s
//     half[rs][p] += v * C[q]        unless (pq) == (rs)
//     half[rs][q] += v * C[p]        unless (pq) == (rs) or p == q
//
// SINGLE-THREADED ON PURPOSE.  The scatter races on half[pair][lam], and the
// obvious fixes (per-thread copies at 4.3 GB each at nbf 302, or atomics on the
// inner axpy) would both distort the measurement.  libtrans's 209.2 s was
// itself measured at OMP_NUM_THREADS=1, so this is the matched comparison;
// parallelising is a separate question and only worth asking if the serial
// number is already competitive.
//
// `C` is (nbf, nocc) row-major.  `half` is (npair_ao, nbf, nocc) and MUST be
// zeroed by the caller.  Returns the number of AO integrals read, -1 on error.
// Dump the FIRST nmax integrals (labels + values) so a caller can check them
// one by one against an independent reference.  Exists because a wrong TOTAL
// can mean either wrong values or a truncated read, and those need different
// fixes.
long qed_iwl_dump(int itap, double cutoff, long nmax, int *labs, double *vals) {
    auto psio = _default_psio_lib_;
    long n = 0;
    try {
    // THE UNIT MUST BE CLOSED FIRST.  mints.integrals() leaves unit 33 open
    // at the END of its write, and IWL(oldfile=1) attaches to the already-open
    // unit at that position -- so the read starts PART WAY IN and silently
    // returns a TAIL of the file.  Measured on ch4/6-31G: 6569 of 9549
    // canonical integrals, beginning at (13,6|6,3) with (0,0|0,0) absent
    // entirely, every value it did return correct.  A partial AO list gives a
    // partial quarter transform -- the worst kind of wrong: smooth, plausible
    // and not obviously broken.
    if (psio->open_check(itap)) psio->close(itap, 1 /* keep */);
        IWL iwl(psio.get(), itap, cutoff, 1, 1);
        while (true) {
            const Label *lab = iwl.labels();
            const Value *val = iwl.values();
            const int m = iwl.buffer_count();
            for (int t = 0; t < m && n < nmax; ++t, ++n) {
                labs[4 * n + 0] = lab[4 * t + 0];
                labs[4 * n + 1] = lab[4 * t + 1];
                labs[4 * n + 2] = lab[4 * t + 2];
                labs[4 * n + 3] = lab[4 * t + 3];
                vals[n] = val[t];
            }
            if (iwl.last_buffer()) break;
            iwl.fetch();
        }
        iwl.set_keep_flag(true);
        iwl.close();
    } catch (const std::exception &e) {
        std::printf("qed_iwl_dump: psi4 threw: %s\n", e.what());
        return -1;
    }
    return n;
}

long qed_ao_quarter1(int itap, double cutoff, const double *C, long nbf,
                     long nocc, double *half) {
    auto psio = _default_psio_lib_;
    long n = 0;
    try {
    // THE UNIT MUST BE CLOSED FIRST.  mints.integrals() leaves unit 33 open
    // at the END of its write, and IWL(oldfile=1) attaches to the already-open
    // unit at that position -- so the read starts PART WAY IN and silently
    // returns a TAIL of the file.  Measured on ch4/6-31G: 6569 of 9549
    // canonical integrals, beginning at (13,6|6,3) with (0,0|0,0) absent
    // entirely, every value it did return correct.  A partial AO list gives a
    // partial quarter transform -- the worst kind of wrong: smooth, plausible
    // and not obviously broken.
    if (psio->open_check(itap)) psio->close(itap, 1 /* keep */);
        IWL iwl(psio.get(), itap, cutoff, /*oldfile=*/1, /*readflag=*/1);
        while (true) {
            const Label *lab = iwl.labels();
            const Value *val = iwl.values();
            const int m = iwl.buffer_count();
            for (int t = 0; t < m; ++t) {
                const long p = lab[4 * t], q = lab[4 * t + 1];
                const long r = lab[4 * t + 2], s = lab[4 * t + 3];
                const double v = val[t];
                const long pq = (p >= q) ? (p * (p + 1) / 2 + q)
                                         : (q * (q + 1) / 2 + p);
                const long rs = (r >= s) ? (r * (r + 1) / 2 + s)
                                         : (s * (s + 1) / 2 + r);
                double *h = half + (pq * nbf + r) * nocc;
                const double *c = C + s * nocc;
                for (long i = 0; i < nocc; ++i) h[i] += v * c[i];
                if (r != s) {
                    h = half + (pq * nbf + s) * nocc;
                    c = C + r * nocc;
                    for (long i = 0; i < nocc; ++i) h[i] += v * c[i];
                }
                if (pq != rs) {
                    h = half + (rs * nbf + p) * nocc;
                    c = C + q * nocc;
                    for (long i = 0; i < nocc; ++i) h[i] += v * c[i];
                    if (p != q) {
                        h = half + (rs * nbf + q) * nocc;
                        c = C + p * nocc;
                        for (long i = 0; i < nocc; ++i) h[i] += v * c[i];
                    }
                }
                ++n;
            }
            if (iwl.last_buffer()) break;
            iwl.fetch();
        }
        iwl.set_keep_flag(true);
        iwl.close();
    } catch (const std::exception &e) {
        std::printf("qed_ao_quarter1: psi4 threw: %s\n", e.what());
        return -1;
    }
    return n;
}

// ---------------------------------------------------------------------------
// QUARTER 1 FROM THE PRESORTED AO FILE (unit 41, "SO Ints (nn|nn)").
//
// psi4's presort_so_tei() already writes the container Molpro streams: AO
// integrals SORTED BY BRA PAIR, dense npair x npair, implicitly indexed.  We
// have been discarding it after libtrans's transform.
//
// Reading it row-block-wise makes this step structurally IDENTICAL to
// orbital.c's S2 -- unpack the ket triangle of one bra row into scratch, one
// dgemm against C -- with two consequences that matter:
//   * the destination half[pq] is BLOCK-LOCAL, so the scatter that costs the
//     IWL-driven version 2.6x (measured: 4.4 vs 11.5 GFlop/s at identical
//     vector length) does not arise at all;
//   * the file is stored SQUARE, so a bra row is already complete -- there is
//     no 8-fold expansion to do, and no bra<->ket writes.
//
// `half` is (npair, nbf, nocc) indexed in DPD's OWN pair order (p>=q, larger
// index outer), which is what the file uses; the caller reorders if it wants
// helper_PFCI's convention.  cblas is resolved at load time from the MKL psi4
// has already pulled in, exactly as cfunctions.so does.
extern "C" void cblas_dgemm(int, int, int, int, int, int, double, const double *,
                            int, const double *, int, double, double *, int);
#define QED_ROWMAJOR 101
#define QED_NOTRANS  111

long qed_ao_quarter1_presort(int filenum, int pqnum, const char *label,
                             const double *C, long nbf, long nocc, long npair,
                             long block_rows, double *half) {
    auto psio = _default_psio_lib_;
    if (block_rows < 1) block_rows = 1;
    if (block_rows > npair) block_rows = npair;
    double *buf = (double *) malloc((size_t) block_rows * npair * sizeof(double));
    double *scr = (double *) malloc((size_t) nbf * nbf * sizeof(double));
    if (!buf || !scr) { free(buf); free(scr); return -3; }

    // DPD ket pair order: p >= q with p in the OUTER loop.
    int *ridx = (int *) malloc((size_t) nbf * nbf * sizeof(int));
    for (long p = 0, k = 0; p < nbf; ++p)
        for (long q = 0; q <= p; ++q, ++k) {
            ridx[p * nbf + q] = (int) k;
            ridx[q * nbf + p] = (int) k;
        }

    const bool was_open = psio->open_check(filenum);
    if (!was_open) psio->open(filenum, PSIO_OPEN_OLD);
    long nrow = 0;
    try {
        dpdbuf4 B;
        global_dpd_->buf4_init(&B, filenum, 0, pqnum, pqnum, pqnum, pqnum, 0, label);
        if (B.params->rowtot[0] != npair || B.params->coltot[0] != npair) {
            std::printf("qed_ao_quarter1_presort: shape %d x %d, expected %ld x %ld\n",
                        B.params->rowtot[0], B.params->coltot[0], npair, npair);
            global_dpd_->buf4_close(&B);
            free(buf); free(scr); free(ridx);
            if (!was_open) psio->close(filenum, 1);
            return -1;
        }
        global_dpd_->buf4_mat_irrep_init_block(&B, 0, (int) block_rows);
        for (long r0 = 0; r0 < npair; r0 += block_rows) {
            long n = block_rows; if (r0 + n > npair) n = npair - r0;
            global_dpd_->buf4_mat_irrep_rd_block(&B, 0, (int) r0, (int) n);
            for (long k = 0; k < n; ++k) {
                const double *row = B.matrix[0][k];
                for (long p = 0; p < nbf; ++p)
                    for (long q = 0; q <= p; ++q) {
                        const double v = row[ridx[p * nbf + q]];
                        scr[p * nbf + q] = v;
                        scr[q * nbf + p] = v;
                    }
                cblas_dgemm(QED_ROWMAJOR, QED_NOTRANS, QED_NOTRANS,
                            (int) nbf, (int) nocc, (int) nbf, 1.0,
                            scr, (int) nbf, C, (int) nocc, 0.0,
                            half + (r0 + k) * nbf * nocc, (int) nocc);
            }
            nrow += n;
        }
        global_dpd_->buf4_mat_irrep_close_block(&B, 0, (int) block_rows);
        global_dpd_->buf4_close(&B);
    } catch (const std::exception &e) {
        std::printf("qed_ao_quarter1_presort: psi4 threw: %s\n", e.what());
        nrow = -2;
    }
    if (!was_open) psio->close(filenum, 1);
    free(buf); free(scr); free(ridx);
    return nrow;
}

}  // extern "C"
