"""BITWISE equivalence of the FILE-BACKED h2e source (SLW99 step 1b).

Step 1(a) blocked S2 over rows of the packed `h2e` (validation/block_invariance.py
proves that is bitwise invariant).  Step 1(b) lets those rows come off DISC
instead of out of a resident (npair x npair) array -- 15.60 GiB at nmo 302.

The file holds exactly the bytes the resident array holds (row-major doubles,
row pq_up at offset pq_up*npair*8), so every row S2 sees is bit-for-bit the row
it saw before and J/K must be BITWISE equal, not merely close.

What this script checks, in one process:
  1. resident source                                  -> reference hashes
  2. file source at several buffer budgets and block sizes -> must all match
  3. the h2e ARGUMENT IS NEVER DEREFERENCED on the file path: the file legs pass
     a 1-element dummy array, so a stray read would segfault or differ
  4. the setter is REVERSIBLE: clearing it returns to the resident source and
     the hashes come back

Run it twice, at OMP_NUM_THREADS=1 and 4, the way block_invariance.py is used:
the file is read by ONE thread inside `omp single` and consumed by all of them,
so the thread count is the interesting axis for that handoff.
"""
import ctypes, hashlib, os, sys, tempfile
import numpy as np

nmo, no = int(os.environ.get("NMO", "24")), int(os.environ.get("NOCC", "7"))
npair = nmo * (nmo + 1) // 2
rng = np.random.default_rng(20260902)          # same seed as block_invariance.py

pair = [(p, q) for p in range(nmo) for q in range(p + 1)]   # DPD-style, larger outer
assert len(pair) == npair
index_map_pq = np.array([x for pq in pair for x in pq], dtype=np.int32)
opair = [(k, l) for k in range(no) for l in range(k + 1)]
index_map_kl = np.array([x for kl in opair for x in kl], dtype=np.int32)

h2e = np.ascontiguousarray(rng.standard_normal((npair, npair)))
h2e = 0.5 * (h2e + h2e.T)                      # (pq|rs) = (rs|pq)
U = np.ascontiguousarray(rng.standard_normal((nmo, nmo)))

# cfunctions.so leaves cblas_*/LAPACKE_* to be resolved at load time from a
# resident MKL; promote libiomp5 AND libmkl_rt to RTLD_GLOBAL first, all of
# them, no early break (the load fails with `undefined symbol: cblas_ddot`).
for cand in (os.path.join(sys.prefix, "lib", "libiomp5.so"),
             os.path.join(sys.prefix, "lib", "libmkl_rt.so.2"),
             os.path.join(sys.prefix, "lib", "libmkl_rt.so.1"),
             os.path.join(sys.prefix, "lib", "libmkl_rt.so"), "libmkl_rt.so.2"):
    try:
        ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        continue
lib = ctypes.cdll.LoadLibrary(os.path.abspath("cfunctions.so"))
ND = np.ctypeslib.ndpointer
lib.full_transformation_macroiteration.argtypes = [
    ND(dtype=np.float64, flags="C_CONTIGUOUS"), ND(dtype=np.float64, flags="C_CONTIGUOUS"),
    ND(dtype=np.float64, flags="C_CONTIGUOUS"), ND(dtype=np.float64, flags="C_CONTIGUOUS"),
    ND(dtype=np.int32, flags="C_CONTIGUOUS"), ND(dtype=np.int32, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int]
lib.full_transformation_macroiteration.restype = None
lib.qed_jk_set_h2e_file.argtypes = [ctypes.c_char_p]
lib.qed_jk_set_h2e_file.restype = None

DUMMY = np.zeros(1)      # what the file legs pass as `h2e`


def run(arr):
    J = np.zeros((no, no, nmo, nmo))
    K = np.zeros((no, no, nmo, nmo))
    lib.full_transformation_macroiteration(U, arr, J, K, index_map_pq,
                                           index_map_kl, nmo, no)
    return J, K


def dig(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def setenv(name, val):
    if val is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(val)


print("nmo %d  n_occupied %d  npair %d  OMP_NUM_THREADS=%s"
      % (nmo, no, npair, os.environ.get("OMP_NUM_THREADS", "unset")))
print("h2e is %.3f MiB resident" % (h2e.nbytes / 1048576.0))

fails = 0
for k in ("QED_JK_BLOCK", "QED_JK_BLOCK_MB", "QED_JK_H2E_FILE"):
    setenv(k, None)
lib.qed_jk_set_h2e_file(None)
J, K = run(h2e)
ref = (dig(J), dig(K))
print("  %-46s J %s  K %s   reference" % ("resident, no knobs", *ref))

with tempfile.NamedTemporaryFile(suffix=".h2e", delete=False) as fh:
    path = fh.name
h2e.tofile(path)
assert os.path.getsize(path) == h2e.nbytes, "packed file is the wrong size"

try:
    # (a) via the SETTER, sweeping the buffer budget and explicit block sizes.
    lib.qed_jk_set_h2e_file(path.encode())
    for mb, blk in ((None, None), (1, None), (64, None), (None, 1), (None, 2),
                    (None, 7), (None, npair - 1), (None, npair), (None, 99999)):
        setenv("QED_JK_BLOCK_MB", mb)
        setenv("QED_JK_BLOCK", blk)
        J, K = run(DUMMY)
        got = (dig(J), dig(K))
        ok = got == ref
        fails += not ok
        print("  %-46s J %s  K %s   %s"
              % ("file (setter) MB=%s BLOCK=%s" % (mb or "default", blk or "default"),
                 *got, "BITWISE MATCH" if ok else "*** DIFFERS ***"))

    # (b) via the ENV VAR, with the setter cleared -- the standalone-C route.
    lib.qed_jk_set_h2e_file(None)
    setenv("QED_JK_BLOCK", None)
    setenv("QED_JK_BLOCK_MB", 1)
    setenv("QED_JK_H2E_FILE", path)
    J, K = run(DUMMY)
    got = (dig(J), dig(K))
    ok = got == ref
    fails += not ok
    print("  %-46s J %s  K %s   %s" % ("file (QED_JK_H2E_FILE env) MB=1", *got,
                                       "BITWISE MATCH" if ok else "*** DIFFERS ***"))

    # (c) the setter must WIN over a stale env var: point the env at a truncated
    # file and the setter at the good one; a correct precedence gives the
    # reference, the wrong one fails the read and exits.
    bad = path + ".short"
    with open(bad, "wb") as fh:
        fh.write(b"\0" * 4096)
    setenv("QED_JK_H2E_FILE", bad)
    lib.qed_jk_set_h2e_file(path.encode())
    J, K = run(DUMMY)
    got = (dig(J), dig(K))
    ok = got == ref
    fails += not ok
    print("  %-46s J %s  K %s   %s" % ("setter overrides a stale env var", *got,
                                       "BITWISE MATCH" if ok else "*** DIFFERS ***"))
    os.unlink(bad)

    # (d) REVERSIBILITY: clear both and the resident source must come back.
    lib.qed_jk_set_h2e_file(None)
    for k in ("QED_JK_BLOCK", "QED_JK_BLOCK_MB", "QED_JK_H2E_FILE"):
        setenv(k, None)
    J, K = run(h2e)
    got = (dig(J), dig(K))
    ok = got == ref
    fails += not ok
    print("  %-46s J %s  K %s   %s" % ("resident again, source cleared", *got,
                                       "BITWISE MATCH" if ok else "*** DIFFERS ***"))
finally:
    os.unlink(path)

print("VERDICT: %s" % ("PASS -- every file leg is bitwise identical"
                       if fails == 0 else "FAIL -- %d leg(s) differ" % fails))
sys.exit(1 if fails else 0)
