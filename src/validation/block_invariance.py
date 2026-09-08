"""BITWISE block-invariance of full_transformation_macroiteration.

Step 1 of the SLW99 restructure blocks the S2 quarter transform over rows of
the packed `h2e` so it can later be streamed.  Blocks partition the pq_up range
and each row's unpack+dgemm is unchanged, so the result must be BITWISE equal,
not merely close -- S4 is a pure scatter, not an accumulation (item 349B), so
there is no reordered summation anywhere to excuse a difference.

Calls the C kernel directly with fixed random inputs: no SCF, no optimiser, so
a difference here is the kernel and nothing else.
"""
import ctypes, os, hashlib, numpy as np

nmo, no = int(os.environ.get("NMO", "24")), int(os.environ.get("NOCC", "7"))
npair = nmo*(nmo+1)//2
rng = np.random.default_rng(20260902)

pair = [(p, q) for p in range(nmo) for q in range(p+1)]   # DPD-style, larger outer
assert len(pair) == npair
index_map_pq = np.array([x for pq in pair for x in pq], dtype=np.int32)
opair = [(k, l) for k in range(no) for l in range(k+1)]
index_map_kl = np.array([x for kl in opair for x in kl], dtype=np.int32)

h2e = np.ascontiguousarray(rng.standard_normal((npair, npair)))
h2e = 0.5*(h2e + h2e.T)                      # (pq|rs) = (rs|pq)
U = np.ascontiguousarray(rng.standard_normal((nmo, nmo)))

# cfunctions.so leaves cblas_*/LAPACKE_* to be resolved at load time from a
# resident MKL (helper_PFCI.py:1934-1947 does the same).  Promote libmkl_rt to
# RTLD_GLOBAL first or the dlopen fails with `undefined symbol: cblas_ddot`.
import sys
for cand in (os.path.join(sys.prefix, "lib", "libiomp5.so"),   # OpenMP runtime
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

def run():
    J = np.zeros((no, no, nmo, nmo)); K = np.zeros((no, no, nmo, nmo))
    lib.full_transformation_macroiteration(U, h2e, J, K, index_map_pq,
                                           index_map_kl, nmo, no)
    return J, K

def dig(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

print("nmo %d  n_occupied %d  npair %d" % (nmo, no, npair))
ref = None
for b in ("", "1", "2", "3", "7", str(npair-1), str(npair), "99999"):
    if b: os.environ["QED_JK_BLOCK"] = b
    else: os.environ.pop("QED_JK_BLOCK", None)
    J, K = run()
    hj, hk = dig(J), dig(K)
    if ref is None:
        ref = (hj, hk); verdict = "reference"
    else:
        verdict = "BITWISE MATCH" if (hj, hk) == ref else "*** DIFFERS ***"
    print("  QED_JK_BLOCK=%-8s J %s  K %s   %s" % (b or "unset", hj, hk, verdict))
