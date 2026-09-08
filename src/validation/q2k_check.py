"""QED_JK_Q2K (SLW99 step 2) vs the gather form: J BITWISE, K to ~1 ULP.

WHY THIS BUILDS ITS OWN h2e RATHER THAN COPYING block_invariance.py.
That script fills `h2e` with a random symmetric (npair x npair) matrix, which is
fine for an INVARIANCE test -- both paths make the same assumption, so any
difference is the change under test.  It is NOT fine here.  Its bra pairs are
enumerated DPD-style (larger index outer) while orbital.c's S2 unpacks the KET
triangle with the SMALLER index outer, so the effective 4-index tensor does not
satisfy (pq|rs) = (rs|pq).  The Q2K form relies on exactly that symmetry (it
obtains the q>p half of T3 from the pair swap), so a random-matrix h2e makes it
look catastrophically wrong -- measured: relative 1.09 -- when nothing is.

So the integrals here are built from a factorisation,
    g[p][q][r][s] = sum_x A[p][q][x] A[r][s][x],  A symmetric in (p,q),
which has the full 8-fold symmetry by construction, and then packed with the
bra order index_map_pq uses and the ket order the S2 unpack uses.
"""
import ctypes, hashlib, os, sys
import numpy as np

nmo, no = int(os.environ.get("NMO", "16")), int(os.environ.get("NOCC", "5"))
naux = int(os.environ.get("NAUX", "9"))
npair = nmo * (nmo + 1) // 2
rng = np.random.default_rng(20260902)

# bra order: helper_PFCI's index_map_pq.  Its convention is not the point here --
# what matters is that the SAME order is used to pack the bra axis below.
pair = [(p, q) for p in range(nmo) for q in range(p + 1)]
index_map_pq = np.array([x for pq in pair for x in pq], dtype=np.int32)
opair = [(k, l) for k in range(no) for l in range(k + 1)]
index_map_kl = np.array([x for kl in opair for x in kl], dtype=np.int32)

A = rng.standard_normal((nmo, nmo, naux))
A = 0.5 * (A + A.transpose(1, 0, 2))                    # (pq|.) symmetric in p,q
g = np.einsum("pqx,rsx->pqrs", A, A)                    # full 8-fold symmetry

# ket order: exactly orbital.c's S2 unpack -- `for s2: for r2 >= s2` -- so the
# k-th entry of a row is (r2, s2) with s2 outer.
ket = [(r2, s2) for s2 in range(nmo) for r2 in range(s2, nmo)]
h2e = np.empty((npair, npair))
for i, (p, q) in enumerate(pair):
    for j, (r2, s2) in enumerate(ket):
        h2e[i, j] = g[p, q, r2, s2]
h2e = np.ascontiguousarray(h2e)
U = np.ascontiguousarray(rng.standard_normal((nmo, nmo)))

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
lib.full_transformation_macroiteration.argtypes = (
    [ND(dtype=np.float64, flags="C_CONTIGUOUS")] * 4 +
    [ND(dtype=np.int32, flags="C_CONTIGUOUS")] * 2 + [ctypes.c_int, ctypes.c_int])


def run():
    J = np.zeros((no, no, nmo, nmo))
    K = np.zeros((no, no, nmo, nmo))
    lib.full_transformation_macroiteration(U, h2e, J, K, index_map_pq,
                                           index_map_kl, nmo, no)
    return J, K


dig = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
for _k in ("QED_JK_Q2K", "QED_JK_T1BLOCK", "QED_JK_FUSE_KL"):
    os.environ.pop(_k, None)
J0, K0 = run()
os.environ["QED_JK_Q2K"] = os.environ.get("Q2KMODE", "1")
if os.environ.get("T1BLOCK"):
    os.environ["QED_JK_T1BLOCK"] = os.environ["T1BLOCK"]
if os.environ.get("FUSEKL"):
    os.environ["QED_JK_FUSE_KL"] = os.environ["FUSEKL"]
J1, K1 = run()

# independent reference, straight from g and U -- K[k,l,r,s] = (kr|ls)
Uo = U[:, :no]
# K[k,l,r,s] = (k r | l s): occupied on indices 1 and 3, general on 2 and 4.
Kref = np.einsum("pqrs,pk,qa,rl,sb->klab", g, Uo, U, Uo, U, optimize=True)

sK = np.max(np.abs(K0))
dJ = np.max(np.abs(J1 - J0))
dK = np.max(np.abs(K1 - K0))
dref = np.max(np.abs(K0 - Kref))
print("nmo %d  n_occupied %d  naux %d   (integrals with FULL 8-fold symmetry)"
      % (nmo, no, naux))
print("  J   %s vs %s   %s   max|dJ| %.3e"
      % (dig(J0), dig(J1), "BITWISE" if dig(J0) == dig(J1) else "*** DIFFERS ***", dJ))
print("  K   max|dK| %.3e   scale %.3e   relative %.3e" % (dK, sK, dK / sK))
print("  gather form vs an INDEPENDENT einsum reference: max|d| %.3e (relative %.3e)"
      % (dref, dref / sK))
ok = dig(J0) == dig(J1) and dK / sK < 1e-13
print("  K   %s vs %s   %s"
      % (dig(K0), dig(K1), "BITWISE" if dig(K0) == dig(K1) else "1-ULP class"))
print("  VERDICT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
