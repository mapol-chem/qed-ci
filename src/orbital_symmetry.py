"""Optional point-group (axial) symmetry constraint for the SA-QED-CASSCF orbital optimization.

WHY THIS EXISTS
---------------
On MgH+ CAS(8,12) our optimizer converges 4.08e-04 Ha BELOW Molpro, and the
reason is not a better minimizer: our converged active space BREAKS the axial
symmetry of the linear molecule, while Molpro's "extra symmetry" machinery
labels every MO sigma/pi/delta and projects out the rest, so Molpro is
constrained to symmetry-pure orbitals and cannot represent our solution.
Handing our orbitals to Molpro with `noextra` confirmed this: it reproduced our
CI energy to 1.7e-10 and STAYED at our point.

This module reproduces Molpro's constraint so the two codes can be compared on
the same footing.

*** IT IS OFF BY DEFAULT AND MUST STAY THAT WAY FOR CAVITY WORK. ***
The Pauli-Fierz coupling carries a DIRECTION (`lambda_vector`).  A cavity mode
polarised off the molecular axis physically breaks the axial symmetry, so
constraining the orbitals to be symmetry-pure would exclude solutions the QED
problem genuinely has.  Use this only as a validation mode against a
symmetry-adapted reference (i.e. at lambda = 0), never as a default.

THE ALGORITHM, IN ONE PARAGRAPH
-------------------------------
For a linear molecule on the z axis, every basis function has a well-defined
signed magnetic quantum number m (sigma = 0, pi_x = +1, pi_y = -1, delta = +-2,
...), and basis functions with different m are exactly orthogonal.  So the AO
overlap S is block diagonal in m.  That means (1) each MO can be assigned the m
it mostly lives in, (2) an MO can be made symmetry-pure by deleting its
coefficients outside that block, and (3) an orbital rotation between two MOs of
DIFFERENT m would break the symmetry, so we simply never create that rotation
variable.  Point (3) is the whole constraint, and it is imposed by leaving those
pairs out of the rotation list -- no solver anywhere has to know about it.

WHAT IT DOES NOT TOUCH
----------------------
Nothing in the microiteration, the internal optimization, the trust-region
solvers or the C backend.  The constraint is expressed entirely as a shorter
`index_map` (the list of allowed orbital-rotation pairs), which every downstream
routine already consumes generically via `range(index_map_size)`.
"""

import numpy as np

# species label -> human name, for the diagnostic report
_NAMES = {0: "sigma", 1: "pi_x", -1: "pi_y", 2: "delta+", -2: "delta-",
          3: "phi+", -3: "phi-", 4: "gamma+", -4: "gamma-"}


def _name(m):
    return _NAMES.get(m, f"m={m:+d}")


def ao_species(basisset):
    """Signed magnetic quantum number m of every AO, in psi4's AO order.

    psi4 orders the 2l+1 pure functions of a shell as m = 0, +1, -1, +2, -2, ...
    so component c maps to m = 0 (c=0), +(c+1)//2 (c odd), -(c//2) (c even>0).
    Cartesian shells have no such labelling and are rejected.
    """
    if not basisset.has_puream():
        raise ValueError(
            "orbital_symmetry: the basis is CARTESIAN. The axial symmetry "
            "labelling is defined for spherical (pure) harmonics only. "
            "Set puream true, or leave QED_ORBITAL_SYMMETRY off."
        )
    species = []
    for ish in range(basisset.nshell()):
        sh = basisset.shell(ish)
        for c in range(sh.nfunction):
            if c == 0:
                species.append(0)
            elif c % 2 == 1:
                species.append((c + 1) // 2)
            else:
                species.append(-(c // 2))
    return np.asarray(species, dtype=int)


def validate_axis(S, species, tol=1e-8):
    """Check S really is block diagonal in m.

    This is the assumption the whole module rests on, and it holds only if the
    molecule is LINEAR and aligned with z. Rather than trusting the geometry we
    test the consequence directly, so a bent or misaligned molecule fails loudly
    here instead of silently producing a wrong constraint.
    """
    worst, where = 0.0, None
    for a in np.unique(species):
        for b in np.unique(species):
            if a >= b:
                continue
            blk = np.abs(S[np.ix_(species == a, species == b)])
            if blk.size and blk.max() > worst:
                worst, where = blk.max(), (a, b)
    if worst > tol:
        raise ValueError(
            "orbital_symmetry: the AO overlap is NOT block diagonal in m "
            f"(largest cross-species element {worst:.3e} between {_name(where[0])} "
            f"and {_name(where[1])}, tolerance {tol:g}).\n"
            "That means the molecule is not linear, or not aligned with the z "
            "axis. This module only supports linear molecules on z. "
            "Leave QED_ORBITAL_SYMMETRY off for this system."
        )
    return worst


def classify(C, S, species):
    """Assign each MO the species it mostly lives in, plus its purity in [0,1].

    Uses Lowdin weights (S^1/2 C), so the weights are a genuine partition of
    each orbital's norm and sum to 1.
    """
    w, V = np.linalg.eigh(S)
    Shalf = V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T
    X = Shalf @ C
    uniq = np.unique(species)
    weights = np.zeros((len(uniq), C.shape[1]))
    for k, m in enumerate(uniq):
        weights[k] = (X[species == m] ** 2).sum(axis=0)
    weights /= weights.sum(axis=0, keepdims=True)
    idx = weights.argmax(axis=0)
    return uniq[idx], weights.max(axis=0), uniq, weights


def symmetrize(C, S, species, labels, drop_tol=1e-6):
    """Project every MO onto its own species, then re-orthonormalize.

    Gram-Schmidt runs over the columns IN ORDER, which is what preserves the
    closed | active | virtual partition: the span of the first k columns after
    the sweep is still the span of the first k projected columns, so orbitals
    never migrate between blocks. MOs of different species are already
    S-orthogonal, so the sweep only ever mixes orbitals within one species --
    exactly the rotations the constraint permits anyway.
    """
    nao, nmo = C.shape
    P = C.copy()
    for j in range(nmo):
        P[species != labels[j], j] = 0.0
    Q = np.zeros_like(P)
    for j in range(nmo):
        v = P[:, j].copy()
        for _ in range(2):                      # twice, for numerical stability
            if j:
                v -= Q[:, :j] @ (Q[:, :j].T @ (S @ v))
        nrm = float(np.sqrt(max(v @ S @ v, 0.0)))
        if nrm < drop_tol:
            raise ValueError(
                f"orbital_symmetry: MO {j+1} collapsed to norm {nrm:.2e} after "
                f"projection onto {_name(labels[j])}. Two orbitals were assigned "
                "the same species but span only one direction in it, so the "
                "symmetry-pure starting guess is rank deficient. This usually "
                "means the input orbitals are too far from symmetric to be "
                "projected; start from RHF orbitals instead."
            )
        Q[:, j] = v / nrm
    return Q


def filtered_index_map(labels, n_in_a, n_occupied, nmo):
    """The rotation list, with symmetry-breaking pairs left out.

    Identical to the loop in helper_PFCI.py:4759-4772 (closed-closed,
    active-active and virtual-virtual pairs are redundant and already excluded)
    with ONE extra filter: a rotation between two MOs of different species would
    mix symmetries, so it is never created.
    """
    pairs = []
    for r in range(nmo):
        for s in range(r + 1, nmo):
            if r < n_in_a and s < n_in_a:
                continue
            if n_in_a <= r < n_occupied and n_in_a <= s < n_occupied:
                continue
            if r >= n_occupied and s >= n_occupied:
                continue
            if labels[r] != labels[s]:          # <-- the entire constraint
                continue
            pairs.append((s, r))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def apply_constraint(C, basisset, S, n_in_a, n_occupied, nmo, verbose=True):
    """Symmetrize the orbitals and return the reduced rotation list.

    Returns (C_sym, index_map, index_map_size, labels).
    """
    species = ao_species(basisset)
    worst = validate_axis(S, species)
    labels, purity, uniq, weights = classify(C, S, species)
    C_sym = symmetrize(C, S, species, labels)
    # re-classify the projected orbitals; purity must now be ~1 by construction
    labels2, purity2, _, _ = classify(C_sym, S, species)
    index_map = filtered_index_map(labels2, n_in_a, n_occupied, nmo)

    if verbose:
        n_full = sum(
            1
            for r in range(nmo)
            for s in range(r + 1, nmo)
            if not (
                (r < n_in_a and s < n_in_a)
                or (n_in_a <= r < n_occupied and n_in_a <= s < n_occupied)
                or (r >= n_occupied and s >= n_occupied)
            )
        )
        print("")
        print(" QED_ORBITAL_SYMMETRY: axial (linear-molecule) constraint ACTIVE")
        print(f"   AO overlap block-diagonality check: max cross-species |S| = {worst:.2e}")
        counts = {int(m): int((labels2 == m).sum()) for m in uniq}
        print("   orbitals per species: "
              + ", ".join(f"{_name(m)}={c}" for m, c in sorted(counts.items(), key=lambda t: -abs(t[0]))))
        moved = np.where(purity < 1.0 - 1e-8)[0]
        if len(moved):
            print("   symmetry contamination removed from the starting orbitals:")
            for j in moved:
                if purity[j] < 0.999999:
                    blk = "closed " if j < n_in_a else ("active " if j < n_occupied else "virtual")
                    print(f"     MO {j+1:3d} ({blk}) -> {_name(labels[j]):7s} "
                          f"purity {purity[j]:.6f} -> {purity2[j]:.6f}")
        print(f"   orbital rotations: {n_full} unconstrained -> "
              f"{len(index_map)} symmetry-allowed "
              f"({n_full - len(index_map)} removed)")
        print("")
    return C_sym, index_map, len(index_map), labels2


def report(labels, labels_before, purity_before, n_in_a, n_occupied,
           n_rot_full, n_rot_sym):
    """Print what the constraint did -- deliberately in Molpro's style, so the
    two codes' output can be compared line for line."""
    print("")
    print(" QED_ORBITAL_SYMMETRY: axial (linear-molecule) constraint ACTIVE")
    counts = {}
    for m in labels:
        counts[int(m)] = counts.get(int(m), 0) + 1
    print("   orbitals per species: " + ", ".join(
        f"{_name(m)}={c}" for m, c in sorted(counts.items(), key=lambda t: -abs(t[0]))))
    dirty = [j for j in range(len(labels)) if purity_before[j] < 0.999999]
    if dirty:
        print("   symmetry contamination removed from the starting orbitals:")
        for j in dirty:
            blk = "closed " if j < n_in_a else ("active " if j < n_occupied else "virtual")
            print(f"     MO {j+1:3d} ({blk}) -> {_name(labels_before[j]):7s} "
                  f"purity {purity_before[j]:.6f} -> 1.000000")
    else:
        print("   starting orbitals were already symmetry pure")
    print(f"   orbital rotations: {n_rot_full} unconstrained -> {n_rot_sym} "
          f"symmetry-allowed ({n_rot_full - n_rot_sym} removed)")
    print("")


def pack_reduced(full, index_map, out):
    """Pack a full (nmo x nmo) matrix into the reduced rotation vector.

    Several routines in helper_PFCI.py re-derive the rotation ordering with
    their own copy of the closed/active/virtual filter loop rather than reading
    `index_map`. That duplication is invisible while the two agree, but the
    symmetry constraint makes `index_map` SHORTER, so those loops overrun.
    This helper reproduces exactly what those loops do -- `out[i] =
    full[larger][smaller]` walked in index_map's own order -- by reading
    index_map directly, so it cannot fall out of step with it.

    index_map[:, 0] is the LARGER MO index and [:, 1] the smaller
    (helper_PFCI.py:11457), and the construction loop at :4759 emits pairs in
    the same order these routines enumerate them, so this is equivalent to the
    original loop whenever no rotation has been filtered out.
    """
    n = len(index_map)
    out[:n] = full[index_map[:, 0], index_map[:, 1]]
    return n
