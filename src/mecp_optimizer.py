"""
Quasi-Newton minimum-energy-crossing-point (MECP) optimizer.

Drop-in replacement for the fixed-step steepest descent used by the
scan_conical_*.py drivers:

    atoms.set_positions(pos + effective_step * total_force)

That update is why those scans need 1500 steps. This module keeps the same
Bearpark-Robb-Schlegel (BRS) effective gradient but takes a BFGS step under a
trust radius, in the translation/rotation-projected subspace.

WHAT CHANGES RELATIVE TO THE ORIGINAL DRIVER
--------------------------------------------
1. BFGS instead of steepest descent. The BRS effective gradient is not the
   gradient of a scalar, so strict quasi-Newton convergence theory does not
   apply; using BFGS on it anyway is standard practice (Gaussian's Opt=Conical,
   ORCA) and is what turns hundreds of steps into tens. Powell damping keeps the
   approximate Hessian positive definite when the curvature condition fails,
   which it does whenever the branching-plane vectors rotate appreciably.

2. Consistent units. The original mixes them: f_gap is built from the gap in eV
   while f_seam comes from gradients in Hartree/Bohr, so the gap-closing term is
   over-weighted by ~13.6x relative to the BRS prescription of 2*dE, and the
   step is then added to positions in Angstrom. Here everything is Hartree and
   Bohr internally; Angstrom appears only at the interface.

3. Translation and rotation are projected out of the STEP as well as the force,
   and the BFGS history is kept in that projected subspace, so the Hessian is
   never polluted by the six zero modes.

The caller still supplies the energies, both state gradients and the coupling
vector, so this module performs no quantum chemistry and has no dependency
beyond numpy.

Typical use, replacing the update line in the driver:

    opt = MECPOptimizer(symbols, atoms.get_positions())
    for step in range(max_steps):
        e_p, e_q, grad_p, grad_q, h = evaluate(atoms)      # your CASSCF calls
        info = opt.step(e_p, e_q, grad_p, grad_q, h)
        atoms.set_positions(opt.positions_ang)
        if info["converged"]:
            break
"""

import numpy as np

ANG2BOHR = 1.8897261254578281
HARTREE2EV = 27.211386245988

# Only the elements these decks use; extend as needed.
ATOMIC_MASSES = {
    "H": 1.00782503207, "Li": 7.016003437, "C": 12.0, "N": 14.0030740048, "O": 15.9949146196,
    "F": 18.99840322, "S": 31.97207100, "Cl": 34.96885268,
}


def transrot_projector_mw(coords_bohr, masses, rot_axes=None, extra_modes=None):
    """Symmetric orthogonal projector onto the internal subspace, in MASS-WEIGHTED
    Cartesians. Annihilates exactly the six zero modes of an isolated molecule.

    WHY MASS-WEIGHTED, AND WHY SYMMETRIC. In plain Cartesians the projector for
    displacements is M^-1/2 P_mw M^1/2 while the one for gradients is its
    transpose M^1/2 P_mw M^-1/2: displacements and gradients transform
    oppositely, so a single non-symmetric matrix cannot serve both. Applying the
    displacement projector to a gradient is silently wrong -- it is still
    idempotent and still kills translations, so it passes the obvious checks
    while quietly mixing modes. Working in mass-weighted coordinates throughout
    removes the trap: there is one symmetric projector and one metric.

    rot_axes: None (default) removes all three rotations -- an isolated molecule,
    or one whose orientation is locked by the caller.  A list of axis vectors
    removes ONLY rotations about those axes.  A molecule free to turn in a cavity
    polarized along e keeps the rotations that change its angle to e, which are
    energy-active there; only rotation about e is a symmetry.  Pass rot_axes=[e].
    """
    natom = len(masses)
    sqrtm = np.sqrt(masses)
    com = (coords_bohr * masses[:, None]).sum(0) / masses.sum()
    x = coords_bohr - com

    modes = []
    for axis in range(3):                       # translations
        m = np.zeros((natom, 3))
        m[:, axis] = sqrtm
        modes.append(m.ravel())
    axes = np.eye(3) if rot_axes is None else [np.asarray(a, float) / np.linalg.norm(a) for a in rot_axes]
    for e in axes:                              # infinitesimal rotations
        m = np.cross(x, np.asarray(e)[None, :]) * sqrtm[:, None]
        modes.append(m.ravel())

    if extra_modes:                             # e.g. bond-length constraints (MECPOptimizer fixed_bonds); none by default
        modes.extend(np.asarray(v, float).ravel() for v in extra_modes)
    B = np.array(modes).T
    Q, R = np.linalg.qr(B)
    keep = np.abs(np.diag(R)) > 1e-8            # drops a redundant mode if linear
    Q = Q[:, keep]
    P = np.eye(3 * natom) - Q @ Q.T
    return 0.5 * (P + P.T)                      # symmetric to machine precision


class MECPOptimizer:
    """BFGS/trust-radius optimizer for a minimum-energy crossing point.

    Parameters
    ----------
    symbols : list of str
    positions_ang : (N, 3) array, Angstrom
    trust : initial trust radius, Bohr
    max_trust, min_trust : trust-radius bounds, Bohr
    gap_tol : convergence threshold on |E_q - E_p|, Hartree
    seam_tol : convergence threshold on the projected seam gradient, Hartree/Bohr
    gap_weight : multiplier on the BRS gap-closing term. 2.0 is the textbook
        value; raise it if the gap closes too slowly relative to the seam.
    cone_step : False (default) closes the gap with a Newton step along g
        alone.  True uses BOTH branching vectors: the local two-state model
        [[E_p + grad_p.d, h.d], [h.d, E_q + grad_q.d]] is degenerate when
        g.d = -(E_q - E_p) and h.d = 0, and the step is the minimum-norm
        (mass-weighted) solution of those two equations -- a Newton step on the
        two intersection conditions (Manaa & Yarkony).  For an exactly linear
        cone it reaches the apex in one step however anisotropic the cone; the
        g-only step zig-zags across an anisotropic cone and converges linearly.
        When the seam space is empty (e.g. a diatomic free to rotate in a
        cavity: the branching plane is the whole internal space) the trust
        radius is then driven by |dE| rather than by the noise-level seam norm.
        Pass the coupling as the gap-divided vector (h_vector); only its
        direction enters.
    rot_axes : None (default) projects out all rotations.  For a molecule free
        to rotate in a cavity polarized along e, pass [e] so that only rotation
        about e is removed (see transrot_projector_mw).
    """

    def __init__(self, symbols, positions_ang, trust=0.30, max_trust=1.0,
                 min_trust=1.0e-3, gap_tol=3.7e-6, seam_tol=5.0e-4,
                 gap_weight=2.0, verbose=True, rot_axes=None, cone_step=False,
                 merit="seam", far_gap=1.0e-3, fixed_bonds=None):
        self.symbols = list(symbols)
        self.natom = len(self.symbols)
        self.ndim = 3 * self.natom
        self.x = np.asarray(positions_ang, dtype=float).reshape(-1, 3) * ANG2BOHR
        self.masses = np.array([ATOMIC_MASSES[s] for s in self.symbols])

        self.trust = float(trust)
        self.max_trust = float(max_trust)
        self.min_trust = float(min_trust)
        self.gap_tol = float(gap_tol)
        self.seam_tol = float(seam_tol)
        self.gap_weight = float(gap_weight)
        self.verbose = verbose
        self.rot_axes = rot_axes   # None: project all rotations (default, unchanged)
        self.cone_step = bool(cone_step)   # False: gap step along g only (default, unchanged)
        # merit="seam" (default, unchanged): the trust radius follows the seam-gradient norm.
        # merit="two_phase": while |dE| > far_gap (Eh) follow |dE| instead -- far from the seam the seam norm does
        # not decrease, so the default halves the radius every step and truncates the Newton gap step to nothing
        # (seen starting ethylene from a generic twisted-pyramidalized guess, 2026-09-23).  Near the seam both
        # merits are identical.  The converged point obeys the same gap_tol / seam_tol either way.
        if merit not in ("seam", "two_phase"):
            raise ValueError(f"merit must be 'seam' or 'two_phase', not {merit!r}")
        self.merit = merit
        self.far_gap = float(far_gap)
        self._phase = "seam"
        # fixed_bonds=None (default, unchanged).  A list of (i, j, r_angstrom) holds those bond lengths fixed: the
        # mass-weighted gradient of each bond length is projected out of every step (as translations/rotations are),
        # and the length is reset exactly after each step (atom j moved along the bond) to remove the second-order drift.
        # Used for constrained MECP scans along a soft seam coordinate (2026-09-24).
        self.fixed_bonds = [(int(i), int(j), float(r) * ANG2BOHR) for i, j, r in fixed_bonds] if fixed_bonds else None
        self._enforce_bonds()

        self.B = None            # approximate Hessian, internal subspace
        self._prev_g = None      # previous effective gradient
        self._prev_s = None      # previous step
        self._prev_norm = None   # previous |g_eff|
        self.nstep = 0
        self.n_shrink = 0
        self._last_u_h = None    # branching-plane continuation (see below)
        self._x_prev = None      # geometry before the most recent accepted step
        self.n_reject = 0

    def rotate_frame(self, R, center_bohr=None):
        """The caller rigidly rotated the molecule by the 3x3 matrix R (about center_bohr; default the centroid): rotate the
        quasi-Newton history with it, so the next BFGS update compares vectors written in the same axes.

        WHY: without this, y = g_seam(new frame) - g_seam(old frame) carries a spurious (R - I) g term, and B is a matrix in
        the old axes applied to vectors in the new ones.  The trans/rot projector does not remove this: it removes rigid
        MOTIONS of the molecule, not a change of the axes the vectors are written in.  Mass weighting commutes with the
        per-atom rotation (the three components of an atom share one mass), so T = I_N (x) R acts directly on the
        mass-weighted vectors.  Opt-in: nothing calls this unless the driver asks (MECP_ROTATE_HISTORY=1); 2026-09-24.
        """
        R = np.asarray(R, dtype=float)
        T = np.kron(np.eye(self.natom), R)
        c = self.x.mean(0) if center_bohr is None else np.asarray(center_bohr, float)
        self.x = (self.x - c) @ R.T + c
        if self._x_prev is not None:
            self._x_prev = (self._x_prev - c) @ R.T + c
        if self.B is not None:
            self.B = T @ self.B @ T.T
        if self._prev_s is not None:
            self._prev_s = T @ self._prev_s
        if self._prev_g is not None:
            self._prev_g = T @ self._prev_g
        if self._last_u_h is not None:
            u = np.asarray(self._last_u_h); self._last_u_h = (T @ u.ravel()).reshape(u.shape)
        self._enforce_bonds()

    def _constraint_modes(self):
        if not self.fixed_bonds:
            return None
        out = []
        for i, j, _ in self.fixed_bonds:
            u = self.x[j] - self.x[i]; u = u / np.linalg.norm(u)
            v = np.zeros_like(self.x)
            v[i] = -u / np.sqrt(self.masses[i]); v[j] = u / np.sqrt(self.masses[j])   # d r_ij in mass-weighted coordinates
            out.append(v.ravel())
        return out

    def _enforce_bonds(self):
        if not self.fixed_bonds:
            return
        for i, j, r in self.fixed_bonds:
            d = self.x[j] - self.x[i]
            self.x[j] = self.x[i] + d * (r / np.linalg.norm(d))

    def reject(self):
        """Undo the last step: the electronic structure failed at the new
        geometry, so that point carries no usable information.

        Restores the previous geometry, halves the trust radius and CLEARS the
        BFGS history. Clearing matters: the curvature pair (s, y) from a step
        whose gradient could not be evaluated is meaningless, and feeding a
        stale pair back in is how a quasi-Newton optimizer poisons its own
        Hessian after a single failed point.
        """
        if self._x_prev is not None:
            self.x = self._x_prev.copy()
        self.trust = max(self.trust * 0.5, self.min_trust)
        self.B = None
        self._prev_g = None
        self._prev_s = None
        self._prev_norm = None
        self.n_reject += 1
        return self.positions_ang

    # -- interface -----------------------------------------------------------

    @property
    def positions_ang(self):
        return self.x / ANG2BOHR

    def effective_gradient(self, e_p, e_q, grad_p, grad_q, h):
        """The BRS effective gradient and its pieces, all Hartree/Bohr.

        f_gap closes the gap along the gradient-difference direction; f_seam is
        the mean gradient with the branching plane (g, h) projected out.
        """
        grad_p = np.asarray(grad_p, dtype=float).reshape(-1, 3)
        grad_q = np.asarray(grad_q, dtype=float).reshape(-1, 3)
        h = np.asarray(h, dtype=float).reshape(-1, 3)

        dE = e_q - e_p
        g = grad_q - grad_p
        sigma = 0.5 * (grad_p + grad_q)

        gnorm = np.linalg.norm(g)
        if gnorm < 1e-12:
            raise ValueError("gradient difference vanished; states are not distinct")
        u_g = g / gnorm

        # BRANCHING-PLANE CONTINUATION.
        # The driver passes h0 = d^{pq} * (E_q - E_p), which VANISHES as the gap
        # closes -- so the second branching direction becomes numerically
        # undefined exactly at convergence, and f_seam then retains a component
        # it should have projected out. Left alone this makes the optimizer
        # declare convergence early, at a geometry displaced along the missing
        # direction; the faster the gap closes, the worse it gets.
        #
        # The fix (as in Maeda/Morokuma's updated branching plane) is to carry
        # the previous step's u_h forward whenever the incoming coupling is too
        # small to define a direction, re-orthogonalising it against the current
        # u_g. Passing the gap-divided coupling (nuclear_grad's `h_vector`)
        # instead of `h0` avoids the problem at source and is preferred.
        h_perp = h - np.dot(h.ravel(), u_g.ravel()) * u_g
        hn = np.linalg.norm(h_perp)
        h_scale = max(np.linalg.norm(h), 1e-30)
        if hn > 1e-10 and hn / h_scale > 1e-6:
            u_h = h_perp / hn
            self._last_u_h = u_h.copy()
        elif getattr(self, "_last_u_h", None) is not None:
            v = self._last_u_h - np.dot(self._last_u_h.ravel(), u_g.ravel()) * u_g
            vn = np.linalg.norm(v)
            u_h = v / vn if vn > 1e-10 else np.zeros_like(u_g)
        else:
            u_h = np.zeros_like(u_g)

        f_gap = self.gap_weight * dE * u_g
        f_seam = (sigma
                  - np.dot(sigma.ravel(), u_g.ravel()) * u_g
                  - np.dot(sigma.ravel(), u_h.ravel()) * u_h)
        cos_phi = abs(np.dot(u_g.ravel(), h.ravel()) / max(np.linalg.norm(h), 1e-30))
        return f_gap + f_seam, f_gap, f_seam, u_g, u_h, cos_phi, gnorm

    def step(self, e_p, e_q, grad_p, grad_q, h):
        """Take one optimizer step. Energies in Hartree, gradients Hartree/Bohr.

        THE STEP IS SPLIT rather than handing the whole BRS gradient to BFGS,
        because its two components have different local structure:

          * along u_g the gap is LOCALLY LINEAR with slope |grad_q - grad_p|, so
            its exact Newton step is dE/|g|, known in closed form. Handing the
            BRS force 2*dE to a quasi-Newton method instead makes the step
            depend on whatever the approximate Hessian holds -- with the usual
            diagonal guess of 0.5 it comes out as 4*dE. Measured on ethylene at
            lambda=0.4: |g| ~ 0.15, so the required step is 6.7*dE against the
            BRS/BFGS 4*dE. Under-corrected ~1.7x, the seam term then opens the
            gap faster than that closes it, and the gap grows monotonically
            (0.00068 -> 0.0024 -> 0.0117 -> 0.0130 -> 0.0137 eV over four steps).

          * within the seam the mean energy is genuinely quadratic, which is
            exactly where BFGS belongs.

        So: exact Newton along u_g, BFGS in the orthogonal complement of the
        branching plane, both in mass-weighted coordinates.
        """
        g_eff, f_gap, f_seam, u_g, u_h, cos_phi, gdiff_norm = self.effective_gradient(
            e_p, e_q, grad_p, grad_q, h)

        w = np.repeat(np.sqrt(self.masses), 3)
        P = transrot_projector_mw(self.x, self.masses, self.rot_axes, self._constraint_modes())
        dE = e_q - e_p
        gp = np.asarray(grad_p, dtype=float).ravel()
        gq = np.asarray(grad_q, dtype=float).ravel()

        # branching-plane directions, mass-weighted and re-orthonormalised
        ug = P @ (u_g.ravel() / w)
        n = np.linalg.norm(ug)
        ug = ug / n if n > 1e-14 else ug
        uh = P @ (u_h.ravel() / w)
        if np.linalg.norm(uh) > 1e-14:
            uh = uh - (uh @ ug) * ug
            n = np.linalg.norm(uh)
            uh = uh / n if n > 1e-14 else np.zeros_like(ug)
        else:
            uh = np.zeros_like(ug)

        # seam subspace: internal, minus the branching plane
        Q = P - np.outer(ug, ug)
        if np.linalg.norm(uh) > 1e-14:
            Q = Q - np.outer(uh, uh)
        Q = 0.5 * (Q + Q.T)

        sigma_mw = (0.5 * (gp + gq)) / w
        g_seam = Q @ sigma_mw                 # drives the STEP (mass-weighted)
        seam_norm_mw = float(np.linalg.norm(g_seam))
        gnorm = float(np.linalg.norm(P @ (g_eff.ravel() / w)))

        # CONVERGENCE IS TESTED ON THE DRIVER'S OWN DEFINITION, not on the
        # quantity the step is built from. The seam norm is projector-dependent:
        # scan_conical_*.py projects the CARTESIAN branching plane out of sigma
        # and then removes translation/rotation, while the step above works in
        # mass-weighted coordinates, where the branching plane is a different
        # subspace. At the lambda=0.4 MECP the two norms are 7.28e-04 and
        # 1.76e-04 -- a factor of 4 at the same geometry. Testing convergence on
        # the mass-weighted number would quietly move the goalpost and make step
        # counts incomparable with the published runs, so seam_tol keeps its
        # original meaning and seam_norm_mw is reported alongside it.
        seam_cart = w * (P @ (f_seam.ravel() / w))     # gradient trans/rot projector
        seam_norm = float(np.linalg.norm(seam_cart))
        converged = (abs(dE) < self.gap_tol) and (seam_norm < self.seam_tol)
        if self.fixed_bonds:
            # With a fixed bond the Lagrange reaction force mu*grad(r) survives the CARTESIAN branching-plane projection of
            # f_seam (a different metric from the mass-weighted constraint mode), so seam_norm cannot reach seam_tol at a
            # constrained stationary point (measured: 5.4e-3 with seam_norm_mw 1e-8).  Constrained runs therefore test the
            # mass-weighted seam norm, which is consistent with the constraint.  Unconstrained runs are unchanged.
            converged = (abs(dE) < self.gap_tol) and (seam_norm_mw < self.seam_tol)

        # trust radius on the seam gradient -- the part BFGS actually models.
        # cone_step with an EMPTY seam space: the seam norm is numerical noise, so
        # steer the radius by |dE| instead (the quantity actually being reduced).
        seam_dim = int(round(float(np.trace(Q))))
        merit = abs(dE) if (self.cone_step and seam_dim == 0) else seam_norm_mw
        phase = "seam"
        if self.merit == "two_phase" and abs(dE) > self.far_gap:
            merit, phase = abs(dE), "gap"
        if phase != self._phase:          # a different quantity: never compare |dE| with a gradient norm
            self._prev_norm = None
        self._phase = phase
        if self._prev_norm is not None:
            if merit < self._prev_norm:
                self.trust = min(self.trust * 1.25, self.max_trust)
            else:
                self.trust = max(self.trust * 0.5, self.min_trust)
                self.n_shrink += 1

        if self.B is None:
            self.B = 0.5 * np.eye(self.ndim)
        elif self._prev_s is not None and self._prev_g is not None:
            sv = self._prev_s
            y = g_seam - self._prev_g
            sy = float(sv @ y); Bs = self.B @ sv; sBs = float(sv @ Bs)
            if sBs > 1e-14:
                theta = 1.0 if sy >= 0.2 * sBs else (0.8 * sBs) / (sBs - sy)
                r = theta * y + (1.0 - theta) * Bs
                sr = float(sv @ r)
                if abs(sr) > 1e-14:
                    self.B = self.B - np.outer(Bs, Bs) / sBs + np.outer(r, r) / sr

        Bq = Q @ self.B @ Q + (np.eye(self.ndim) - Q)
        Bq = 0.5 * (Bq + Bq.T)
        try:
            s_seam = -np.linalg.solve(Bq, g_seam)
        except np.linalg.LinAlgError:
            s_seam = -2.0 * g_seam
        s_seam = Q @ s_seam
        if float(s_seam @ g_seam) > 0.0:
            s_seam = -2.0 * g_seam
            self.B = 0.5 * np.eye(self.ndim)

        gdiff_mw = np.linalg.norm(P @ ((gq - gp) / w))
        s_gap = -(dE / gdiff_mw) * ug if gdiff_mw > 1e-12 else np.zeros_like(s_seam)
        if self.cone_step and gdiff_mw > 1e-12:
            gm = P @ ((gq - gp) / w)
            hm = P @ (np.asarray(h, dtype=float).ravel() / w)
            if np.linalg.norm(hm) > 1e-8 * gdiff_mw:
                A = np.vstack([gm, hm])                       # rows: the two conditions
                s_gap = np.linalg.pinv(A) @ np.array([-dE, 0.0])   # minimum-norm solution

        s_int = s_gap + s_seam
        snorm = np.linalg.norm(s_int)
        if snorm > self.trust:
            s_int *= self.trust / snorm

        self._prev_s = (Q @ s_int).copy()
        self._prev_g = g_seam.copy()
        self._prev_norm = merit
        self.nstep += 1

        if not converged:
            self._x_prev = self.x.copy()
            self.x = self.x + (s_int / w).reshape(-1, 3)
            self._enforce_bonds()

        return {
            "converged": converged, "gap_eV": dE * HARTREE2EV, "gap_hartree": dE,
            "g_eff_norm": gnorm, "seam_norm": seam_norm, "seam_norm_mw": seam_norm_mw,
            "step_norm_bohr": float(np.linalg.norm(s_int / w)),
            "gap_step_norm": float(np.linalg.norm(s_gap / w)),
            "trust": self.trust, "cos_phi": cos_phi,
            "n_shrink": self.n_shrink, "n_reject": self.n_reject,
        }
