# -*- coding: utf-8 -*-
"""Level-shifted (Jacobi-)Davidson trust-region solver.

A faithful port of OpenTrustRegion's ``level_shifted_davidson`` and its
collaborators, from ``/home/nvu12/software/opentrustregion/src/opentrustregion.f90``.
Every routine below carries the ``otr:LINE`` citation of the Fortran it was
ported from, in the same style as ``cpp_casscf/``.

Reference:
    Greiner, J.; Hoyvik, I.-M.; Lehtola, S.; Eriksen, J. J.
    "A Reusable Library for Second-Order Orbital Optimization Using the Trust
    Region Method." J. Chem. Theory Comput. 2026, 22(2), 881-895.
    arXiv:2509.13931.  Library: https://github.com/eriksen-lab/opentrustregion

WHY THIS EXISTS
---------------
``helper_PFCI.py`` currently runs the LSTRS ``beta`` bisection OUTSIDE the
Davidson solve (:11516-11560 and the twin at :13330-13440)::

    while True:                                   # beta bisection
        Q = self.Davidson_augmented_hessian_solve6(..., beta, ...,
                                                   restart=(count10 > 0))

so every trial ``beta`` re-enters a full Davidson solve and pays fresh
Hessian-vector products.  Measured on c10h8 at true single core: 9 cold builds
(302 matvecs) + 36 warm restarts (414 matvecs) = 716 of ``lstrs_davidson``'s
749 matvecs are spent re-deriving a search space for a new beta.

This module inverts that nesting.  ONE search space ``(V, HV)`` is built and
grown; the level-shift search runs entirely inside the projected
``(n_red+1) x (n_red+1)`` augmented Hessian, where changing the shift costs one
small ``eigh`` and one ``dgemv`` and NO Hessian-vector product at all.

That is exactly the acceleration of

    Lampe, Rojas, Sorensen, Voss, "Accelerating the LSTRS Algorithm",
    IMM-Technical Report-2009-09 (DTU),

which replaces LSTRS's implicitly restarted Lanczos with the Nonlinear Arnoldi
method precisely so that "these projected problems can be updated and reused
without further cost when changing the parameter alpha".  Note that report's
verdict on the strategy the current code uses -- seeding the next eigensolve
with the previous eigenvector -- is explicit: "Recycling of prior information
can be accomplished in ARPACK by choosing the eigenvector of the last step
corresponding to the smallest eigenvalue as initial vector in the next step.
However, this usually does not significantly speed convergence."  Our 36 warm
restarts still costing 11.5 matvecs each is that sentence, measured.

CONVENTIONS AND ONE PARAMETERISATION DIFFERENCE
-----------------------------------------------
LSTRS (Lampe et al. Eq 1.2) borders with the parameter in the corner::

    B_alpha = [[alpha, g^T],
               [g,     H  ]]

OpenTrustRegion instead pins the corner at zero and scales the gradient
border, ``aug_hess[0, 1] = aug_hess[1, 0] = alpha * |g|`` (otr:810-846).  These
are two parameterisations of the same one-parameter family of bordered pencils;
this port follows OpenTrustRegion's.  A further difference worth knowing if the
bisection ever shows up in a profile: LSTRS updates alpha through a rational
Hermitian interpolant of ``phi(lambda) = g^T (H - lambda I)^dagger g`` rather
than by bisection, which converges in far fewer alpha steps.  Since every alpha
step here is free, bisection is ported as-is.

DOCUMENTED DEVIATIONS FROM THE FORTRAN
--------------------------------------
1. ``minres`` (otr:1524-1731) is NOT ported.  The Jacobi-Davidson branch here
   solves the same correction equation with ``scipy.sparse.linalg.minres``
   against a ``LinearOperator``.  Mathematically the right substitute, not a
   literal port.  This branch is inactive under the default
   ``subsystem_solver="davidson"``, which is OpenTrustRegion's own default.
2. ``generate_random_trial_vectors`` (otr:1228) seeds from a fixed RNG seed
   (settings.seed) as the Fortran does, but numpy's generator is not
   bit-compatible with the Fortran one, so trial subspaces will differ in the
   random directions.  With the default ``n_random_trial_vectors=1`` this is a
   single vector.
3. The Fortran's error-code plumbing (``error``/``add_error_origin``) is
   replaced by exceptions.
4. ``settings%project`` (the constraint projector hook) is supported but the
   symmetry/redundancy projections helper_PFCI applies are the caller's job.

This module is standalone: it imports only numpy (scipy only if the
Jacobi-Davidson branch is requested) and does not import or modify
``helper_PFCI.py``.
"""

import numpy as np

# otr:38-39, otr:23
NUMERICAL_ZERO = 1e-14
PRECOND_FLOOR = 1e-10
HESS_SYMM_THRES = 1e-12
# otr:1286
ZERO_THRES = 1e-16
ORTH_THRES = 1e-14

# otr:26-29
TRUST_RADIUS_SHRINK_RATIO = 0.25
TRUST_RADIUS_EXPAND_RATIO = 0.75
TRUST_RADIUS_SHRINK_FACTOR = 0.7
TRUST_RADIUS_EXPAND_FACTOR = 1.2

# otr:1905-1909 (level_shifted_davidson local parameters)
NEWTON_EIGVAL_THRESH = -1e-5
LEVEL_SHIFT_LOCAL_THRES = 1e-12
SOLUTION_OVERLAP_THRESH = 0.5
RESIDUAL_NORM_FLOOR = 1e-12
RESIDUAL_NORM_MAX_RED_FACTOR = 0.8

# otr:734-736 (bisection local parameters)
LOWER_ALPHA_BOUND = 1e-4
UPPER_ALPHA_BOUND = 1e6
ALPHA_CONV_FACTOR = 1e-12


class TrustRegionSettings:
    """Solver settings.  Defaults are OpenTrustRegion's ``default_solver_settings``
    (otr:174-179)."""

    def __init__(
        self,
        conv_tol=1e-5,
        start_trust_radius=0.4,
        global_red_factor=1e-3,
        local_red_factor=1e-4,
        n_random_trial_vectors=1,
        n_macro=150,
        n_micro=50,
        jacobi_davidson_start=30,
        seed=42,
        verbose=0,
        subsystem_solver="davidson",
        precond=None,
        project=None,
    ):
        self.conv_tol = conv_tol
        self.start_trust_radius = start_trust_radius
        self.global_red_factor = global_red_factor
        self.local_red_factor = local_red_factor
        self.n_random_trial_vectors = n_random_trial_vectors
        self.n_macro = n_macro
        self.n_micro = n_micro
        self.jacobi_davidson_start = jacobi_davidson_start
        self.seed = seed
        self.verbose = verbose
        self.subsystem_solver = subsystem_solver
        self.precond = precond
        self.project = project

    def log(self, message, level=1):
        if self.verbose >= level:
            print(message)


class HessianCounter:
    """Wraps the caller's ``hess_x`` and counts Hessian-vector products.

    The point of this port is to cut that count, so it is instrumented from the
    start.  Mirrors the Fortran's global ``tot_hess_x`` (otr:187).
    """

    def __init__(self, hess_x):
        self._hess_x = hess_x
        self.count = 0

    def __call__(self, v):
        self.count += 1
        return np.asarray(self._hess_x(v), dtype=float)


# ---------------------------------------------------------------------------
# small reduced-space helpers
# ---------------------------------------------------------------------------


def symm_mat_min_eig(symm_matrix):
    """Lowest eigenpair of a small symmetric matrix.  otr:1052-1107 (DSYEV)."""
    eigvals, eigvecs = np.linalg.eigh(symm_matrix)
    return eigvals[0], eigvecs[:, 0]


def min_eigval(matrix):
    """Lowest eigenvalue only.  otr:1109-1160."""
    return np.linalg.eigvalsh(matrix)[0]


def extend_symm_matrix(matrix, vector):
    """Grow a symmetric matrix by one row/column.  otr:1001-1025."""
    n = matrix.shape[0]
    out = np.empty((n + 1, n + 1), dtype=float)
    out[:n, :n] = matrix
    out[:, n] = vector
    out[n, :] = vector
    return out


def add_column(matrix, new_col):
    """Append a column.  otr:1027-1050."""
    return np.column_stack([matrix, new_col])


def gram_schmidt(vector, space, lin_trans_vector=None, lin_trans_space=None):
    """Orthonormalise ``vector`` against the columns of ``space``.  otr:1268-1360.

    Repeats the sweep until the largest overlap falls below ``ORTH_THRES``
    (the Fortran's ``do while (.true.)`` with a 100-iteration guard).  When
    ``lin_trans_vector``/``lin_trans_space`` are given, the SAME operations are
    applied to the linear transformation so that ``H v`` is carried along
    without a fresh Hessian-vector product -- this is what makes the
    Jacobi-Davidson expansion cost one matvec instead of two.
    """
    vector = np.array(vector, dtype=float)
    n_param, n_vectors = space.shape

    if np.linalg.norm(vector) < ZERO_THRES:
        raise RuntimeError("Vector is numerically zero and cannot be orthonormalized.")
    if n_vectors > n_param - 1:
        raise RuntimeError("Reduced space has reached the size of the full space.")

    carry = lin_trans_vector is not None and lin_trans_space is not None
    if carry:
        lin_trans_vector = np.array(lin_trans_vector, dtype=float)

    for _ in range(101):
        if not carry:
            # otr:1316-1325
            for i in range(n_vectors):
                vector -= np.dot(space[:, i], vector) * space[:, i]
            vector /= np.linalg.norm(vector)
        else:
            # otr:1334-1345 -- project the transformation with the same coefficients
            for i in range(n_vectors):
                dot = np.dot(space[:, i], vector)
                vector -= dot * space[:, i]
                lin_trans_vector -= dot * lin_trans_space[:, i]
            norm = np.linalg.norm(vector)
            vector /= norm
            lin_trans_vector /= norm

        if np.max(np.abs(space.T @ vector)) < ORTH_THRES:
            break
    else:
        raise RuntimeError("Maximum number of Gram-Schmidt iterations reached.")

    return (vector, lin_trans_vector) if carry else vector


def level_shifted_diag_precond(vector, mu, h_diag, settings):
    """Level-shifted Jacobi preconditioner ``r / (h_diag - mu)``.  otr:1412-1446."""
    if settings.precond is not None:
        out = np.asarray(settings.precond(vector, mu), dtype=float)
    else:
        denom = h_diag - mu
        denom = np.where(np.abs(denom) < PRECOND_FLOOR, PRECOND_FLOOR, denom)
        out = vector / denom
    if settings.project is not None:
        out = np.asarray(settings.project(out), dtype=float)
    return out


def generate_trial_vectors(grad, grad_norm, h_diag, settings):
    """Initial search space.  otr:1179-1226.

    Column 1 is the normalised gradient.  If the smallest Hessian diagonal is
    negative, a unit vector along it is added second (a negative-curvature
    seed), then ``n_random_trial_vectors`` random columns.  Note there is no
    P-space here: OpenTrustRegion, like BAGEL, does not build an exact
    sub-block of the Hessian.  helper_PFCI's 200-dimensional
    ``build_orbital_hessian_guess`` P-space (Molpro/Kreplin style) is a
    STRONGER seed and can be dropped in here as extra leading columns -- see
    the module docstring's note on keeping what already works.
    """
    n_param = grad.shape[0]
    min_idx = int(np.argmin(h_diag))

    if h_diag[min_idx] < 0.0 and n_param > 2:
        basis = np.zeros((n_param, 2))
        basis[:, 0] = grad / grad_norm
        basis[min_idx, 1] = 1.0
        if settings.project is not None:
            basis[:, 1] = np.asarray(settings.project(basis[:, 1]), dtype=float)
        basis[:, 1] = gram_schmidt(basis[:, 1], basis[:, :1])
    else:
        basis = (grad / grad_norm).reshape(n_param, 1)

    # otr:1228-1266
    rng = np.random.default_rng(settings.seed)
    for _ in range(settings.n_random_trial_vectors):
        vec = rng.uniform(-1.0, 1.0, size=n_param)
        if settings.project is not None:
            vec = np.asarray(settings.project(vec), dtype=float)
        try:
            vec = gram_schmidt(vec, basis)
        except RuntimeError:
            continue
        basis = add_column(basis, vec)

    return basis


# ---------------------------------------------------------------------------
# the two reduced-space subproblem solvers -- NEITHER costs a matvec
# ---------------------------------------------------------------------------


def newton_step(aug_hess, grad_norm, red_space_basis):
    """Newton equations solved in the REDUCED space.  otr:655-717.

    ``red_hess = aug_hess[1:, 1:]`` is n_red x n_red, so this is a dense
    ``dsysv`` on a matrix of dimension ~50, not a matrix-free iterative solve
    in the full parameter space.

    This is the routine that replaces helper_PFCI's ``newton_small_gradient``
    branch, which spends 728 Hessian-vector products -- 30% of the run's total
    -- solving the same equations matrix-free at full dimension.
    """
    n_red = red_space_basis.shape[1]
    red_hess = np.array(aug_hess[1:, 1:], dtype=float)
    rhs = np.zeros(n_red)
    rhs[0] = -grad_norm
    red_space_solution = np.linalg.solve(red_hess, rhs)
    solution = red_space_basis @ red_space_solution
    return solution, red_space_solution


def bisection(aug_hess, grad_norm, red_space_basis, trust_radius):
    """Find the level shift matching the trust radius.  otr:719-850.

    THIS IS THE WHOLE POINT OF THE PORT.  Every trial ``alpha`` writes two
    scalars into the border of the projected augmented Hessian, re-diagonalises
    an ``(n_red+1) x (n_red+1)`` matrix and back-transforms with one ``dgemv``.
    Up to 100 bisection steps, ZERO Hessian-vector products.

    Returns ``(solution, red_space_solution, mu, bracketed)``.  ``bracketed``
    False means the trust radius is not attainable within the current search
    space, and the caller must expand it (otr:763-770).
    """
    n_red = red_space_basis.shape[1]
    aug_hess = np.array(aug_hess, dtype=float)
    state = {}

    def get_ah_lowest_eigenvec(alpha):
        """otr:810-846.  Only the border depends on alpha."""
        aug_hess[0, 1] = alpha * grad_norm
        aug_hess[1, 0] = alpha * grad_norm
        mu, eigvec = symm_mat_min_eig(aug_hess)
        if abs(eigvec[0]) <= NUMERICAL_ZERO:
            raise RuntimeError(
                "Trial subspace too small. Increase n_random_trial_vectors."
            )
        red_space_solution = eigvec[1:] / eigvec[0] / alpha
        solution = red_space_basis @ red_space_solution
        state["mu"] = mu
        state["red"] = red_space_solution
        state["sol"] = solution
        return solution

    lower_alpha = LOWER_ALPHA_BOUND
    upper_alpha = UPPER_ALPHA_BOUND

    lower_trust_dist = np.linalg.norm(get_ah_lowest_eigenvec(lower_alpha)) - trust_radius
    upper_trust_dist = np.linalg.norm(get_ah_lowest_eigenvec(upper_alpha)) - trust_radius

    # otr:763-770 -- trust radius outside the bracket, caller must expand V
    if lower_trust_dist * upper_trust_dist > 0.0:
        n_param = red_space_basis.shape[0]
        return np.zeros(n_param), np.zeros(n_red), 0.0, False

    middle_alpha = np.sqrt(upper_alpha * lower_alpha)
    middle_trust_dist = np.linalg.norm(get_ah_lowest_eigenvec(middle_alpha)) - trust_radius

    it = 0
    while upper_alpha - lower_alpha > ALPHA_CONV_FACTOR * upper_alpha:
        if lower_trust_dist * middle_trust_dist > 0.0:
            lower_alpha = middle_alpha
            lower_trust_dist = middle_trust_dist
        else:
            upper_alpha = middle_alpha
            upper_trust_dist = middle_trust_dist
        middle_alpha = np.sqrt(upper_alpha * lower_alpha)
        middle_trust_dist = (
            np.linalg.norm(get_ah_lowest_eigenvec(middle_alpha)) - trust_radius
        )
        it += 1
        if it > 100:
            raise RuntimeError("Maximum number of bisection iterations reached.")

    return state["sol"], state["red"], state["mu"], True


def accept_trust_region_step(solution, ratio, micro_converged, trust_radius, settings):
    """Trust-radius update and step acceptance.  otr:2396-2439.

    NOTE this is the piece Molpro and BAGEL do NOT have -- their augmented
    Hessian schemes carry no trust radius and no step rejection at all.
    OpenTrustRegion keeps classic trust-region control, and so does this port:
    moving the level-shift search into the reduced space and dropping step
    control are independent choices.
    """
    max_precision_reached = False

    if (not micro_converged) or ratio < 0.0 or np.any(np.abs(solution) > np.pi / 4):
        trust_radius = TRUST_RADIUS_SHRINK_FACTOR * trust_radius
        if trust_radius < NUMERICAL_ZERO:
            settings.log(
                "Trust radius too small. Convergence criterion is not fulfilled "
                "but calculation should be converged up to floating point precision."
            )
            max_precision_reached = True
        return False, trust_radius, max_precision_reached

    if ratio < TRUST_RADIUS_SHRINK_RATIO:
        trust_radius = TRUST_RADIUS_SHRINK_FACTOR * trust_radius
    elif ratio >= TRUST_RADIUS_EXPAND_RATIO:
        trust_radius = TRUST_RADIUS_EXPAND_FACTOR * trust_radius
    return True, trust_radius, max_precision_reached


def _jacobi_davidson_correction(hess_x, residual, solution_normalized, mu, tol, n_param):
    """DEVIATION (see module docstring, item 1).  otr:2052-2056 calls the
    library's own MINRES; this uses scipy's on the projected operator

        (I - x x^T)(H - mu I)(I - x x^T) t = -r,   x = solution_normalized.
    """
    from scipy.sparse.linalg import LinearOperator, minres

    x = solution_normalized

    def matvec(v):
        v = v - x * np.dot(x, v)
        w = hess_x(v) - mu * v
        return w - x * np.dot(x, w)

    op = LinearOperator((n_param, n_param), matvec=matvec, dtype=float)
    t, _ = minres(op, -residual, rtol=tol)
    return t




# ---------------------------------------------------------------------------
# the search space -- built ONCE, reused across level shifts AND retries
# ---------------------------------------------------------------------------


class SearchSpace:
    """The persistent ``(V, HV, aug_hess)`` triple.  otr:1917-1946.

    This object IS the fix.  ``helper_PFCI``'s beta loop
    (:11516-11560, :13330-13440) throws this state away and rebuilds it for
    every trial level shift; here it is built once and every subsequent level
    shift is a small ``eigh`` on ``aug_hess``.

    Also reused across trust-radius retries: OpenTrustRegion's
    ``while not accept_step`` loop (otr:1948) sits OUTSIDE the allocation, so a
    rejected step keeps every Hessian-vector product already paid for.
    """

    def __init__(self, grad, grad_norm, h_diag, hess_x, settings):
        self.basis = generate_trial_vectors(grad, grad_norm, h_diag, settings)
        n_trial = self.basis.shape[1]
        self.h_basis = np.column_stack(
            [hess_x(self.basis[:, i]) for i in range(n_trial)]
        )
        # otr:1934-1940 -- corner pinned at zero; only the border carries alpha
        self.aug_hess = np.zeros((n_trial + 1, n_trial + 1))
        self.aug_hess[1:, 1:] = self.basis.T @ self.h_basis

    @property
    def n_trial(self):
        return self.basis.shape[1]

    def expand(self, basis_vec, h_basis_vec):
        """Add one vector and its transformation.  otr:2074-2093."""
        self.basis = add_column(self.basis, basis_vec)
        self.h_basis = add_column(self.h_basis, h_basis_vec)
        red_hess_vec = np.empty(self.n_trial + 1)
        red_hess_vec[0] = 0.0
        red_hess_vec[1:] = self.basis.T @ self.h_basis[:, -1]
        self.aug_hess = extend_symm_matrix(self.aug_hess, red_hess_vec)


def _micro_loop(space, grad, grad_norm, h_diag, hess_x, trust_radius, settings):
    """The micro iterations.  otr:1950-2094.

    One Hessian-vector product per iteration, and none at all for the level
    shift.  Returns ``(solution, h_solution, mu, micro_converged, n_micro)``.
    """
    n_param = grad.shape[0]
    last_solution_normalized = np.zeros(n_param)
    solution = np.zeros(n_param)
    h_solution = np.zeros(n_param)
    mu = 0.0
    micro_converged = False
    jacobi_davidson_started = False
    imicro_jacobi_davidson = 0
    initial_imicro = 0
    initial_residual_norm = np.inf
    imicro = 0

    for imicro in range(1, settings.n_micro + 1):
        # otr:1953-1962 -- Newton step if the projected model is positive
        # definite and the step lands inside the trust region
        newton = False
        if min_eigval(space.aug_hess[1:, 1:]) > NEWTON_EIGVAL_THRESH:
            solution, red_space_solution = newton_step(
                space.aug_hess, grad_norm, space.basis
            )
            mu = 0.0
            if np.linalg.norm(solution) < trust_radius:
                newton = True

        # otr:1964-1971 -- otherwise bisect for the level shift.  NO MATVECS.
        if not newton:
            solution, red_space_solution, mu, bracketed = bisection(
                space.aug_hess, grad_norm, space.basis, trust_radius
            )
            if not bracketed:
                break

        # otr:1973-1980 -- residual from the STORED transformations, no matvec
        h_solution = space.h_basis @ red_space_solution
        residual = grad + h_solution - mu * solution
        residual_norm = np.linalg.norm(residual)

        # otr:1982-1988
        red_factor = (
            settings.local_red_factor
            if abs(mu) < LEVEL_SHIFT_LOCAL_THRES
            else settings.global_red_factor
        )

        solution_normalized = solution / np.linalg.norm(solution)

        # otr:1997-2003 -- restart the stagnation clock if the solution moved
        if np.dot(last_solution_normalized, solution_normalized) ** 2 < (
            SOLUTION_OVERLAP_THRESH
        ):
            initial_imicro = imicro
            initial_residual_norm = residual_norm

        # otr:2005-2027
        if residual_norm < max(red_factor * grad_norm, RESIDUAL_NORM_FLOOR):
            micro_converged = True
            break
        elif settings.subsystem_solver == "jacobi-davidson" and (
            not jacobi_davidson_started
        ):
            if (
                imicro - initial_imicro >= 10
                and residual_norm > RESIDUAL_NORM_MAX_RED_FACTOR * initial_residual_norm
            ) or imicro > settings.jacobi_davidson_start:
                jacobi_davidson_started = True
                imicro_jacobi_davidson = imicro
        elif (
            imicro - initial_imicro >= 10
            and residual_norm > RESIDUAL_NORM_MAX_RED_FACTOR * initial_residual_norm
        ):
            break

        last_solution_normalized = solution_normalized

        # --- expand by exactly ONE vector, at the cost of ONE matvec ---
        if not jacobi_davidson_started:
            # otr:2032-2049
            basis_vec = level_shifted_diag_precond(residual, mu, h_diag, settings)
            basis_vec = gram_schmidt(basis_vec, space.basis)
            h_basis_vec = hess_x(basis_vec)
        else:
            # otr:2051-2072
            tol = 3.0 ** (-(imicro - imicro_jacobi_davidson))
            basis_vec = _jacobi_davidson_correction(
                hess_x, residual, solution_normalized, mu, tol, n_param
            )
            h_basis_vec = hess_x(basis_vec)
            basis_vec, h_basis_vec = gram_schmidt(
                basis_vec, space.basis, h_basis_vec, space.h_basis
            )
            # otr:2065-2071 -- rebuild if the projection broke Hessian symmetry
            if (
                abs(
                    np.dot(space.basis[:, -1], h_basis_vec)
                    - np.dot(basis_vec, space.h_basis[:, -1])
                )
                > HESS_SYMM_THRES
            ):
                h_basis_vec = hess_x(basis_vec)

        space.expand(basis_vec, h_basis_vec)

    return solution, h_solution, mu, micro_converged, imicro


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def solve_trust_region_subproblem(grad, h_diag, hess_x, trust_radius, settings=None):
    """Solve ONE trust-region subproblem and return the step.

    This is the entry point ``helper_PFCI.py`` uses under ``QED_OTR_TR=1``.  It
    is deliberately the micro loop ONLY: no ratio test, no trust-radius update,
    no step acceptance.  Those stay with the caller, so switching the knob on
    replaces the subproblem solver and NOTHING else -- helper_PFCI's own step
    control, energy checks and trust-radius logic remain in charge.

    Parameters
    ----------
    grad : (n,) ndarray
    h_diag : (n,) ndarray
    hess_x : callable, ``v -> H v``
    trust_radius : float
    settings : TrustRegionSettings, optional

    Returns
    -------
    (step, info) where info holds ``mu``, ``n_micro``, ``n_hess_x``,
    ``micro_converged``.
    """
    if settings is None:
        settings = TrustRegionSettings()

    grad = np.asarray(grad, dtype=float)
    h_diag = np.asarray(h_diag, dtype=float)
    grad_norm = np.linalg.norm(grad)

    counter = HessianCounter(hess_x)
    space = SearchSpace(grad, grad_norm, h_diag, counter, settings)
    solution, _, mu, micro_converged, n_micro = _micro_loop(
        space, grad, grad_norm, h_diag, counter, trust_radius, settings
    )

    return solution, {
        "mu": mu,
        "n_micro": n_micro,
        "n_hess_x": counter.count,
        "micro_converged": micro_converged,
    }


def level_shifted_davidson(
    func,
    grad,
    h_diag,
    hess_x,
    obj_func,
    trust_radius,
    settings=None,
):
    """Full OpenTrustRegion subproblem solve WITH its own step control.
    otr:1879-2119.

    Use this when you want OpenTrustRegion's trust-radius management too; use
    ``solve_trust_region_subproblem`` when the caller keeps its own (which is
    what helper_PFCI does).

    Returns a dict with ``solution``, ``mu``, ``trust_radius``, ``n_micro``,
    ``n_hess_x``, ``micro_converged``, ``max_precision_reached``.
    """
    if settings is None:
        settings = TrustRegionSettings()

    grad = np.asarray(grad, dtype=float)
    h_diag = np.asarray(h_diag, dtype=float)
    grad_norm = np.linalg.norm(grad)

    counter = HessianCounter(hess_x)
    # built once, OUTSIDE the retry loop (otr:1917-1946 vs otr:1948)
    space = SearchSpace(grad, grad_norm, h_diag, counter, settings)

    solution = np.zeros(grad.shape[0])
    mu = 0.0
    micro_converged = False
    max_precision_reached = False
    imicro = 0

    accept_step = False
    while not accept_step:
        solution, h_solution, mu, micro_converged, imicro = _micro_loop(
            space, grad, grad_norm, h_diag, counter, trust_radius, settings
        )

        # otr:2096-2110 -- ratio test against the quadratic model
        new_func = obj_func(solution)
        predicted = np.dot(solution, grad + 0.5 * h_solution)
        ratio = (new_func - func) / predicted if predicted != 0.0 else -1.0
        accept_step, trust_radius, max_precision_reached = accept_trust_region_step(
            solution, ratio, micro_converged, trust_radius, settings
        )
        if max_precision_reached:
            break

    return {
        "solution": solution,
        "mu": mu,
        "trust_radius": trust_radius,
        "n_micro": imicro,
        "n_hess_x": counter.count,
        "micro_converged": micro_converged,
        "max_precision_reached": max_precision_reached,
    }
