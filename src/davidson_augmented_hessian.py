"""The augmented-Hessian Davidson subspace solver, split out of helper_PFCI.py.

Third and last file in the trust-region split that produced
``otr_trust_region.py`` and ``lstrs_trust_region.py``.  Those two hold the
OUTER machinery -- OpenTrustRegion's step, and the LSTRS beta-bisection at the
microiteration and internal-optimization sites.  This holds the INNER kernel
the LSTRS bisection calls once per beta: a Davidson diagonalization of the
augmented (bordered) Hessian

    [ alpha    g^T ]
    [   g      H   ]

for its one or two lowest roots, matrix-free in ``H`` via
``ctx.orbital_sigma3``.

The subspace is carried ACROSS calls on the generator, not rebuilt per call:
``restart=True`` (every beta after the first in a bisection) reuses the search
space and its stored sigma vectors, which is the whole reason the bisection is
affordable.  Note the parameter name reads backwards -- ``restart=True`` is the
CHEAP warm-start path, ``restart=False`` is the cold build.  The solver stats
are labelled by what the paths DO (``davidson_ah_warmstart`` /
``davidson_ah_coldbuild``) rather than by the flag.

MOVED VERBATIM, NOT REWRITTEN -- see the function docstring.  The extraction is
gated bit-exactly by ``validation/lstrs_refactor_gate.sh``, whose six drivers
enter this solver 44-46 times each.
"""

import copy
import time
from timeit import default_timer as timer

import numpy as np


def davidson_augmented_hessian_solve(
    ctx,
    U,
    A_tilde,
    G,
    G_blocks,
    reduced_hessian_diagonal,
    reduced_gradient,
    alpha,
    alpha_range,
    trust_radius,
    aug_hessian_eigenvecs,
    aug_hessian_eigenvals,
    guess_vector,
    restart
):
    """The augmented-Hessian Davidson at the heart of the LSTRS beta-bisection.

    MOVED VERBATIM from ``PFHamiltonianGenerator.Davidson_augmented_hessian_solve6``
    (helper_PFCI.py:17248-18108 before the move).  The only edits are the
    ``self`` -> ``ctx`` rename and this docstring; the arithmetic, the print
    stream, and even the dead commented-out blocks are byte-for-byte what they
    were.  ``helper_PFCI.PFHamiltonianGenerator.Davidson_augmented_hessian_solve6``
    survives as a one-line delegator, so every existing call site -- the three
    in ``lstrs_trust_region.solve_lstrs_microiteration`` and the cpp_casscf
    validation dumps -- is untouched.

    WHY IT MOVED.  This was the last 861-line block still inline in
    ``helper_PFCI.py`` after the LSTRS extraction (item 373) and the
    microiteration/internal-site extractions (items 377-378).  It is also the
    thing those extracted solvers CALL, so leaving it behind meant
    ``lstrs_trust_region.py`` reaching back into the generator for its own
    innermost kernel.

    DELIBERATELY NOT the ``hess_x``-closure interface that ``otr_trust_region.py``
    uses.  This solver does not merely multiply by the Hessian: it builds the
    guess subspace from the orbital-Hessian diagonal
    (``ctx.build_orbital_hessian_guess``), keeps cross-call subspace state on the
    generator (``ctx.projected_augmented_hessian``, ``ctx.sigma_total``,
    ``ctx.H_pp``, ``ctx.H_qp``, ``ctx.idx_hessian``, ``ctx.indim``,
    ``ctx.collapse_subspace_check``), and calls back into ``ctx.orbital_sigma3``
    and ``ctx.inner_solve_pcg``.  Narrowing that to three closures would be a
    redesign, not a move; ``ctx``-passing is what
    ``lstrs_trust_region.solve_lstrs_microiteration`` already does for the same
    reason.

    KNOWN LATENT BUG, PRESERVED.  In the ``restart == False`` initial-guess
    block, the ``nroots == 2`` / both-roots-converged branch reads
    ``full_eigvecs``, which is not bound until the Davidson loop below it -- so
    that branch raises ``NameError`` if it is ever reached.  It is left exactly
    as it was: this commit is a move, and fixing it here would make the gate
    unable to tell a fix from a mistake.

    ``ctx`` is a ``PFHamiltonianGenerator``.  Returns the final search space
    ``Q``; the converged roots are written in place into
    ``aug_hessian_eigenvecs`` / ``aug_hessian_eigenvals``.
    """
    # Module constants and ctypes wrappers still live in helper_PFCI, which
    # imports THIS module, so a module-level import would be circular.  All of
    # these are fixed at import time, so reading them per call is
    # value-identical to reading them as globals (same idiom as
    # lstrs_trust_region.solve_lstrs_microiteration).
    from helper_PFCI import (
        _CPP_CASSCF_VALIDATION_DIR, _QED_SOLVER_STATS,
        _dump_cpp_casscf_validation_case, _solver_stats,
        _stats_begin, _stats_end,
        c_gram_schmidt_add, c_gram_schmidt_orthogonalization,
    )
    # Attributes this Davidson solve's matvecs separately from the LSTRS
    # bisection that wraps it, split by how the subspace was seeded.
    # NOTE the parameter name reads backwards: restart=True means "restart
    # FROM the previous search space" (Q = deepcopy(guess_vector), the warm
    # start used for every alpha after the first), while restart=False
    # builds a fresh guess subspace from |g|/H_diag.  So restart=True is the
    # CHEAP path.  Labelled by what they do, not by the flag's name.
    _stats_span_d = _stats_begin(
        "davidson_ah_warmstart" if restart else "davidson_ah_coldbuild"
    )
    # Analyze problem structure
    n_negative = np.sum(reduced_hessian_diagonal < 0)
    min_diag = np.min(reduced_hessian_diagonal)
    max_diag = np.max(reduced_hessian_diagonal)
    grad_norm = np.linalg.norm(reduced_gradient)
    
    print(f"Problem structure:")
    print(f"  Negative curvature directions: {n_negative}/{ctx.index_map_size}")
    print(f"  H_diag range: [{min_diag:.2e}, {max_diag:.2e}]")
    print(f"  ||gradient||: {grad_norm:.2e}")
    print(f"  Condition estimate: {max_diag/abs(min_diag) if min_diag != 0 else 'inf'}") 
    print("collapse subspace check", ctx.collapse_subspace_check)
    threshold = 1e-7
    count = 0
    # start = timer()
    print("restart", restart, "alpha", alpha)
    for i in range(ctx.index_map_size):
        if reduced_hessian_diagonal[i] <= 1e-14:
            count += 1
    if ctx.index_map_size > 600:
        dim0 = 200
    else:
        dim0 = ctx.index_map_size // 2
        #dim0 = ctx.index_map_size 
    dim1 = max(count, dim0)
    # print("eepp",reduced_hessian_diagonal)

    dim2 = ctx.index_map_size + 1
    H_dim = dim2
    H_diag0 = copy.deepcopy(reduced_hessian_diagonal)
    # print(H_diag)
    H_diag = np.concatenate(([alpha], H_diag0))
    #print("H_diag", H_diag)
    indim = dim1 + 1
    # maxdim = dim1 + 3
    maxdim = min(H_dim, dim1 + min(dim1, 40))
    ctx.project_augmented_hessian00 = np.zeros((maxdim, maxdim))
    print("indim", indim, "maxdim", maxdim, "Hdim", H_dim)
    H_pp = np.zeros((indim, indim))
    unconverged_idx = []
    if alpha_range > 1e-5:
        nroots = 1
        roots_to_check = [0]
    else:
        nroots = 2
        roots_to_check = [0, 1]

    nroots_target = 2
    root_0_locked = False

    # end   = timer()
    # print("guess1 took", end - start)

    # start = timer()
    if restart == False:
        # start1 = timer()
        out = np.empty(ctx.index_map_size)
        out.fill(1e14)
        d = np.divide(
                np.absolute(reduced_gradient),
                reduced_hessian_diagonal,
                out,
                where=reduced_hessian_diagonal > 1e-5,
                )
        

        ctx.idx_hessian = (-d).argsort()[:dim1]

        ctx.indim = indim


        #print(d)
        #print("order", ctx.idx_hessian)
        ctx.guess_hessian = np.zeros((dim1, dim1))
        ctx.guess_gradient = np.zeros(dim1)
        sym_A_tilde = A_tilde + A_tilde.T
        # end1   = timer()
        # print("building orbital guess1 took1", end1 - start1, flush = True)

        

        print("davidson iteration", 1)
        ctx.build_orbital_hessian_guess(
                U,
                sym_A_tilde,
                reduced_gradient,
                G,
                ctx.guess_hessian,
                ctx.guess_gradient,
                dim1,
                ctx.idx_hessian,
                )
        if _CPP_CASSCF_VALIDATION_DIR:
            _dump_cpp_casscf_validation_case(
                "hessian_guess",
                U=U, sym_A_tilde=sym_A_tilde, reduced_gradient=reduced_gradient, G=G,
                idx=np.asarray(ctx.idx_hessian, dtype=float),
                guess_hessian=ctx.guess_hessian, guess_gradient=ctx.guess_gradient,
                dims=np.array([ctx.n_in_a, ctx.n_act_orb, ctx.n_virtual, ctx.nmo, ctx.n_occupied]),
            )

        aug_eigvecs = np.zeros((dim1 + 1, dim1 + 1))
        Lmax = maxdim
        L = indim

        H_pp = np.zeros((indim, indim))
        H_pp[0, 0] = alpha
        H_pp[0, 1:] = ctx.guess_gradient
        H_pp[1:, 0] = ctx.guess_gradient.T
        H_pp[1:, 1:] = ctx.guess_hessian
        ctx.H_pp = copy.deepcopy(H_pp)
        ctx.H_qp = np.zeros((0, indim))
        ctx.sigma_total = np.zeros((maxdim, H_dim))
        Q = np.zeros((indim, H_dim))
        projected_step = np.zeros(dim1)
        theta = ctx.projection_step2(
            ctx.guess_gradient, ctx.guess_hessian, aug_eigvecs, alpha, dim1 + 1
        )

        # start1 = timer()
        for i in range(dim1 + 1):
            for j in range(dim1):
                index1 = ctx.idx_hessian[j]
                Q[i][0] = aug_eigvecs[0][i]
                Q[i][index1 + 1] = aug_eigvecs[j + 1][i]
        
        
        num_to_build = len(roots_to_check)
        w_sigma = np.zeros((num_to_build, H_dim)) # Holds sigma vectors
        # Build sigma only for active roots (e.g., [0] or [0, 1])
        ##ctx.orbital_sigma(U, A_tilde, G1, Q, w, nroots, 1)
        start = timer()
        Q_slice = np.ascontiguousarray(Q[roots_to_check, :])
        ctx.orbital_sigma3(U, A_tilde, G_blocks, Q_slice, w_sigma, num_to_build, 1)
        end = timer()
        print("build orbital sigma for w took", end - start)


        w = np.zeros((nroots_target, H_dim)) # Full residual matrix
        residual_norm = np.full(nroots_target, np.inf)
        conv_status = [False] * nroots_target
        
        for i, root_idx in enumerate(roots_to_check):
            vec = Q[root_idx, :]
            w_vec = w_sigma[i, :] # Get the corresponding sigma
            w_vec[1:] += reduced_gradient * vec[0]
            w_vec[0] = alpha * vec[0] + np.dot(vec[1:], reduced_gradient)
            
            r_vec = w_vec - theta[root_idx] * vec
            w[root_idx, :] = r_vec # Store the *residual*
            residual_norm[root_idx] = np.linalg.norm(r_vec)
            
            if residual_norm[root_idx] < threshold:
                conv_status[root_idx] = True 



        unconverged_idx = []
        exit_solver = False
        converged_indices = []
        # --- NEW LSTRS CONVERGENCE CHECK (Initial Guess) ---
        if nroots == 1:
            if conv_status[0]:
                converged_indices.append(0)
                # Root 0 is converged. Call the callback.
                if ctx.check_root_0(Q, theta[0], reduced_gradient, trust_radius):
                    # === EASY CASE ===
                    print("LSTRS (Initial Guess): Root 0 converged (Easy Case). Exiting.")
                    aug_hessian_eigenvals[0] = theta[0]
                    aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                    aug_hessian_eigenvals[1] = 1e10 # Fake root 1
                    aug_hessian_eigenvecs[:, 1] = 1e-14
                    exit_solver = True
                else:
                    # === HARD CASE ===
                    print("LSTRS (Initial Guess): Root 0 converged (Hard Case). Checking Root 1.")
                    root_0_locked = True
                    roots_to_check = [1] # Now we only care about Root 1
                    
                    # We must check Root 1 *now*
                    start1 = timer()
                    w1_sigma = np.zeros((1, H_dim))
                    ctx.orbital_sigma3(U, A_tilde, G_blocks, Q[1:2, :], w1_sigma, 1, 1)
                    #w1_sigma[0, 1:] += np.einsum("p,q->qp", reduced_gradient, Q[1, 0])
                    w1_sigma[0, 1:] += reduced_gradient * Q[1, 0] # Changed from einsum
                    w1_sigma[0, 0] = alpha * Q[1, 0] + np.dot(Q[1, 1:], reduced_gradient)
                    end1 = timer()
                    print("build orbital sigma for w took", end1 - start1)

       
                    r_1 = w1_sigma[0, :] - theta[1] * Q[1, :]
                    w[1, :] = r_1 # Store residual
                    residual_norm[1] = np.linalg.norm(r_1)
                    
                    if residual_norm[1] < threshold:
                        converged_indices.append(1)
                        conv_status[1] = True
                        print("LSTRS (Initial Guess): Root 1 also converged. Exiting.")
                        exit_solver = True
                        aug_hessian_eigenvals[0] = theta[0]
                        aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                        aug_hessian_eigenvals[1] = theta[1]
                        aug_hessian_eigenvecs[:, 1] = Q[1, :].T
                    else:
                        unconverged_idx = [1]

                    #if is_root_1_conv:
                    #    print("LSTRS (Initial Guess): Root 1 also converged. Exiting.")
                    #    aug_hessian_eigenvals[0] = theta[0]
                    #    aug_hessian_eigenvecs[:, 0] = Q[0, :].T
                    #    aug_hessian_eigenvals[1] = theta[1]
                    #    aug_hessian_eigenvecs[:, 1] = Q[1, :].T
                    #    exit_solver = True
                    #else:
                    #    print("LSTRS (Initial Guess): Root 1 unconverged. Adding to search.")
                    #    unconverged_idx = [1]
            else:
                # Root 0 is not converged
                unconverged_idx = [0]
        elif nroots == 2:
            if conv_status[0] and conv_status[1]:
                if conv_status[0]: converged_indices.append(0)
                if conv_status[1]: converged_indices.append(1)
                # Both are converged
                print("LSTRS: Both roots converged. Exiting.")
                aug_hessian_eigenvals[:] = theta[:nroots_target]
                aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                exit_solver = True
            else:
                # Find whichever is not converged
                if not conv_status[0]: unconverged_idx.append(0)
                if not conv_status[1]: unconverged_idx.append(1)
                #if not is_root_0_conv:
                #    unconverged_idx.append(0)
                #if not is_root_1_conv:
                #    unconverged_idx.append(1)

        print(unconverged_idx)
        #print("root", "AH residual norm", "Eigenvalue", "Convergence")
        #for j in range(nroots_target):
        #    print(
        #        j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
        #    )
        print("root", "AH residual norm", "Eigenvalue", "Convergence")
        for j in range(nroots_target):
            print(
                j + 1, residual_norm[j], theta[j], "Yes" if conv_status[j] else "No", flush=True
            )

        if exit_solver:
            # ctx.projected_augmented_hessian = np.block([[0, ctx.guess_gradient], [ctx.guess_gradient.T, ctx.guess_hessian]])
            ctx.projected_augmented_hessian = np.zeros((L, L))
            ctx.projected_augmented_hessian[0, 0] = 0
            ctx.projected_augmented_hessian[0, 1:] = ctx.guess_gradient
            ctx.projected_augmented_hessian[1:, 0] = ctx.guess_gradient.T
            ctx.projected_augmented_hessian[1:, 1:] = ctx.guess_hessian

            print("converged from the first iteration!", flush=True)

        # end1   = timer()
        # print("building orbital guess7 took", end1 - start1, flush = True)
        QQ = np.zeros((dim1, ctx.index_map_size))
        for i in range(dim1):
           index1 = ctx.idx_hessian[i]
           r = ctx.index_map[index1][0]
           k = ctx.index_map[index1][1]
           #guess_gradient2[i] = reduced_gradient[index1]
           QQ[i][index1] = 1.0
        Q = np.zeros((indim, H_dim))
        Q[0,0] = 1
        Q[1:,1:] = QQ

        preconditioned_w = np.zeros((0, H_dim)) # Initialize empty
        if len(unconverged_idx) > 0:
            w_unconv = w[unconverged_idx, :]
            precon_denom = theta[unconverged_idx].reshape(
                len(unconverged_idx), 1
            ) - H_diag.reshape(1, H_dim)
            preconditioned_w = np.divide(
                w_unconv, 
                precon_denom,
                out=np.zeros_like(w_unconv),
                where=precon_denom != 0,
            )
       
        
        Q = np.concatenate((Q, preconditioned_w), axis=0)
        c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
        L = Q.shape[0]
        print("current dimension", L, flush = True)
        
    else:
        print("restart from previous search space")
        # if indim == maxdim:
        #    maxdim +=1
        #print(H_dim, indim)
        Q = copy.deepcopy(guess_vector)
        Lmax = maxdim
        L = Q.shape[0]
        indim = ctx.indim
        print("current dim", L)
        print("indim", indim)
        if ctx.collapse_subspace_check == False:
            if L == indim:
                ctx.projected_augmented_hessian[0, 0] = alpha
            else:
                print("indim", indim)
                start1 = timer()
                b_dim = L - indim
                ctx.projected_augmented_hessian[0, 0] = alpha
                H_qp = np.zeros((b_dim, indim))
                Sq = np.zeros((b_dim, H_dim))
                Sq[:, 0] += alpha * Q[indim:, 0]
                ctx.sigma_total[indim:L,0] += Sq[:, 0]
                if indim > 0:
                    for i in range(b_dim):
                        ctx.projected_augmented_hessian[indim + i][0] += Sq[i][0]
                        ctx.projected_augmented_hessian[0][indim + i] += Sq[i][0]
                        ctx.H_qp[i][0] += Sq[i][0]
                        # for j in range(dim1):
                        #    #index1 = ctx.idx_hessian[j]
                        #    #H_qp[i][0] = Sq[i][0]
                        #    #H_qp[i][j+1] = Sq[i][index1+1]
                    ctx.H_pp[0, 0] = alpha
                    H_qq = np.dot(Sq, Q[indim:, :].T)
                    ctx.projected_augmented_hessian[indim:, indim:] += H_qq
                else:
                    ctx.projected_augmented_hessian[indim:, indim:] = np.dot(ctx.sigma_total[:L,:], Q[:L, :].T)
                    print("qqqq")
                theta, aug_eigvecs = np.linalg.eigh(ctx.projected_augmented_hessian)
                #print("iiii", theta)
                end1 = timer()
                print("restart took", end1 - start1)
        else:
            L_new = Q.shape[0]
            Sq = np.zeros((L_new, H_dim))
            Sq[:, 0] += alpha * Q[:, 0]
            ctx.sigma_total[:L_new,0] += Sq[:,0]
            ctx.projected_augmented_hessian = np.dot(ctx.sigma_total[:L_new,:], Q[:L_new, :].T)
                         
            # Set unconverged indices based on current LSTRS state
            if nroots == 1 and not root_0_locked:
                unconverged_idx = [0]
            elif nroots == 1 and root_0_locked:
                unconverged_idx = [1]
            elif nroots == 2:
                unconverged_idx = [0, 1]
            # --- END MODIFIED RESTART=TRUE BLOCK ---
    if (len(unconverged_idx) > 0 and restart == False) or restart == True:
        num_iter = 10000
        # num_iter = 20
        L_old = indim
        collapse = True
        for davidson_iteration in range(1, num_iter):
            if davidson_iteration == num_iter - 1:
                print(
                    "maximum iteration reaches but no converged roots have been found"
                )
                exit()
            print("\n")
            L_new = Q.shape[0]
            L = Q.shape[0]
            projected_step = np.zeros(L - 1)
            full_eigvecs = np.zeros((nroots, H_dim))
            full_eigvecs2 = np.zeros((nroots, H_dim))
            projected_augmented_hessian = np.zeros((1, 1))
            if restart == True:
                print("davidson iteration", davidson_iteration)
            else:
                print("davidson iteration", davidson_iteration + 1)
            if restart == True and davidson_iteration == 1:
                projected_augmented_hessian = ctx.projected_augmented_hessian
            else:
                if ctx.collapse_subspace_check == False:   
                    

                    start = timer()
                    c_dim = L_new - L_old
                    Sq1 = np.zeros((c_dim, H_dim))
                    # ctx.build_sigma_reduced2(U, A_tilde, G, Q[indim:,:], Sq0, b_dim, 1)
                    # c_build_sigma_reduced(U, A_tilde, ctx.index_map1, G1, Q[indim:,:], Sq0, b_dim, 1, ctx.nmo, ctx.index_map_size, ctx.n_occupied)
                    #ctx.orbital_sigma(U, A_tilde, G1,  Q[L_old:, :], Sq1, c_dim, 1)
                    
                    ctx.orbital_sigma3(U, A_tilde, G_blocks,  Q[L_old:, :], Sq1, c_dim, 1)
                    
                    gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[L_old:, 0])
                    gradient2 = np.dot(Q[L_old:, 1:], reduced_gradient)
                    print("c dim, L old, L new", c_dim, L_old, L_new) 
                    end = timer()
                    print("build orbital sigma for Q space took", end - start)
                    Sq = copy.deepcopy(Sq1)
                    # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                    Sq[:, 1:] += gradient1
                    # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                    Sq[:, 0] = alpha * Q[L_old:, 0] + gradient2
                    ctx.sigma_total[L_old:L_new,:] = np.copy(Sq)
                    if indim > 0:
                        H_qp = np.zeros((c_dim, indim))
                        for i in range(c_dim):
                            for j in range(dim1):
                                index1 = ctx.idx_hessian[j]
                                H_qp[i][0] = Sq[i][0]
                                H_qp[i][j + 1] = Sq[i][index1 + 1]
                        ctx.H_qp = np.vstack([ctx.H_qp, H_qp])
                        H_qq = np.dot(ctx.sigma_total[indim:L_new,:], Q[indim:, :].T)
                        print("dimension of sigma", np.shape(ctx.sigma_total))
                        H1 = np.concatenate((ctx.H_pp, ctx.H_qp), axis=0)
                        H2 = np.concatenate((ctx.H_qp.T, H_qq), axis=0)
                        ctx.projected_augmented_hessian00 = np.concatenate((H1, H2), axis=1)
                        projected_augmented_hessian = np.concatenate((H1, H2), axis=1)
                    else: 
                        # Pure Q-space case (Krylov guess start)
                        # The projected matrix is just the Q-space projection H_qq
                        H_qq = np.dot(ctx.sigma_total[:L_new,:], Q[:L_new, :].T)

                        # No H_pp, H_qp, or H_pq to concatenate
                        ctx.projected_augmented_hessian00 = H_qq
                        projected_augmented_hessian = H_qq
                        #ctx.print_matrix_nice(H_qq, precision=10, width=14, cols_per_line=6)
                else:   
                    print("collapse", collapse)
                    if collapse == False:
                        L_new = Q.shape[0]
                        c_dim = L_new - L_old
                        Sq1 = np.zeros((c_dim, H_dim))
                        # ctx.build_sigma_reduced2(U, A_tilde, G, Q[indim:,:], Sq0, b_dim, 1)
                        # c_build_sigma_reduced(U, A_tilde, ctx.index_map1, G1, Q[indim:,:], Sq0, b_dim, 1, ctx.nmo, ctx.index_map_size, ctx.n_occupied)
                        start = timer()
                        #ctx.orbital_sigma(U, A_tilde, G1,  Q[L_old:, :], Sq1, c_dim, 1)
                        ctx.orbital_sigma3(U, A_tilde, G_blocks,  Q[L_old:, :], Sq1, c_dim, 1)
                        end = timer()
                        print("build orbital sigma for Q space took", end - start)

                        gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[L_old:, 0])
                        gradient2 = np.dot(Q[L_old:, 1:], reduced_gradient)
                        print("c dim, L old, L new", c_dim, L_old, L_new) 
                        Sq = copy.deepcopy(Sq1)
                        # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                        Sq[:, 1:] += gradient1
                        # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                        Sq[:, 0] = alpha * Q[L_old:, 0] + gradient2
                        ctx.sigma_total[L_old:L_new,:] = np.copy(Sq)
                        projected_augmented_hessian = np.dot(ctx.sigma_total[:L_new,:], Q[:L_new, :].T)
                        L_old = Q.shape[0]
                    else:
                        L_new = Q.shape[0]
                        ctx.sigma_total[:,:] = 0
                        c_dim = L_new
                        Sq1 = np.zeros((c_dim, H_dim))
                        start = timer()
                        #ctx.orbital_sigma(U, A_tilde, G1,  Q, Sq1, c_dim, 1)
                        ctx.orbital_sigma3(U, A_tilde, G_blocks,  Q, Sq1, c_dim, 1)
                        end = timer()
                        print("build orbital sigma for Q space took", end - start)

                        gradient1 = np.einsum("p,q->qp", reduced_gradient, Q[:, 0])
                        gradient2 = np.dot(Q[:, 1:], reduced_gradient)
                        print("c dim, L old, L new", c_dim, L_old, L_new) 
                        Sq = copy.deepcopy(Sq1)
                        # Sq[:,1:] += alpha * np.einsum("p,q->qp", reduced_gradient, Q[indim:,0])
                        Sq[:, 1:] += gradient1
                        # Sq[:,0] = alpha * np.einsum("p,rp->r", reduced_gradient, Q[indim:,1:] )
                        Sq[:, 0] = alpha * Q[:, 0] + gradient2
                        ctx.sigma_total[:L_new,:] = np.copy(Sq)
                        projected_augmented_hessian = np.dot(ctx.sigma_total[:L_new,:], Q[:L_new, :].T)

            theta, aug_eigvecs = np.linalg.eigh(projected_augmented_hessian)
            #print("zopp",theta)
            # print(np.shape(aug_eigvecs))
            # print(aug_eigvecs.T[0,:])
            full_eigvecs = np.dot(aug_eigvecs.T[:nroots_target, :], Q[:,])
            full_eigvecs2 = copy.deepcopy(full_eigvecs)
              


            start = timer()
            roots_to_check = []
            if nroots == 1 and not root_0_locked:
                roots_to_check = [0]
            elif nroots == 1 and root_0_locked:
                roots_to_check = [1]
            elif nroots == 2:
                roots_to_check = [0, 1]
            
            num_to_build = len(roots_to_check)
            w = np.zeros((nroots_target, H_dim)) # Full residual matrix
            residual_norm = np.full(nroots_target, np.inf)
            conv_status = [False] * nroots_target
            
            if num_to_build > 0:
                w_sigma = np.zeros((num_to_build, H_dim)) # For sigma vectors
                
                # --- CORE BUG FIX ---
                if ctx.collapse_subspace_check == True:
                    # sigma_total *already* has gradient/alpha terms from the rebuild
                    w_sigma = np.dot(aug_eigvecs.T[roots_to_check, :], ctx.sigma_total[:L_new,:])
                else:
                    # We must build sigma from scratch and add gradient/alpha terms
                    ctx.orbital_sigma3(U, A_tilde, G_blocks, full_eigvecs[roots_to_check, :], w_sigma, num_to_build, 1)
                    for i, root_idx in enumerate(roots_to_check):
                        vec = full_eigvecs[root_idx, :]
                        w_vec = w_sigma[i, :] # Get the corresponding sigma
                        w_vec[1:] += reduced_gradient * vec[0] 
                        w_vec[0] = alpha * vec[0] + np.dot(vec[1:], reduced_gradient)
                # --- END FIX ---

                # Now, calculate residuals using the correct w_sigma
                for i, root_idx in enumerate(roots_to_check):
                    vec = full_eigvecs[root_idx, :]
                    w_vec = w_sigma[i, :] # This is (H*y + G*y)
                    r_vec = w_vec - theta[root_idx] * vec
                    w[root_idx, :] = r_vec # Store the *residual*
                    residual_norm[root_idx] = np.linalg.norm(r_vec)
                    
                    if residual_norm[root_idx] < threshold:
                        conv_status[root_idx] = True

            if root_0_locked:
                conv_status[0] = True # It's locked, so it's "converged"
 
           

            

            unconverged_idx = []
            exit_solver = False
            # --- NEW LSTRS CONVERGENCE CHECK (Main Loop) ---
            if nroots == 1:
                if conv_status[0] and not root_0_locked: # Root 0 *just* converged
                #if is_root_0_conv:
                    if ctx.check_root_0(full_eigvecs, theta[0], reduced_gradient, trust_radius):
                        # === EASY CASE ===
                        print("LSTRS: Root 0 converged (Easy Case). Exiting.")
                        aug_hessian_eigenvals[0] = theta[0]
                        aug_hessian_eigenvecs[:, 0] = full_eigvecs[0, :].T
                        aug_hessian_eigenvals[1] = 1e10 # Fake root 1
                        aug_hessian_eigenvecs[:, 1] = 0.0
                        exit_solver = True
                    else:
                        # === HARD CASE ===
                        root_0_locked = True
                        print("LSTRS: Root 0 converged (Hard Case). Checking Root 1.")
                        # *Now* we must check Root 1 for the *first time*
                        start1 = timer()
                        w1_sigma = np.zeros((1, H_dim))
                        # --- CORE BUG FIX (Hard Case Check) ---
                        if ctx.collapse_subspace_check == True:
                            # sigma_total *already* has gradient/alpha terms
                            w1_sigma = np.dot(aug_eigvecs.T[1:2, :], ctx.sigma_total[:L_new,:])
                        else:
                            ctx.orbital_sigma3(U, A_tilde, G_blocks, full_eigvecs[1:2, :], w1_sigma, 1, 1) # <-- SLICE FIX
                            w1_sigma[0, 1:] += reduced_gradient * full_eigvecs[1, 0] # Changed from einsum
                            w1_sigma[0, 0] = alpha * full_eigvecs[1, 0] + np.dot(full_eigvecs[1, 1:], reduced_gradient)
                        # --- END FIX ---
                        
                        r_1 = w1_sigma[0, :] - theta[1] * full_eigvecs[1, :]
                        w[1, :] = r_1 # Store residual
                        residual_norm[1] = np.linalg.norm(r_1)
                        
                        #if is_root_1_conv:
                        if residual_norm[1] < threshold:
                            conv_status[1] = True
                            print("LSTRS: Root 1 also converged. Exiting.")
                            aug_hessian_eigenvals[:] = theta[:nroots_target]
                            aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                            exit_solver = True
                        else:
                            print("LSTRS: Root 1 unconverged. Adding to search.")
                            unconverged_idx = [1]
                elif not conv_status[0]: # Root 0 still unconverged
                    unconverged_idx = [0]    
                elif root_0_locked: # Root 0 is locked, check Root 1
                    if not conv_status[1]:
                        unconverged_idx = [1]
                #else:
                #    # Root 0 is not converged
                #    unconverged_idx = [0]

            elif nroots == 2:
                #if is_root_0_conv and is_root_1_conv:
                #    # Both are converged
                #    print("LSTRS: Both roots converged. Exiting.")
                #    aug_hessian_eigenvals[:] = theta[:nroots_target]
                #    aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                #    exit_solver = True
                #else:
                # Find whichever is not converged
                if not conv_status[0]: unconverged_idx.append(0)
                if not conv_status[1]: unconverged_idx.append(1)
                #if not is_root_0_conv:
                #    unconverged_idx.append(0)
                #if not is_root_1_conv:
                #    unconverged_idx.append(1)
            # Check for final convergence
            if (nroots == 2 and not unconverged_idx) or \
               (nroots == 1 and root_0_locked and not unconverged_idx):
                print("LSTRS: All required roots converged.")
                exit_solver = True
                aug_hessian_eigenvals[:] = theta[:nroots_target]
                aug_hessian_eigenvecs[:, :] = full_eigvecs.T
    
            # --- END NEW LOGIC ---
            #print(unconverged_idx)
            #print("root", "AH residual norm", "Eigenvalue", "Convergence")
            #for j in range(nroots_target):
            #    print(
            #        j + 1,
            #        residual_norm[j],
            #        theta[j],
            #        convergence_check[j],
            #        flush=True,
            #    )
            print(unconverged_idx)
            print("root", "AH residual norm", "Eigenvalue", "Convergence")
            for j in range(nroots_target):
                print(
                    j + 1, residual_norm[j], theta[j], "Yes" if conv_status[j] else "No", flush=True
                )
             



            if exit_solver:
                aug_hessian_eigenvecs[:, :] = full_eigvecs.T
                aug_hessian_eigenvals[:] = theta[:nroots_target]
                if ctx.collapse_subspace_check == False:
                    Q1 = full_eigvecs
                    start1 = timer()
                    b_dim = L - indim
                    print("b_dim", b_dim)
                    print("indim", indim)

                    ctx.projected_augmented_hessian = copy.deepcopy(
                        projected_augmented_hessian
                    )
                    ctx.projected_augmented_hessian[0, 0] = alpha
                    if b_dim > 0:
                        Sq = np.zeros((b_dim, H_dim))
                        Sq[:, 0] = alpha * Q[indim:, 0]
                        ctx.sigma_total[indim:L,0] -= Sq[:, 0]
                        if indim > 0:
                            # H_qp = np.zeros((b_dim, indim))
                            for i in range(b_dim):
                                ctx.projected_augmented_hessian[indim + i][0] -= Sq[i][0]
                                ctx.projected_augmented_hessian[0][indim + i] -= Sq[i][0]
                                ctx.H_qp[i][0] -= Sq[i][0]
                                # for j in range(dim1):
                                #    #index1 = ctx.idx_hessian[j]
                                #    #H_qp[i][0] = Sq[i][0]
                                #    #H_qp[i][j+1] = Sq[i][index1+1]
                            H_qq = np.dot(Sq, Q[indim:, :].T)
                            ctx.projected_augmented_hessian[indim:, indim:] -= H_qq
                        else:
                            ctx.projected_augmented_hessian[indim:, indim:] = 0.0 

                    end1 = timer()
                    print("initialize took", end1 - start1)
                else:
                    Sq = np.zeros((L_new, H_dim))
                    Sq[:, 0] = alpha * Q[:, 0]
                    ctx.sigma_total[:L_new,0] -= Sq[:, 0]

                print("converged!", flush=True)
                break
            
            # --- NEW HYBRID LOGIC ---
            preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
            converged_indices = [i for i in range(nroots) if i not in unconverged_idx]
            
            if not converged_indices:
                # preconditioned_w = np.zeros((len(unconverged_idx),H_dim))
                if len(unconverged_idx) > 0:
                    preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
                    preconditioned_w = theta[unconverged_idx].reshape(
                        len(unconverged_idx), 1
                    ) - H_diag.reshape(1, H_dim)
                    # print(np.shape(preconditioned_w))
                    preconditioned_w = np.divide(
                        w[unconverged_idx],
                        preconditioned_w,
                        out=np.zeros_like(w[unconverged_idx]),
                        where=preconditioned_w != 0,
                    )


                
            
            else:
                # === PHASE 2: At least one root is converged ===
                # NOW use the robust (but tuned-down) Jacobi-Davidson solver
                # to prevent the 70-iteration stall for Root 1.
                
                Q_conv = full_eigvecs2[converged_indices, :]

                if _QED_SOLVER_STATS:
                    # How wide could a BLOCKED correction solve be here?  This
                    # branch is guarded by "at least one root converged", and
                    # nroots is 1 or 2, so the answer is expected to be 1 --
                    # i.e. there is nothing to block across.  Counted rather
                    # than assumed.
                    _solver_stats.jd_width(len(unconverged_idx))

                for i in range(len(unconverged_idx)):
                    idx = unconverged_idx[i]
                    current_residual = w[idx, :]
                    current_theta = theta[idx]
                    P_diag = H_diag - current_theta
                    # ... (handle P_diag safety) ...
                    
                    # Call the inner solver, but make sure its
                    # max_inner_iter is set to a low number, like 3 or 5.
                    v_corr = ctx.inner_solve_pcg(
                        current_residual,
                        current_theta,
                        Q_conv,
                        P_diag,
                        U, A_tilde, G_blocks, reduced_gradient, alpha # Need to pass these through
                    )
                    
                    preconditioned_w[i, :] = v_corr
            # --- END HYBRID LOGIC ---


            ## --- NEW JACOBI-DAVIDSON LOGIC ---
            #preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
            #converged_indices = [i for i in range(nroots) if i not in unconverged_idx]
            #Q_conv = full_eigvecs2[converged_indices, :] # Converged eigenvectors (N_conv, H_dim)
            #
            #for i in range(len(unconverged_idx)):
            #    idx = unconverged_idx[i] # The root we are solving for
            #    
            #    # Get inputs for the inner solver
            #    current_residual = w[idx, :]
            #    current_theta = theta[idx]
            #    
            #    # This is your *diagonal* preconditioner. We still use it
            #    # to precondition the *inner* solver.
            #    P_diag = H_diag - current_theta
            #    # (Handle divide by zero, e.g., set P_diag[P_diag == 0] = 1.0)
            #    
            #    # Call the inner solver to find the correction vector
            #    v_corr = ctx.inner_solve_pcg(
            #        current_residual,
            #        current_theta,
            #        Q_conv,
            #        P_diag,
            #        U, A_tilde, G1, reduced_gradient, alpha # Need to pass these through
            #    )
            #    
            #    preconditioned_w[i, :] = v_corr        


            
            
            ## preconditioned_w = np.zeros((len(unconverged_idx),H_dim))
            #if len(unconverged_idx) > 0:
            #    preconditioned_w = np.zeros((len(unconverged_idx), H_dim))
            #    preconditioned_w = theta[unconverged_idx].reshape(
            #        len(unconverged_idx), 1
            #    ) - H_diag.reshape(1, H_dim)
            #    # print(np.shape(preconditioned_w))
            #    preconditioned_w = np.divide(
            #        w[unconverged_idx],
            #        preconditioned_w,
            #        out=np.zeros_like(w[unconverged_idx]),
            #        where=preconditioned_w != 0,
            #    )

                
           
            end = timer()
            print("residual took", end - start)
            L_old = Q.shape[0]
            start = timer()

            if Lmax - L < len(unconverged_idx):
                if Lmax > 40:
                    num_restart_vecs = 40
                else:
                    num_restart_vecs = Lmax//2
                
                print(f"Subspace limit reached. Performing soft restart from {L} to {num_restart_vecs} vectors.")
                t_collapsing_begin = time.time()
                
                # 1. Choose how many vectors to keep.
                # This is a tunable parameter. Let's say 40.

                # 2. 'aug_eigvecs' holds the (L, L) eigenvectors of the projected Hessian.
                # 'theta' holds the eigenvalues. You already have these from np.linalg.eigh.
                
                # 3. Get the k "best" projected eigenvectors (the first k columns)
                # Note: np.linalg.eigh already sorts them by eigenvalue.
                best_projected_vecs = aug_eigvecs[:, :num_restart_vecs] # Shape (L, k)

                # 4. Transform these k vectors back into the full H_dim space.
                # Q is (L, H_dim). We need to project it down.
                # (k, L) @ (L, H_dim) -> (k, H_dim)
                Q_restarted = np.dot(best_projected_vecs.T, Q) # Shape (k, H_dim)
                
                # 5. Add your new correction vectors to this restarted subspace
                Q = np.concatenate((Q_restarted, preconditioned_w), axis=0) # Shape (k + n_unconv, H_dim)

                # 6. Re-orthogonalize the new, smaller subspace
                c_gram_schmidt_orthogonalization(
                   Q, Q.shape[0], H_dim
                )
                
                t_collapsing_end = time.time()
                print(
                    "restart took", t_collapsing_end - t_collapsing_begin, "seconds"
                )
                
                # 7. IMPORTANT: You MUST tell your code to rebuild the projected Hessian
                # from this new Q, as it's not just an expansion.
                # Your existing logic handles this perfectly.
                ctx.collapse_subspace_check = True 
                collapse = True
                L_old = 0 # Tell the expansion logic the "old" size is 0 

            else:
                t_expanding_begin = time.time()

                Q = np.concatenate((Q, preconditioned_w), axis=0)
                c_gram_schmidt_add(Q, L, H_dim, len(preconditioned_w))
                # gc.collect()
                t_expanding_end = time.time()
                print("expand took", t_expanding_end - t_expanding_begin, "seconds")
                collapse = False
                print("collapse subspace check", ctx.collapse_subspace_check)
                # print(Q)
            end = timer()

    _stats_end(_stats_span_d, status="warm" if restart else "cold")
    return Q
