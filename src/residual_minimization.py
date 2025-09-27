# -*- coding: utf-8 -*-
"""Python Implementation of LinearRM Solver

"""

import numpy as np
from collections import deque
import sys
#np.set_printoptions(threshold=sys.maxsize)
class LinearRMSolver:
    """
    A Python implementation of the Linear Residual Minimization (LinearRM)
    iterative solver, based on the algorithm in BAGEL's `linearRM.h`.
    """

    def __init__(self, b_vector, max_subspace=10):
        """
        Initializes the LinearRM solver.

        Args:
            b_vector (np.ndarray): The constant right-hand-side vector 'b' of the
                                   linear system. The solver solves Ax + b = 0.
            max_subspace (int): The maximum number of vectors to store in the
                                iterative subspace.
        """
        if max_subspace < 1:
            raise ValueError("max_subspace must be 1 or greater.")

        self.b = b_vector
        self.max_subspace = max_subspace
        self._reset()

    def _reset(self):
        """Resets the solver's internal state."""
        # Use standard lists to easily manage the subspace transformation
        self.c_vectors = []
        self.s_vectors = []

    def _canonical_orthogonalization(self, S, thresh=1e-9):
        """
        Performs canonical orthogonalization, explicitly removing linear dependencies.
        """
        try:
            eigvals, eigvecs = np.linalg.eigh(S)
            non_redundant_indices = np.where(eigvals > thresh)[0]

            if len(non_redundant_indices) == 0:
                print("Warning: Subspace is fully linearly dependent.")
                return np.zeros((S.shape[0], 0))

            U = eigvecs[:, non_redundant_indices]
            inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals[non_redundant_indices])
            X = U * inv_sqrt_eigvals
            return X

        except np.linalg.LinAlgError:
            print("Warning: Canonical orthogonalization failed during eigh.")
            return np.identity(S.shape[0])

    def update_subspace_and_extrapolate(self, c_new, s_new):
        """
        Performs one update step of the LinearRM algorithm. 
        """
        if len(self.c_vectors) == self.max_subspace:
            # Drop the second-oldest vector, keeping the first (best) one
            self.c_vectors.pop(1)
            self.s_vectors.pop(1)
        self.c_vectors.append(c_new)
        self.s_vectors.append(s_new)
        size = len(self.c_vectors)

        # Build the small matrices from scratch
        mat = np.zeros((size, size))
        overlap = np.zeros((size, size))
        prod = np.zeros(size)

        for i in range(size):
            prod[i] = -np.dot(self.s_vectors[i], self.b)
            for j in range(i, size):
                mat[i, j] = mat[j, i] = np.dot(self.s_vectors[i], self.s_vectors[j])
                overlap[i, j] = overlap[j, i] = np.dot(self.c_vectors[i], self.c_vectors[j])
        ## Add this block for debugging
        #print("    PYTHON DEBUG: Overlap Matrix (size {}x{}):".format(size, size))
        #for row in overlap:
        #    print("      [" + ", ".join(["{:.4e}".format(x) for x in row]) + "]")
        ## End of debug block

        transform_matrix = self._canonical_orthogonalization(overlap)

        if transform_matrix.shape[1] == 0:
            return self.b + s_new

        mat_ortho = transform_matrix.T @ mat @ transform_matrix
        prod_ortho = transform_matrix.T @ prod

        try:
            coeffs_ortho = np.linalg.solve(mat_ortho, prod_ortho)
        except np.linalg.LinAlgError:
            print("Warning: Small RM system is singular. Cannot extrapolate.")
            return self.b + s_new
            
        # Back-transform coefficients to the original, non-orthogonal basis
        coeffs = transform_matrix @ coeffs_ortho
        # --- CRITICAL FIX: Correctly update the subspace state ---
        # Build the transformation matrix T for the subspace basis.
        # The new basis is {c_optimal, c_1, c_2, ...}, where c_optimal is a
        # linear combination of the old basis vectors.
        T = np.identity(size)
        T[:, 0] = coeffs

        # Update the basis vectors themselves by applying the transformation
        old_c_matrix = np.array(self.c_vectors).T
        old_s_matrix = np.array(self.s_vectors).T

        new_c_matrix = old_c_matrix @ T
        new_s_matrix = old_s_matrix @ T

        self.c_vectors = [v for v in new_c_matrix.T]
        self.s_vectors = [v for v in new_s_matrix.T]

        # The new optimal sigma vector is the first in the transformed basis
        s_optimal = self.s_vectors[0]

        return self.b + s_optimal

    def get_solution(self):
        """Returns the current best solution vector from the subspace."""
        if not self.c_vectors:
            return np.zeros(self.b.shape)
        # The optimal solution is the first vector in the list after extrapolation
        return self.c_vectors[0]

    def solve(self, matvec_product, preconditioner, max_iter=100, conv_thresh=1e-8):
        """
        Main driver loop to solve the linear system Ax + b = 0.
        """
        self._reset()
        residual = self.b.copy()

        print("--- Starting LinearRM Solver ---")
        for i in range(max_iter):
            residual_norm = np.linalg.norm(residual)
            print(f"Iter: {i+1:3d}   Residual Norm: {residual_norm:.4e}")

            if residual_norm < conv_thresh:
                print("\n--- Convergence Achieved ---")
                return self.get_solution()

            trial_c = preconditioner(residual)
            norm = np.linalg.norm(trial_c)
            if norm > 1e-12:
                trial_c /= norm

            sigma = matvec_product(trial_c)
            residual = self.update_subspace_and_extrapolate(trial_c, sigma)

        print("\n--- Solver did not converge within max iterations ---")
        return self.get_solution()

