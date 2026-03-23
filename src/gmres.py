import numpy as np

class GMRESSolver:
    def __init__(self, b_vector, max_subspace=60):
        """
        Args:
            b_vector: The RHS vector (your 'reduced_state_gradient')
            max_subspace: The restart dimension (size of Krylov space)
        """
        self.b = b_vector
        self.m = max_subspace 

    def solve(self, matvec_product, preconditioner=None, x0=None, max_iter=100, conv_thresh=1e-8):
        """
        Solves Ax = b.
        
        Args:
            matvec_product: Function x -> Ax
            preconditioner: Function r -> M^-1 r
            x0: Initial guess vector (optional). 
                If None, assumes x0=0 (so initial residual = b).
        
        Returns:
            x_final: The solution vector.
        """
        n = len(self.b)
        
        # --- OPTIMIZATION: Smart Initialization ---
        # If no x0 is provided, assume x0 = 0.
        # This means Residual = b - A(0) = b.
        # We skip the first expensive Matrix-Vector product.
        if x0 is None:
            x = np.zeros(n)
            r = self.b.copy() # r = b
        else:
            x = x0
            r = self.b - matvec_product(x)

        b_norm = np.linalg.norm(self.b)
        
        # Arnoldi storage
        V = np.zeros((self.m + 1, n)) 
        H = np.zeros((self.m + 1, self.m))
        
        print("--- Starting GMRES Solver ---", flush = True)

        for iteration in range(max_iter):
            # 1. Normalize residual
            r_norm = np.linalg.norm(r)
            print(f"Iter (Restart): {iteration+1:3d}   Residual Norm: {r_norm:.4e}", flush = True)
            
            if r_norm < conv_thresh:
                print("\n--- Convergence Achieved ---", flush = True)
                return x

            V[0] = r / r_norm
            s = np.zeros(self.m + 1)
            s[0] = r_norm
            
            # 2. Inner Arnoldi Loop (GMRES generates the vectors here)
            for k in range(self.m):
                # A. Precondition (Your 'trial_c = residual/denom' logic happens here)
                if preconditioner:
                    v_preconditioned = preconditioner(V[k])
                    w = matvec_product(v_preconditioned)
                else:
                    w = matvec_product(V[k])
                
                # B. Orthogonalize (Gram-Schmidt)
                for j in range(k + 1):
                    H[j, k] = np.dot(V[j], w)
                    w = w - H[j, k] * V[j]
                
                H[k + 1, k] = np.linalg.norm(w)
                
                # Check for breakdown
                if H[k + 1, k] < 1e-12:
                    return self._build_solution(k, H, s, V, x, preconditioner)
                    
                V[k + 1] = w / H[k + 1, k]
                
                # C. Solve Least Squares (Givens Rotations)
                self._apply_givens_rotation(H, s, k)
                
                # D. Check Convergence early
                if abs(s[k+1]) < conv_thresh:
                    print(f"   Convergence at Inner Iter: {k+1}", flush = True)
                    return self._build_solution(k + 1, H, s, V, x, preconditioner)

            # 3. Restart: Update x and recompute real residual
            x = self._build_solution(self.m, H, s, V, x, preconditioner)
            r = self.b - matvec_product(x)

        print("\n--- Max Iterations Reached ---", flush = True)
        return x

    def _apply_givens_rotation(self, H, s, k):
        # Apply previous rotations to new column
        for i in range(k):
            temp = H[i, k]
            H[i, k] = self.cs[i] * temp + self.sn[i] * H[i+1, k]
            H[i+1, k] = -self.sn[i] * temp + self.cs[i] * H[i+1, k]
        
        # Calculate new rotation
        a, b = H[k, k], H[k+1, k]
        hyp = np.hypot(a, b)
        cs, sn = a / hyp, b / hyp
        
        if not hasattr(self, 'cs'): self.cs = {}
        if not hasattr(self, 'sn'): self.sn = {}
        self.cs[k] = cs
        self.sn[k] = sn
        
        H[k, k] = cs * a + sn * b
        H[k+1, k] = 0.0
        s[k+1] = -sn * s[k]
        s[k] = cs * s[k]

    def _build_solution(self, k, H, s, V, x_current, preconditioner):
        # Solve upper triangular system
        y = np.linalg.solve(H[:k, :k], s[:k])
        
        # Construct update in Krylov basis
        update = np.zeros_like(x_current)
        for i in range(k):
            update += y[i] * V[i]
            
        # Apply preconditioner to the update (Right Preconditioning)
        if preconditioner:
            update = preconditioner(update)
            
        return x_current + update
