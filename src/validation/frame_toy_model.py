#!/usr/bin/env python3
"""Why transforming H is not enough: an exactly solvable frame demonstration.

Needs nothing but numpy/scipy -- no psi4, no CASSCF -- and answers the natural
objection to the photon frame term:

    "The coherent-state transformation acts on the Hamiltonian, and I
     diagonalise the transformed Hamiltonian.  Why does the transformation
     still have to act on the eigenfunction?"

Because a unitary transformation leaves eigenVALUES alone but moves
eigenVECTORS.  If H_CS = U+ H U then

    H |Psi> = E |Psi>   ==>   H_CS (U+|Psi>) = E (U+|Psi>)

so diagonalising H_CS returns Psi_CS = U+ Psi_lab: the SAME physical state
expressed in the displaced photon basis, with the same energy but a different
component vector.  Recovering the physical state means applying U.  This only
has consequences because U depends on geometry -- a constant U would drop out
of <Psi_m|d/dx|Psi_n> entirely, since U+ dU/dx = 0.

CONVENTIONS ARE THE PAPER'S:  H_CS = U H_PF U+  (U on the LEFT) with
U = exp[z(b+ - b)] and z = -<d>/sqrt(2w).  Both differ from the commoner
choices, and they CANCEL in the final formula -- so getting one right and the
other wrong flips the sign of the whole frame term.  With these,
Psi_lab = U+ Psi_CS, which this file checks explicitly.

THE MODEL.  Strip out the electrons and let the dipole be a c-number that
moves with the nuclear coordinate:

    H(x) = w b+b - sqrt(w/2) d(x) (b + b+) + d(x)^2 / 2

Completing the square gives exact eigenstates D(alpha(x))|n> with
alpha = d/sqrt(2w) and energies n*w -- INDEPENDENT of x.  In the coherent-state
frame H_CS = U+ H U = w b+b exactly, which is x-independent, so its
eigenvectors are the bare |n> at every geometry and the CS-frame coupling is
IDENTICALLY ZERO.  The physical coupling is not:

    d_lab(0,1) = -d(alpha)/dx = (dz/dx) <0|b+ - b|1>

Run this file to see all of it verified numerically.

WHAT IT DOES AND DOES NOT SHOW.  It is a demonstration, not the production
path: with electrons, H_CS genuinely does depend on x, so d_CS is nonzero and
carries a real part of the coupling.  What the model isolates is the piece that
comes from the FRAME moving, which is missing from any calculation done
entirely inside that frame.  On the real h2o_cav case that piece is -0.899 --
larger than the working-frame coupling itself.  See NUMERICAL_NAC_METHOD.md.
"""
import numpy as np
from scipy.linalg import expm

W, NFOCK, H_STEP = 0.12, 40, 1e-4


def operators(n=NFOCK):
    b = np.diag(np.sqrt(np.arange(1, n)), 1)
    return b, b.T, b.T @ b


def hamiltonian(d, b, bd, num):
    return W * num - np.sqrt(W / 2) * d * (b + bd) + 0.5 * d * d * np.eye(len(b))


def main():
    b, bd, num = operators()
    d_of_x = lambda x: 0.30 + 0.50 * x           # the dipole moves with x
    alpha = lambda d: d / np.sqrt(2 * W)

    print("1. The CS-frame Hamiltonian is x-INDEPENDENT here")
    print("   (paper conventions: H_CS = U H_PF U+, z = -<d>/sqrt(2w)):")
    for x in (0.0, 0.2, 0.5):
        d = d_of_x(x)
        U = expm(-alpha(d) * (bd - b))          # z = -<d>/sqrt(2w)
        H_cs = U @ hamiltonian(d, b, bd, num) @ U.conj().T
        print(f"     x={x:<4} || U H(x) U+ - w b+b || = "
              f"{np.abs(H_cs - W * num)[:8, :8].max():.2e}")
    print("   so its eigenvectors are the bare |n> at every x, and therefore")
    print("   d_CS(0,1) = <0| d/dx |1> = 0  IDENTICALLY.\n")

    d0 = d_of_x(0.0)
    U0 = expm(-alpha(d0) * (bd - b))
    _, vchk = np.linalg.eigh(hamiltonian(d0, b, bd, num))
    ket = np.zeros(len(b)); ket[0] = 1.0
    print("   and Psi_lab = U+ Psi_CS, not U Psi_CS:")
    print(f"     |<H_PF ground | U+ |0>| = {abs(vchk[:,0] @ (U0.conj().T @ ket)):.10f}")
    print(f"     |<H_PF ground | U  |0>| = {abs(vchk[:,0] @ (U0 @ ket)):.10f}\n")

    def lab(x):
        return np.linalg.eigh(hamiltonian(d_of_x(x), b, bd, num))

    e0, v0 = lab(0.0)
    _, vp = lab(+H_STEP)
    _, vm = lab(-H_STEP)
    for V in (vp, vm):                            # phase-align to the reference
        for k in range(3):
            if v0[:, k] @ V[:, k] < 0:
                V[:, k] *= -1

    d_lab = (v0[:, 0] @ vp[:, 1] - v0[:, 0] @ vm[:, 1]) / (2 * H_STEP)
    # alpha = <d>/sqrt(2w) = -z in the paper's convention, so the frame term
    # -(dz/dx) G is +(d alpha/dx) G.
    dz_dx = (alpha(d_of_x(H_STEP)) - alpha(d_of_x(-H_STEP))) / (2 * H_STEP)
    pred = dz_dx * (v0[:, 0] @ (bd - b) @ v0[:, 1])

    print("2. The LAB-frame states do move, and their coupling is not zero:")
    print(f"     energies of H(x)                : {e0[:3]}  -> n*w, x-independent")
    print(f"     d_lab(0,1), finite difference   : {d_lab:+.8f}")
    print(f"     -(dz/dx) <0|b+ - b|1>           : {pred:+.8f}")
    print(f"       = (1/sqrt(2w)) d<d>/dx * G      (paper's z = -<d>/sqrt(2w))")
    print(f"     -d(alpha)/dx                    : {-dz_dx:+.8f}")
    print(f"     residual                        : {abs(d_lab - pred):.2e}\n")
    print("3. Consequence: a calculation done entirely in the CS frame reports")
    print("   ZERO coupling here, so dynamics driven by it would predict that a")
    print("   moving nucleus never excites the field.  The energies are right in")
    print("   both frames -- eigenvalues are invariant -- which is why gradients,")
    print("   g-vectors and MECPs are unaffected and only the NAC is.")
    return 0 if abs(d_lab - pred) < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
