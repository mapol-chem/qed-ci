#!/usr/bin/python
import psi4
# some global variables
nirrep = None
nrdoccpi = None
nruoccpi = None
nactvpi = None
nmopi = None

"""
  This function makes (C1)^T S2 C1 orthogonal
  C1: converged CASSCF orbitals at geometry 1
  S2: SO overlap matrix at geometry 2
  return: orthogonal orbitals
"""
def ortho_orbs(wfn1, wfn2, semi = True):
    title = "\n  ==> Orthogonalize Orbitals Between Different Geometries <==\n"
    psi4.core.print_out(title)

    # make sure there is no frozen orbitals
    psi4.core.print_out("\n    Testing frozen orbitals ... ")
    global nirrep
    nirrep = wfn2.nirrep()
    nfdoccpi = psi4.core.Dimension.from_list(psi4.core.get_option("DETCI","FROZEN_DOCC"))
    nfuoccpi = psi4.core.Dimension.from_list(psi4.core.get_option("DETCI","FROZEN_UOCC"))
    nf = nfdoccpi.n() + nfuoccpi.n()
    if nf != 0:
        psi4.core.print_out("False")
        raise ValueError("I am too lazy to consider frozen orbitals.")
    else:
        psi4.core.print_out("Pass")

    # get C1 and S2
    C1 = wfn1.Ca()
    S2 = wfn2.S()

    # figure out irreps and orbital spaces
    global nmopi
    global nrdoccpi
    global nactvpi
    global nruoccpi
    nmopi = wfn2.nmopi()
    nrdoccpi = psi4.core.Dimension.from_list(psi4.core.get_option("DETCI","RESTRICTED_DOCC"))
    nactvpi = psi4.core.Dimension.from_list(psi4.core.get_option("DETCI","ACTIVE"))
    nruoccpi = psi4.core.Dimension(nirrep)
    for i in range(nirrep):
        nruoccpi[i] = nmopi[i] - nrdoccpi[i] - nactvpi[i]

    # create subspace orbitals: core, active, virtual
    psi4.core.print_out("\n    Preparing orbitals of subspaces ... ")
    Ccore = psi4.core.Matrix("C core", nmopi, nrdoccpi)
    Cactv = psi4.core.Matrix("C actv", nmopi, nactvpi)
    Cvirt = psi4.core.Matrix("C virt", nmopi, nruoccpi)
    print("ccore shape",Ccore.shape)
    print("cactv shape",Cactv.shape)
    print("cvirt shape",Cvirt.shape)
    # fill in data to orbitals of subspaces
    for h in range(nirrep):
        offset1 = nrdoccpi[h]
        offset2 = nactvpi[h] + offset1

        for i in range(nmopi[h]):
            # core
            for j in range(nrdoccpi[h]):
                Ccore.set(h, i, j, C1.get(h, i, j))

            # active
            for j in range(nactvpi[h]):
                Cactv.set(h, i, j, C1.get(h, i, j + offset1))

            # virtual
            for j in range(nruoccpi[h]):
                Cvirt.set(h, i, j, C1.get(h, i, j + offset2))
    psi4.core.print_out("Done")

    # orthogonalize core
    psi4.core.print_out("\n    Orthogonalizing orbitals of subspaces ... ")
    Ccore = ortho_subspace(Ccore, S2)
    
    # orthogonalize active
    Cactv = projectout(Cactv, Ccore, S2)
    Cactv = ortho_subspace(Cactv, S2)

    # orthogonalize virtual
    Cvirt = projectout(Cvirt, Ccore, S2)
    Cvirt = projectout(Cvirt, Cactv, S2)
    Cvirt = ortho_subspace(Cvirt, S2)
    psi4.core.print_out("Done")

    # fill in data to the new combined orbitals
    psi4.core.print_out("\n    Combining orbitals of subspaces ... ")
    Cnew = psi4.core.Matrix("new C", C1.rowdim(), C1.coldim())
    for h in range(nirrep):
        offset1 = nrdoccpi[h]
        offset2 = nactvpi[h] + offset1

        for i in range(nmopi[h]):
            # core
            for j in range(nrdoccpi[h]):
                Cnew.set(h, i, j, Ccore.get(h, i, j))

            # active
            for j in range(nactvpi[h]):
                Cnew.set(h, i, j + offset1, Cactv.get(h, i, j))

            # virtual
            for j in range(nruoccpi[h]):
                Cnew.set(h, i, j + offset2, Cvirt.get(h, i, j))
    psi4.core.print_out("Done")

    if semi:
        psi4.core.print_out("\n    Semicanonicalizing orbitals ... ")
        U = semicanonicalize(wfn2.Fa(), Cnew)
        Cnew = psi4.core.doublet(Cnew, U, False, False)
        psi4.core.print_out("Done")

    psi4.core.print_out("\n\n")
    return Cnew

"""
  This function project CP out of C
  C: orbitals to be projected by P
  CP: orbitals of the projector
  S: SO overlap matrix
  return: Cp = (1 - P) C = C - CP (CP^T S C)
"""
def projectout(C, CP, S):
    M = psi4.core.triplet(CP, S, C, True, False, False)
    P = psi4.core.doublet(CP, M, False, False)
    Cp = C.clone()
    Cp.subtract(P)
    return Cp

"""
  This function orthogonalize C
  C: the orbitals of a subspace at geometry 1
  S: the SO overlap matrix at geometry 2
  return: C X where X is the canonical orthogonalizing transformation matrix
"""
def ortho_subspace(C, S):
    M = None
    try:
        M = psi4.core.triplet(C, S, C, True, False, False)
    except ValueError as e:
        print("The dimensions are wrong. Cannot do Matrix::triplet.")
    X = canonicalX(M)

    Cnew = psi4.core.doublet(C, X, False, False)
    return Cnew

"""
  This function returns the canonical orthogonalizing transformation matrix
  S: overlap matrix
  return: X = U s^(-1/2)
"""
def canonicalX(S):
    rdim = S.rowdim()
    evals = psi4.core.Vector("evals", rdim)
    evecs = psi4.core.Matrix("evecs", rdim, rdim)

    S.diagonalize(evecs, evals, psi4.core.DiagonalizeOrder.Descending)
    shalf_inv = psi4.core.Matrix("s^(-1/2)", rdim, rdim)
    for h in range(nirrep):
        for i in range(rdim[h]):
            shalf_inv.set(h, i, i, evals.get(h, i) ** -0.5)

    X = psi4.core.doublet(evecs, shalf_inv, False, False)
    return X

"""
  This function semicanonicalize the orbitals
  Fso: SO Fock matrix at geometry 2
  C: molecular orbitals that transforms Fso to Fmo
  return: unitary matrix that transforms orbitals to semicanonical orbitals
"""
def semicanonicalize(Fso, C):
    # transform SO Fock to MO Fock
    Fmo = psi4.core.triplet(C, Fso, C, True, False, False)

    offsets = psi4.core.Dimension.from_list([0 * i for i in range(nirrep)])

    U = psi4.core.Matrix("U to semi", nmopi, nmopi)

    # diagonalize each blcok of Fmo
    for block in [nrdoccpi,nactvpi,nruoccpi]:
        F = psi4.core.Matrix("Fock",block,block)
        for h in range(nirrep):
            offset = offsets[h]
            for i in range(block[h]):
                for j in range(block[h]):
                    F.set(h, i, j, Fmo.get(h, i + offset, j + offset))
      
        evals = psi4.core.Vector("F Evals", block)
        evecs = psi4.core.Matrix("F Evecs", block, block)
        F.diagonalize(evecs, evals, psi4.core.DiagonalizeOrder.Ascending)
      
        for h in range(nirrep):
            offset = offsets[h]
            for i in range(block[h]):
                for j in range(block[h]):
                    U.set(h, i + offset, j + offset, evecs.get(h, i, j))
            offsets[h] += block[h] ### important ###

    return U

