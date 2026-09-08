"""Retired pure-Python CI kernels and superseded solvers.  NOTHING IMPORTS THIS.

QUARANTINE, NOT A LIBRARY.  Every function here was proved unreachable from the
live code before it was moved (2026-09-04), and is kept only so the Python
reference implementation of each algorithm stays readable next to the C that
replaced it.  Do not add a caller.  If something here turns out to be needed,
the fix is to establish why, restore it deliberately, and gate that restoration.

HOW "UNREACHABLE" WAS ESTABLISHED -- all three steps, not just the first:
  1. a static call-graph pass over helper_PFCI.py, lstrs_trust_region.py,
     davidson_augmented_hessian.py and otr_trust_region.py: no live caller for
     any of them, only commented-out references;
  2. every one wrapped with a counter and the FULL PYTEST SUITE run (136 passed
     / 2 known failures): all recorded ZERO calls;
  3. the same probe against both real SA-QED-CASSCF drivers
     (validation/h2e_disk_case.py, validation/internal_opt_case.py): again zero,
     while the live neighbours they sat beside recorded hundreds
     (orbital_sigma3 685-837, mv2 93-228, Davidson_augmented_hessian_solve6
     44-46).  Step 3 is not redundant: the suite alone UNDER-REPORTS the solver,
     and several live solver methods also score zero there.

WHAT SUPERSEDED WHAT.  ci_solver.c / orbital.c carry an equivalent or improved
implementation of nearly all of this, reached through helper_PFCI's c_* ctypes
wrappers:

    graph, arc_weight                 -> get_graph
    string_to_binary, string_to_index,
      binary_to_index, index_to_string -> index_to_string, get_string,
                                          string_to_obtlist
    phase_single_excitation,
      single_replacement_list2         -> single_replacement_list
    one_e_contraction, two_e_contraction,
      constant_terms_contraction       -> constant_terms_contraction
    sigma3, sigma12, sigma_dipole,
      build_sigma                      -> the same four, driven by c_sigma
    build_H_diag                       -> build_H_diag, build_H_diag_cas
    Davidson                           -> davidson, davidson_spin (and, for
                                          CIS, np.linalg.eigh -- see below)

`check_total_spin` is deliberately NOT here.  It is live (116 calls in the
suite) and already C-backed: its body is six lines of glue around
c_sigma_s_square, so there is no pure-Python kernel left in it to retire.

WHY `Davidson` IS HERE -- the one non-obvious case.  It was not dead before
2026-09-04: it was how `ci_level="cis"` was solved whenever
`full_diagonalization` was left at its default False, and four tests went
through it.  But it was not a direct solver either.  For CIS its sigma build
fell through to a dense ``S = np.einsum("ip,pq->iq", Q, H)`` against
``self.H_PF`` -- the matrix that had already been built and stored a few lines
earlier -- so it was a subspace iteration converging to what ``eigh`` returns
exactly, on a matrix that was resident either way.  The ``build_sigma``
callback it was handed was never invoked: that call had been commented out
(``# build_sigma(Q,S,H_dim)``), and THAT one dead line is what made this file's
entire CI half unreachable.  CIS now goes to full diagonalization (helper_PFCI,
``if self.full_diagonalization or self.ci_level == "cis"``), verified to move no
root by more than 2.3e-13 -- see validation/cis_full_diagonalization_check.py.

THE SUPERSEDED SOLVERS at the end are a different case: never wrong, just
replaced.  `build_sigma_reduced`/`2`/`3`/`4` and `orbital_sigma` are earlier
generations of the orbital Hessian-vector product now done by
`build_sigma_reduced7` / `orbital_sigma3`.  `solve_pcg_trust_region` /
`solve_pcg_with_operator` are the Steihaug-Toint trust-region solver, kept
rather than deleted because the C++ port's `PcgTrustRegionSolver` ("fully
ported, tested") cites them as its source.

ADAPTED, NOT BYTE-VERBATIM.  This is where these differ from the otr_ / lstrs_ /
davidson_augmented_hessian_ extractions, which are provable no-ops.  Because
this code is dead, NO TEST EXERCISES THE COPIES BELOW; treat them as reference
text, not as working code:
  * `self` became `ctx`, and each is now a module-level function taking the
    PFHamiltonianGenerator first;
  * calls between them became module-local calls threading `ctx` through;
  * `build_sigma_reduced4` lost its bare `@staticmethod`;
  * two docstrings became raw strings.  They contain LaTeX (`\\ket`) and had
    been emitting `SyntaxWarning: invalid escape sequence '\\k'` on every cold
    import of helper_PFCI.py -- which is what made the refactor gate flaky
    enough to need `PYTHONWARNINGS=ignore::SyntaxWarning`.  Moving them here
    removes both warnings from helper_PFCI.py outright.
What IS proven is the REMOVAL: with all of this gone, the suite is unchanged at
136 passed / 2 known failures, and validation/lstrs_refactor_gate.sh is 12/12
bit-identical.
"""

import time

import numpy as np
import psutil


def _helper():
    """Deferred access to helper_PFCI's module globals.

    helper_PFCI does not import this module and must never need to, so there is
    no import cycle to dodge.  This is deferred purely so that importing the
    quarantine does not drag in psi4 and the ctypes backend as a side effect.
    """
    import helper_PFCI
    return helper_PFCI


def graph(ctx, N, n_o):
    """
    lexical ordering graph with unoccupied arc set to be zero, return vertex weight
    """
    rows, cols = ((N + 1) * (n_o - N + 1), 3)
    rows1, cols1 = (N + 1, n_o + 1)
    graph = [[0 for i in range(cols)] for j in range(rows)]
    graph_big = [[0 for i in range(cols1)] for j in range(rows1)]
    graph_big[N][n_o] = 1
    # weight of vertex
    for e in range(N, -1, -1):
        for o in range(n_o - 1, -1, -1):
            # print(e,o)
            if e == N and o >= e:
                graph_big[e][o] = graph_big[e][o + 1]
            elif e <= N and o < e:
                graph_big[e][o] = 0
            else:
                graph_big[e][o] = graph_big[e + 1][o + 1] + graph_big[e][o + 1]

    count = 0
    for e in range(0, N + 1):
        for o in range(0, n_o + 1):
            # print(e,o,graph_big[e][o])
            if graph_big[e][o] != 0:
                graph[count][0] = e
                graph[count][1] = o
                graph[count][2] = graph_big[e][o]
                count += 1

    return graph, graph_big


def arc_weight(ctx, graph, graph_big, N, n_o):
    """
    return diagonal (occupied) arc weight for lexical ordering graph above
    Y=arc weight
    """
    Y = []
    for row in range(0, len(graph)):
        # print(graph[i])
        e = graph[row][0]
        o = graph[row][1]
        B = []
        if e == N:
            continue
        i = o - e
        c = 0
        if i == 0:
            c = 0
            B.extend([e, o, c])
            # print(B)
        else:
            for j in range(1, i + 1):
                c += graph_big[e + 1][o + 2 - j]
            B.extend([e, o, c])
            # print(B)
        Y.append(B)

    return Y


def string_to_binary(ctx, string, n_o):
    b = [int(d) for d in str(bin(string))[2:]]
    if len(b) < n_o:
        c = [0] * (n_o - len(b))
        d = c + b
    else:
        d = b
    d.reverse()
    return d


def string_to_index(ctx, string, N, n_o, Y):
    """
    return index using arc weight for a string(integer represents binary string)
    """
    count = 0
    index = 0  # counting from 0 so that determinant index can be counted from 0
    b = [int(d) for d in str(bin(string))[2:]]
    if len(b) < n_o:
        c = [0] * (n_o - len(b))
        d = c + b
    else:
        d = b
    d.reverse()
    for i in range(n_o):
        if d[i] == 1:
            e = count
            o = i
            for j in range(0, len(Y)):
                if Y[j][0] == e and Y[j][1] == o:
                    index += Y[j][2]
            count += 1
    return index


def binary_to_index(ctx, binary_string, N, n_o, Y):
    count = 0
    index = 0  # count from 0
    string = 0
    for i in range(0, len(binary_string)):
        if binary_string[i] == 1:
            e = count
            o = i
            string += pow(2, i)
            for j in range(0, len(Y)):
                if Y[j][0] == e and Y[j][1] == o:
                    index += Y[j][2]
            count += 1
    # print(string)
    return index, string


def index_to_string(ctx, index, N, n_o, Y, return_binary=False):
    """
    return string for an index using graph
    """
    I = 0  # count from 0 in stead of 1 so that determinant index can be counted from 0
    e = N
    o = n_o
    count = 0
    record = []
    while I <= index and e <= o:
        # print(count,e,o,I,record)
        if e == o and I < index:
            count2 = 0
            for i in range(len(record) - 1, -1, -1):
                if record[i] == 0:
                    count2 += 1
                    record.pop(i)
                else:
                    record[i] = 0
                    break
            o = o + count2
            e = e + 1
            for j in range(0, len(Y)):
                if Y[j][0] == e - 1 and Y[j][1] == o:
                    b = Y[j][2]
            I = I - b
            # print(record,o,e,b,I)

        else:
            if e > 0:
                for j in range(0, len(Y)):
                    if Y[j][0] == e - 1 and Y[j][1] == o - 1:
                        a = Y[j][2]
                        if a <= index - I:
                            e = e - 1
                            o = o - 1
                            I = I + a
                            record.append(1)
                        else:
                            o = o - 1
                            record.append(0)
            else:
                o = o - 1
                record.append(0)
        count += 1
        if count == 15000000 or (e == 0 and o == 0 and I == index):
            break
    string = 0
    for i in range(len(record)):
        if record[i] == 1:
            string += pow(2, len(record) - i - 1)
    # print(string)
    # print(record)

    if return_binary == True:
        return record, string
    else:
        return string


def phase_single_excitation(ctx, p, q, string):
    r"""getting the phase(-1 or 1) for E_pq\ket{I_a} where I_a is an alpha or beta string
    p=particle,q=hole
    """
    if p > q:
        mask = (1 << p) - (1 << (q + 1))
    else:
        mask = (1 << q) - (1 << (p + 1))
    if bin(mask & string).count("1") % 2:
        return -1
    else:
        return 1


def single_replacement_list2(ctx, num_strings, N, n_o, n_in_a, Y):
    r"""getting the sign, string address and pq for sign(pq)E_pq\ket{I_a}
    N = number of active alpha electrons
    n_o = number of active alpha orbitals
    n_in_a = number of inactive alpha orbitals
    p=particle, q=hole
    """
    rows, cols = (num_strings * (N * (n_o - N) + N + n_in_a), 4)
    ctx.num_links = N * (n_o - N) + N + n_in_a
    table = [[0 for i in range(cols)] for j in range(rows)]
    count = 0
    for index in range(0, num_strings):
        string = index_to_string(ctx, index, N, n_o, Y)
        # d = string_to_binary(ctx, string,n_o)
        # print('single replacement list for binary string',d,'string',string,'index',index)
        occ = []
        vir = []
        # print(occ,vir)
        for i in range(n_o):
            if string & (1 << i):
                occ.append(i)
            else:
                vir.append(i)
        if n_in_a > 0:
            for i in range(n_in_a):
                table[count][0] = index
                table[count][1] = 1
                table[count][2] = i
                table[count][3] = i
                count += 1

        for i in range(N):
            table[count][0] = index
            table[count][1] = 1
            table[count][2] = occ[i] + n_in_a
            table[count][3] = occ[i] + n_in_a
            count += 1
        for i in range(N):
            for a in range(n_o - N):
                string1 = (string ^ (1 << occ[i])) | (1 << vir[a])
                # c=string_to_binary(string1,n_o)
                table[count][0] = string_to_index(ctx, string1, N, n_o, Y)
                table[count][1] = phase_single_excitation(ctx, 
                    vir[a], occ[i], string
                )
                table[count][2] = vir[a] + n_in_a
                table[count][3] = occ[i] + n_in_a
                count += 1
    return table


def one_e_contraction(
    ctx, h1e, c_vectors, c1_vectors, num_strings, num_links, scale
):
    for indexa in range(0, num_strings):
        stride = indexa * num_links
        # print('indexa',indexa,'stride',stride)
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            # print(index1,sign,a,i)
            tmp = sign * h1e[a, i]
            for indexb in range(0, num_strings):
                # composite index of determinant
                index_I = indexa * num_strings + indexb
                index_J = index1 * num_strings + indexb
                c1_vectors[index_I] += scale * tmp * c_vectors[index_J]
    for indexb in range(0, num_strings):
        stride = indexb * num_links
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            tmp = sign * h1e[a, i]
            for indexa in range(0, num_strings):
                index_I = indexa * num_strings + indexb
                index_J = indexa * num_strings + index1
                c1_vectors[index_I] += scale * tmp * c_vectors[index_J]


def two_e_contraction(ctx, h2e, c_vectors, c1_vectors, num_strings, num_links):
    # build intermediate D^K_kl = \gamma^KJ_kl * c_J
    D = np.zeros((ctx.num_det, ctx.nmo * ctx.nmo))
    for indexa in range(0, num_strings):
        stride = indexa * num_links
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            for indexb in range(0, num_strings):
                # composite index of determinant
                index_I = indexa * num_strings + indexb
                index_J = index1 * num_strings + indexb
                ai = a * ctx.nmo + i
                D[index_I][ai] = sign * c_vectors[index_J]
    for indexb in range(0, num_strings):
        stride = indexb * num_links
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            for indexa in range(0, num_strings):
                index_I = indexa * num_strings + indexb
                index_J = indexa * num_strings + index1
                ai = a * ctx.nmo + i
                D[index_I][ai] += sign * c_vectors[index_J]

    # build intermediate E^K_ij = D^K_kl * (ij|kl)
    E = np.einsum("pr,kr->kp", h2e, D)

    # build \sigma_I = 0.5 * \gamma^IK_ij * E^K_ij
    for indexa in range(0, num_strings):
        stride = indexa * num_links
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            for indexb in range(0, num_strings):
                # composite index of determinant
                index_I = indexa * num_strings + indexb
                index_J = index1 * num_strings + indexb
                ai = a * ctx.nmo + i
                c1_vectors[index_I] += 0.5 * sign * E[index_J][ai]
    for indexb in range(0, num_strings):
        stride = indexb * num_links
        for excitation in range(num_links):
            index1 = ctx.table[stride + excitation][0]
            sign = ctx.table[stride + excitation][1]
            a = ctx.table[stride + excitation][2]
            i = ctx.table[stride + excitation][3]
            for indexa in range(0, num_strings):
                index_I = indexa * num_strings + indexb
                index_J = indexa * num_strings + index1
                ai = a * ctx.nmo + i
                c1_vectors[index_I] += 0.5 * sign * E[index_J][ai]


def constant_terms_contraction(ctx, c_vectors, c1_vectors, A):
    for i in range(len(c_vectors)):
        c1_vectors[i] += A * c_vectors[i]


def sigma3(ctx, h2e, c_vectors, c1_vectors, num_strings, num_links):
    for k in range(ctx.nmo):
        for l in range(ctx.nmo):
            kl = k * ctx.nmo + l
            # print('l','r','sgn')
            L = []
            R = []
            sgn = []
            dim = 0
            for N in range(len(ctx.table)):
                if ctx.table[N][2] == k and ctx.table[N][3] == l:
                    index = N // num_links
                    L.append(ctx.table[N][0])
                    R.append(index)
                    sgn.append(ctx.table[N][1])
                    dim += 1
            # print(L,R,sgn,dim)
            cp = np.zeros((dim, num_strings))
            for I in range(dim):
                for index_jb in range(0, num_strings):
                    index_ljb = L[I] * num_strings + index_jb
                    cp[I][index_jb] = c_vectors[index_ljb] * sgn[I]
            for index_ib in range(0, num_strings):
                F = np.zeros((num_strings))
                stride = index_ib * num_links
                for excitation in range(num_links):
                    index_jb = ctx.table[stride + excitation][0]
                    sign = ctx.table[stride + excitation][1]
                    i = ctx.table[stride + excitation][2]
                    j = ctx.table[stride + excitation][3]
                    ij = i * ctx.nmo + j
                    F[index_jb] += sign * h2e[ij][kl]
                v = np.zeros((dim))
                v = np.einsum("pq,q->p", cp, F)
                for I in range(dim):
                    index_I = R[I] * num_strings + index_ib
                    c1_vectors[index_I] += v[I]


def sigma12(ctx, h1e, h2e, c_vectors, c1_vectors, num_strings, num_links):
    for index_ib in range(0, num_strings):
        F = np.zeros((num_strings))
        stride1 = index_ib * num_links
        # print('index_ib',index_ib,'stride',stride1)
        for excitation1 in range(num_links):
            index_kb = ctx.table[stride1 + excitation1][0]
            sign1 = ctx.table[stride1 + excitation1][1]
            k = ctx.table[stride1 + excitation1][2]
            l = ctx.table[stride1 + excitation1][3]
            # print(index_kb,sign1,k,l)
            kl = k * ctx.nmo + l
            F[index_kb] += sign1 * h1e[k, l]
            stride2 = index_kb * num_links
            for excitation2 in range(num_links):
                index_jb = ctx.table[stride2 + excitation2][0]
                sign2 = ctx.table[stride2 + excitation2][1]
                i = ctx.table[stride2 + excitation2][2]
                j = ctx.table[stride2 + excitation2][3]
                ij = i * ctx.nmo + j
                if ij >= kl:
                    F[index_jb] += (sign1 * sign2 * h2e[ij][kl]) / (1 + (ij == kl))
        for index_jb in range(0, num_strings):
            for index_ia in range(0, num_strings):
                # composite index of determinant
                index_I = index_ia * num_strings + index_ib
                index_J = index_ia * num_strings + index_jb
                c1_vectors[index_I] += F[index_jb] * c_vectors[index_J]
    for index_ia in range(0, num_strings):
        F = np.zeros((num_strings))
        stride1 = index_ia * num_links
        for excitation1 in range(num_links):
            index_ka = ctx.table[stride1 + excitation1][0]
            sign1 = ctx.table[stride1 + excitation1][1]
            k = ctx.table[stride1 + excitation1][2]
            l = ctx.table[stride1 + excitation1][3]
            kl = k * ctx.nmo + l
            F[index_ka] += sign1 * h1e[k, l]
            stride2 = index_ka * num_links
            for excitation2 in range(num_links):
                index_ja = ctx.table[stride2 + excitation2][0]
                sign2 = ctx.table[stride2 + excitation2][1]
                i = ctx.table[stride2 + excitation2][2]
                j = ctx.table[stride2 + excitation2][3]
                ij = i * ctx.nmo + j
                if ij >= kl:
                    F[index_ja] += (sign1 * sign2 * h2e[ij][kl]) / (1 + (ij == kl))
        for index_ja in range(0, num_strings):
            for index_ib in range(0, num_strings):
                # composite index of determinant
                index_I = index_ia * num_strings + index_ib
                index_J = index_ja * num_strings + index_ib
                c1_vectors[index_I] += F[index_ja] * c_vectors[index_J]


def sigma_dipole(
    ctx, h1e, c_vectors, c1_vectors, num_strings, num_links, someconstant
):
    for index_ib in range(0, num_strings):
        F = np.zeros((num_strings))
        stride1 = index_ib * num_links
        # print('index_ib',index_ib,'stride',stride1)
        for excitation1 in range(num_links):
            index_kb = ctx.table[stride1 + excitation1][0]
            sign1 = ctx.table[stride1 + excitation1][1]
            k = ctx.table[stride1 + excitation1][2]
            l = ctx.table[stride1 + excitation1][3]
            # print(index_kb,sign1,k,l)
            kl = k * ctx.nmo + l
            F[index_kb] += sign1 * h1e[k, l]
        for index_jb in range(0, num_strings):
            for index_ia in range(0, num_strings):
                # composite index of determinant
                index_I = index_ia * num_strings + index_ib
                index_J = index_ia * num_strings + index_jb
                c1_vectors[index_I] += (
                    someconstant * F[index_jb] * c_vectors[index_J]
                )
    for index_ia in range(0, num_strings):
        F = np.zeros((num_strings))
        stride1 = index_ia * num_links
        for excitation1 in range(num_links):
            index_ka = ctx.table[stride1 + excitation1][0]
            sign1 = ctx.table[stride1 + excitation1][1]
            k = ctx.table[stride1 + excitation1][2]
            l = ctx.table[stride1 + excitation1][3]
            kl = k * ctx.nmo + l
            F[index_ka] += sign1 * h1e[k, l]
        for index_ja in range(0, num_strings):
            for index_ib in range(0, num_strings):
                # composite index of determinant
                index_I = index_ia * num_strings + index_ib
                index_J = index_ja * num_strings + index_ib
                c1_vectors[index_I] += (
                    someconstant * F[index_ja] * c_vectors[index_J]
                )


def build_sigma(ctx, c_vectors, s_vectors, H_dim):
    """
    build sigma for FCI and CASCI. The function use Knowles-Handy algorithm for fci, and Olsen algorithm for casci
    """

    if ctx.ci_level == "fci":
        print("using Knowles-Handy algorithm")
        np1 = ctx.N_p + 1
        for n in range(c_vectors.shape[1]):
            for m in range(ctx.N_p + 1):
                start = m * H_dim // np1
                end = (m + 1) * H_dim // np1
                one_e_contraction(ctx, 
                    ctx.oneeint,
                    c_vectors[start:end, n],
                    s_vectors[start:end, n],
                    ctx.num_alpha,
                    ctx.num_links,
                    1,
                )
                two_e_contraction(ctx, 
                    ctx.twoeint,
                    c_vectors[start:end, n],
                    s_vectors[start:end, n],
                    ctx.num_alpha,
                    ctx.num_links,
                )
                someconstant = m * ctx.omega + ctx.Enuc + ctx.d_c
                constant_terms_contraction(ctx, 
                    c_vectors[start:end, n], s_vectors[start:end, n], someconstant
                )
                start0 = (m - 1) * H_dim // np1
                end0 = m * H_dim // np1
                start2 = (m + 1) * H_dim // np1
                end2 = (m + 2) * H_dim // np1
                if 0 < m < ctx.N_p:
                    someconstant = -np.sqrt(m * ctx.omega / 2)
                    one_e_contraction(ctx, 
                        ctx.d_cmo,
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                    someconstant = -np.sqrt((m + 1) * ctx.omega / 2)
                    one_e_contraction(ctx, 
                        ctx.d_cmo,
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                elif m == ctx.N_p:
                    someconstant = -np.sqrt(m * ctx.omega / 2)
                    one_e_contraction(ctx, 
                        ctx.d_cmo,
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                else:
                    someconstant = -np.sqrt((m + 1) * ctx.omega / 2)
                    one_e_contraction(ctx, 
                        ctx.d_cmo,
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
    elif ctx.ci_level == "cas":
        print("using Olsen algorithm")
        np1 = ctx.N_p + 1
        for n in range(c_vectors.shape[1]):
            for m in range(ctx.N_p + 1):
                start = m * H_dim // np1
                end = (m + 1) * H_dim // np1
                sigma12(ctx, 
                    ctx.gkl,
                    ctx.twoeint,
                    c_vectors[start:end, n],
                    s_vectors[start:end, n],
                    ctx.num_alpha,
                    ctx.num_links,
                )
                sigma3(ctx, 
                    ctx.twoeint,
                    c_vectors[start:end, n],
                    s_vectors[start:end, n],
                    ctx.num_alpha,
                    ctx.num_links,
                )
                someconstant = m * ctx.omega + ctx.Enuc + ctx.d_c
                if ctx.break_degeneracy == True:
                    # print('only ground state')
                    someconstant = m * (ctx.omega + 1) + ctx.Enuc + ctx.d_c

                constant_terms_contraction(ctx, 
                    c_vectors[start:end, n], s_vectors[start:end, n], someconstant
                )
                start0 = (m - 1) * H_dim // np1
                end0 = m * H_dim // np1
                start2 = (m + 1) * H_dim // np1
                end2 = (m + 2) * H_dim // np1
                if 0 < m < ctx.N_p:
                    someconstant = -np.sqrt(m * ctx.omega / 2)
                    sigma_dipole(ctx, 
                        ctx.d_cmo,
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                    someconstant = -np.sqrt((m + 1) * ctx.omega / 2)
                    sigma_dipole(ctx, 
                        ctx.d_cmo,
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                elif m == ctx.N_p:
                    someconstant = -np.sqrt(m * ctx.omega / 2)
                    sigma_dipole(ctx, 
                        ctx.d_cmo,
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start0:end0, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )
                else:
                    someconstant = -np.sqrt((m + 1) * ctx.omega / 2)
                    sigma_dipole(ctx, 
                        ctx.d_cmo,
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        ctx.num_alpha,
                        ctx.num_links,
                        someconstant,
                    )
                    constant_terms_contraction(ctx, 
                        c_vectors[start2:end2, n],
                        s_vectors[start:end, n],
                        -ctx.d_exp * someconstant,
                    )


def build_H_diag(ctx, H_dim, h1e, h2e, num_strings, n_act_a, n_act_orb, n_in_a):
    d = np.zeros((H_dim))
    np1 = ctx.N_p + 1
    for m in range(ctx.N_p + 1):
        start = m * H_dim // np1
        end = (m + 1) * H_dim // np1

        for I in range(ctx.num_det):
            index_a = I // num_strings
            index_b = I % num_strings
            string_a = index_to_string(ctx, 
                index_a, n_act_a, n_act_orb, ctx.Y, return_binary=False
            )
            string_b = index_to_string(ctx, 
                index_b, n_act_a, n_act_orb, ctx.Y, return_binary=False
            )
            inactive_list = list(x for x in range(n_in_a))
            double_occupation = Determinant.obtBits2ObtIndexList(
                string_a & string_b
            )
            double_occupation = [x + n_in_a for x in double_occupation]
            double_occupation[0:0] = inactive_list
            dim_d = len(double_occupation)
            e = string_a ^ string_b
            ea = e & string_a
            eb = e & string_b
            single_occupation_a = Determinant.obtBits2ObtIndexList(ea)
            single_occupation_a = [x + n_in_a for x in single_occupation_a]
            single_occupation_b = Determinant.obtBits2ObtIndexList(eb)
            single_occupation_b = [x + n_in_a for x in single_occupation_b]
            dim_sa = len(single_occupation_a)
            dim_sb = len(single_occupation_b)
            # print(double_occupation,single_occupation_a,single_occupation_b)
            occupation_list_spin = np.zeros((dim_sa + dim_sb + dim_d, 3), dtype=int)
            for i in range(dim_d):
                occupation_list_spin[i][0] = double_occupation[i]
                occupation_list_spin[i][1] = 1
                occupation_list_spin[i][2] = 1
            for i in range(dim_sa):
                occupation_list_spin[i + dim_d][0] = single_occupation_a[i]
                occupation_list_spin[i + dim_d][1] = 1
            for i in range(dim_sb):
                occupation_list_spin[i + dim_d + dim_sa][0] = single_occupation_b[i]
                occupation_list_spin[i + dim_d + dim_sa][2] = 1
            c = 0
            for a in range(dim_d + dim_sa + dim_sb):
                i = occupation_list_spin[a][0]
                n_ia = occupation_list_spin[a][1]
                n_ib = occupation_list_spin[a][2]
                n_i = n_ia + n_ib
                ii = i * ctx.nmo + i
                c += n_i * h1e[i, i]
                for b in range(dim_d + dim_sa + dim_sb):
                    j = occupation_list_spin[b][0]
                    n_ja = occupation_list_spin[b][1]
                    n_jb = occupation_list_spin[b][2]
                    n_j = n_ja + n_jb
                    jj = j * ctx.nmo + j
                    c += 0.5 * n_i * n_j * h2e[ii][jj]
                    ij = i * ctx.nmo + j
                    c -= 0.5 * (n_ia * n_ja + n_ib * n_jb) * h2e[ij][ij]
            d[I + start] = c + m * ctx.omega + ctx.Enuc + ctx.d_c
    # print('d',d)
    return d


def Davidson(
    ctx, H, nroots, threshold, indim, maxdim, maxiter, build_sigma, H_diag
):
    # Supplied by helper_PFCI at call time; see _helper() above.
    _h = _helper()
    Determinant, c_graph, c_index_to_string, c_sigma = (_h.Determinant, _h.c_graph, _h.c_index_to_string, _h.c_sigma)
    if ctx.ci_level == "cis":
        H_diag = np.diag(H)
    H_dim = len(H_diag)
    print(H_dim, indim)
    L = indim
    # When L exceeds Lmax we will collapse the guess space so our sub-space
    # diagonalization problem does not grow too large
    Lmax = maxdim
    if ctx.ci_level == "cas" or ctx.ci_level == "fci":
        rows = ctx.num_alpha * (
            ctx.n_act_a * (ctx.n_act_orb - ctx.n_act_a)
            + ctx.n_act_a
            + ctx.n_in_a
        )
        num_links = rows // ctx.num_alpha

    # An array to hold the excitation energies
    theta = [0.0] * L

    # an array to hold the excitation rank count and cumulative weight
    ctx.determinant_count_by_excitation_rank = np.zeros(ctx.n_act_el)
    ctx.determinant_cumulative_weight_by_excitation_rank = np.zeros(ctx.n_act_el)

    # generate initial guess
    Q_idx = H_diag.argsort()[:indim]
    # print(Q_idx)
    if ctx.davidson_guess == "unit guess":
        Q = np.zeros((indim, H_dim))
        print("use unit guess")
        BIGNUM = 10**100
        H_diag2 = np.copy(H_diag)
        for i in range(indim):
            minimum = H_diag2[0]
            min_pos = 0
            for j in range(H_dim):
                if H_diag2[j] < minimum:
                    minimum = H_diag2[j]
                    min_pos = j
            Q[i][min_pos] = 1.0
            H_diag2[min_pos] = BIGNUM
    else:
        print("use random guess")
        Q = np.random.rand(indim, H_dim)
    # print(Q)
    print("qshape", np.shape(Q))
    print("initial_mem_q", Q.size * Q.itemsize / 1024 / 1024)

    num_iter = maxiter
    for a in range(0, num_iter):
        print("\n")
        print("mem_q1", Q.size * Q.itemsize / 1024 / 1024)
        # orthonormalization of basis vectors by QR
        t_qr_begin = time.time()
        Q, R = np.linalg.qr(Q.T)
        t_qr_end = time.time()
        print("QR took", t_qr_end - t_qr_begin, "seconds")
        Q = np.ascontiguousarray(Q.T)
        R = None
        del R
        L = Q.shape[0]  # dynamic dimension of subspace
        print(np.shape(Q))
        print("iteration", a + 1, "subspace dimension", L)
        theta_old = theta[:nroots]
        # singma build
        S = np.zeros_like(Q)
        if ctx.ci_level == "fci" or ctx.ci_level == "cas":
            # S = np.einsum("pq,qi->pi", H, Q)
            t_sigma_begin = time.time()
            # build_sigma(Q,S,H_dim)
            if ctx.ignore_dse_terms:
                ctx.d_c = 0.0
            print(psutil.Process().memory_info().rss / (1024 * 1024))

            c_sigma(
                ctx.gkl2,
                ctx.twoeint,
                ctx.d_cmo,
                Q,
                S,
                ctx.table,
                ctx.table_creation,
                ctx.table_annihilation,
                ctx.n_act_a,
                ctx.n_act_orb,
                ctx.n_in_a,
                ctx.nmo,
                L,
                ctx.N_p,
                ctx.Enuc,
                ctx.d_c,
                ctx.omega,
                ctx.d_exp,
                ctx.E_core,
                ctx.break_degeneracy,
            )

            print(psutil.Process().memory_info().rss / (1024 * 1024))

            t_sigma_end = time.time()
            print("build sigma took", t_sigma_end - t_sigma_begin, "seconds")
        else:
            S = np.einsum("ip,pq->iq", Q, H)
        # Build the subspace Hamiltonian
        G = np.dot(Q, S.T)
        # Diagonalize it, and sort the eigenvector/eigenvalue pairs
        theta, alpha = np.linalg.eigh(G)
        # print(theta)
        idx = theta.argsort()[:nroots]
        theta = theta[idx]
        alpha = alpha[:, idx]
        # This vector will hold the new guess vectors to add to our space
        # add_Q = []
        w = np.zeros((nroots, H_dim))
        residual_norm = np.zeros((nroots))
        unconverged_idx = []
        convergence_check = np.zeros((nroots), dtype=str)
        conv = 0
        for j in range(nroots):
            # Compute a residual vector "w" for each root we seek
            w[j, :] = np.dot(alpha[:, j].T, S) - theta[j] * np.dot(alpha[:, j].T, Q)
            residual_norm[j] = np.sqrt(np.dot(w[j, :], w[j, :].T))
            if residual_norm[j] < threshold:
                conv += 1
                convergence_check[j] = "Yes"
            else:
                unconverged_idx.append(j)
                convergence_check[j] = "No"
        print(unconverged_idx)

        print("root", "residual norm", "Eigenvalue", "Convergence")
        for j in range(nroots):
            print(
                j + 1, residual_norm[j], theta[j], convergence_check[j], flush=True
            )

        if conv == nroots:
            print("converged!")
            break

        if conv < nroots and a + 1 == num_iter:
            print("maxiter reached, try to increase maxiter or subspace size")
            break

        t_precondition_begin = time.time()
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

        t_precondition_end = time.time()
        print(
            "build precondition took",
            t_precondition_end - t_precondition_begin,
            "seconds",
        )
        if Lmax - L < len(unconverged_idx):
            t_collapsing_begin = time.time()
            # unconverged_w = np.zeros((len(unconverged_idx), H_dim))
            Q = np.dot(alpha.T, Q)

            # for i in range(len(unconverged_idx)):
            #    unconverged_w[i, :] = w[unconverged_idx[i], :]

            # Q=np.append(Q,unconverged_w)
            # print(Q)
            # print(unconverged_w)
            Q = np.concatenate((Q, preconditioned_w), axis=0)
            print(Q.shape)
            # Q=np.column_stack(Qtup)

            # These vectors will give the same eigenvalues at the next
            # iteration so to avoid a false convergence we reset the theta
            # vector to theta_old
            theta = theta_old
            t_collapsing_end = time.time()
            print("restart took", t_collapsing_end - t_collapsing_begin, "seconds")
        else:
            t_expanding_begin = time.time()
            # if not we add the preconditioned residuals to the guess
            # space, and continue. Note that the set will be orthogonalized
            # at the start of the next iteration
            Q = tuple(Q[i, :] for i in range(L)) + tuple(preconditioned_w)
            Q = np.row_stack(Q)
            t_expanding_end = time.time()
            print("expand took", t_expanding_end - t_expanding_begin, "seconds")
            # print(Q)
    newQ = np.dot(alpha.T, Q)
    Q = np.zeros((H_dim, nroots))
    Q = newQ.T
    singletcount = 0
    print(Q.shape)
    for i in range(Q.shape[1]):
        print("STATE", i, "energy =", theta[i])
    if ctx.ci_level == "cis":
        for i in range(Q.shape[1]):
            # print("state",i, "energy =",theta[i])
            print(
                "        amplitude",
                "      position",
                "         most important determinants",
                "             number of photon",
            )
            index = np.argsort(np.abs(Q[:, i]))
            c0 = index[Q.shape[0] - 1] % (H_dim // 2)
            d0 = (index[Q.shape[0] - 1] - c0) // (H_dim // 2)
            a0, b0 = ctx.detmap[c0]
            alphalist = Determinant.obtBits2ObtIndexList(a0)
            betalist = Determinant.obtBits2ObtIndexList(b0)

            singlet = 1
            for j in range(min(H_dim, 10)):
                c = index[Q.shape[0] - j - 1] % (H_dim // 2)
                d = (index[Q.shape[0] - j - 1] - c) // (H_dim // 2)
                a, b = ctx.detmap[c]
                if (
                    a == b0
                    and b == a0
                    and np.abs(
                        Q[index[Q.shape[0] - j - 1]][i]
                        - (-1) * Q[index[Q.shape[0] - 1]][i]
                    )
                    < 1e-4
                ):
                    singlet = singlet * 0
                else:
                    singlet = singlet * 1
                alphalist = Determinant.obtBits2ObtIndexList(a)
                betalist = Determinant.obtBits2ObtIndexList(b)

                print(
                    "%20.12lf" % (Q[index[Q.shape[0] - j - 1]][i]),
                    "%9.3d" % (index[Q.shape[0] - j - 1]),
                    "      alpha",
                    alphalist,
                    "   beta",
                    betalist,
                    "%4.1d" % (d),
                    "photon",
                )
            # print("state",i, "energy =",theta[i], singlet)
            if singlet == 1:
                print("state", i, "energy =", theta[i], "  singlet", singletcount)
                singletcount += 1
            else:
                print(
                    "state",
                    i,
                    "energy =",
                    theta[i],
                    "  triplet",
                    "%2.1d" % (d0),
                    "photon",
                )
    else:
        print(
            "\nACTIVE PART OF DETERMINANTS THAT HAVE THE MOST IMPORTANT CONTRIBUTIONS"
        )
        Y = np.zeros(
            ctx.n_act_a * (ctx.n_act_orb - ctx.n_act_a + 1) * 3, dtype=np.int32
        )
        c_graph(ctx.n_act_a, ctx.n_act_orb, Y)
        np1 = ctx.N_p + 1
        singlet_count = 0
        triplet_count = 0
        for i in range(Q.shape[1]):
            total_spin = ctx.check_total_spin(newQ[i : (i + 1), :])
            print("state", i, "energy =", theta[i], "<S^2>=", total_spin, end="")
            if np.abs(total_spin) < 1e-5:
                singlet_count += 1
                print("\tsinglet", singlet_count)
            elif np.abs(total_spin - 2.0) < 1e-5:
                triplet_count += 1
                print("\ttriplet", triplet_count)
            elif np.abs(total_spin - 6.0) < 1e-5:
                print("\tquintet")
            # print("state",i, "energy =",theta[i])
            print(
                "        amplitude",
                "      position",
                "         most important determinants",
                "             number of photon",
            )
            index = np.argsort(np.abs(Q[:, i]))
            # print(index)
            Idet0 = (
                index[Q.shape[0] - 1] % ctx.num_det
            )  # determinant index of most significant contribution
            photon_p0 = (
                index[Q.shape[0] - 1] - Idet0
            ) // ctx.num_det  # photon number block of determinant
            Ib0 = Idet0 % ctx.num_alpha
            Ia0 = Idet0 // ctx.num_alpha
            a0 = c_index_to_string(Ia0, ctx.n_act_a, ctx.n_act_orb, Y)
            b0 = c_index_to_string(Ib0, ctx.n_act_a, ctx.n_act_orb, Y)

            
            alphalist = Determinant.obtBits2ObtIndexList(a0)
            betalist = Determinant.obtBits2ObtIndexList(b0)

            # store alphalist and betalist for the reference to a numpy array
            a_ref = np.array(alphalist)
            b_ref = np.array(betalist)

            #for j in range(min(H_dim, 10)):
            for j in range(H_dim):
                Idet = index[Q.shape[0] - j - 1] % ctx.num_det
                photon_p = (index[Q.shape[0] - j - 1] - Idet) // ctx.num_det
                Ib = Idet % ctx.num_alpha
                Ia = Idet // ctx.num_alpha
                a = c_index_to_string(Ia, ctx.n_act_a, ctx.n_act_orb, Y)
                b = c_index_to_string(Ib, ctx.n_act_a, ctx.n_act_orb, Y)

                alphalist = Determinant.obtBits2ObtIndexList(a)
                betalist = Determinant.obtBits2ObtIndexList(b)

                inactive_list = list(x for x in range(ctx.n_in_a))
                alphalist2 = [x + ctx.n_in_a for x in alphalist]
                # alphalist2[0:0] = inactive_list
                betalist2 = [x + ctx.n_in_a for x in betalist]
                # betalist2[0:0] = inactive_list

                print(
                    "%20.12lf" % (Q[index[Q.shape[0] - j - 1]][i]),
                    "%9.3d" % (index[Q.shape[0] - j - 1]),
                    "      alpha",
                    alphalist2,
                    "   beta",
                    betalist2,
                    "%4.1d" % (photon_p),
                    "photon",
                )

    davidson_dict = {
        "DAVIDSON EIGENVALUES": theta,
        "DAVIDSON EIGENVECTORS": Q,
    }

    return davidson_dict


def build_sigma_reduced(ctx, U, A, G, step):
    ctx.reduced_hessian2 = np.zeros((ctx.index_map_size, ctx.index_map_size))
    # build the component of tranformed hessian from gradient
    B = np.zeros((ctx.nmo, ctx.nmo))
    ctx.T = U - np.eye(ctx.nmo)
    B[:, : ctx.n_occupied] = A[:, : ctx.n_occupied] + np.einsum(
        "klrs,sl->rk", G[:, :, :, :], ctx.T[:, : ctx.n_occupied]
    )
    # B2 = np.zeros((ctx.nmo, ctx.nmo))
    # for r in range(ctx.nmo):
    #    for k in range(ctx.n_occupied):
    #        B2[r][k] = A[r][k]
    #        for s in range(ctx.nmo):
    #            for l in range(ctx.n_occupied):
    #                B2[r][k] += G[k][l][r][s] * ctx.T[s][l]
    # print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))

    A_tilde = np.zeros((ctx.nmo, ctx.nmo))
    # A_tilde[:,:ctx.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:ctx.n_occupied])
    # A2_tilde = np.zeros((ctx.nmo, ctx.nmo))
    A_tilde = np.einsum("rs,sk->rk", U.T, B)
    # A2_tilde = np.zeros((ctx.nmo, ctx.nmo))
    # for r in range(ctx.nmo):
    #    for k in range(ctx.n_occupied):
    #        for s in range(ctx.nmo):
    #            A2_tilde[r][k] += U[s][r] * B[s][k]
    # print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))

    A3_tilde = A_tilde + A_tilde.T

    hessian_tilde = np.zeros((ctx.n_occupied, ctx.n_occupied, ctx.nmo, ctx.nmo))

    hessian_tilde -= 0.5 * np.einsum(
        "kl,rs->klrs", np.eye(ctx.n_occupied), A3_tilde
    )
    hessian_tilde -= 0.5 * np.einsum(
        "rs,kl->klrs",
        np.eye(ctx.nmo),
        A3_tilde[: ctx.n_occupied, : ctx.n_occupied],
    )
    hessian_tilde[:, :, : ctx.n_occupied, :] += 0.5 * np.einsum(
        "rl,ks->klrs", np.eye(ctx.n_occupied), A3_tilde[: ctx.n_occupied, :]
    )
    hessian_tilde[:, :, :, : ctx.n_occupied] += 0.5 * np.einsum(
        "ks,rl->klrs", np.eye(ctx.n_occupied), A3_tilde[:, : ctx.n_occupied]
    )

    ctx.R_total = np.zeros((ctx.nmo, ctx.n_occupied))
    ctx.sigma_total = np.zeros((ctx.nmo, ctx.n_occupied))
    for i in range(ctx.index_map_size):
        s = ctx.index_map[i][0]
        l = ctx.index_map[i][1]
        ctx.R_total[s][l] = step[i]
    temp1 = np.zeros((ctx.nmo, ctx.n_occupied))
    temp2 = np.zeros((ctx.nmo, ctx.n_occupied))
    temp1 = np.einsum("qs,sl->ql", U, ctx.R_total)
    temp2 = np.einsum("klpq,ql->pk", G, temp1)
    ctx.sigma_total = np.einsum("pk,pr->rk", temp2, U)
    ctx.sigma_total[: ctx.n_occupied, :] -= np.einsum(
        "pr,pk->rk", temp2, U[:, : ctx.n_occupied]
    )

    temp1 = np.einsum(
        "ql,sl->qs", U[:, : ctx.n_occupied], ctx.R_total[: ctx.n_occupied, :]
    )
    temp2 = np.einsum("kspq,qs->pk", G, temp1)
    ctx.sigma_total -= np.einsum("pk,pr->rk", temp2, U)
    ctx.sigma_total[: ctx.n_occupied, :] += np.einsum(
        "pr,pk->rk", temp2, U[:, : ctx.n_occupied]
    )
    ctx.sigma_total += np.einsum("klrs,sl->rk", hessian_tilde, ctx.R_total)


def build_sigma_reduced2(
    ctx, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer
):
    R_total = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    print("weqeqw", num_states)
    # print(ctx.index_map)
    for i in range(num_states):
        for j in range(ctx.index_map_size):
            r = ctx.index_map[j][0]
            k = ctx.index_map[j][1]
            R_total[i][r][k] = R_reduced[i][j + pointer]
    # print("R total")
    # for i in range(num_states):
    #    for r in range(ctx.nmo):
    #        for k in range(ctx.n_occupied):
    #            print("%20.12lf" %(R_total[i,r,k]))
    # print("\n")

    ###B = np.zeros((ctx.nmo, ctx.nmo))
    ###T = U - np.eye(ctx.nmo)
    ###B[:,:ctx.n_occupied] = A[:,:ctx.n_occupied] + np.einsum("klrs,sl->rk", G[:,:,:,:], T[:,:ctx.n_occupied])
    ####B2 = np.zeros((ctx.nmo, ctx.nmo))
    ####for r in range(ctx.nmo):
    ####    for k in range(ctx.n_occupied):
    ####        B2[r][k] = A[r][k]
    ####        for s in range(ctx.nmo):
    ####            for l in range(ctx.n_occupied):
    ####                B2[r][k] += G[k][l][r][s] * ctx.T[s][l]
    ####print("sdgs", np.allclose(B2,B, rtol=1e-14,atol=1e-14))
    ###
    ###A_tilde = np.zeros((ctx.nmo, ctx.nmo))
    ####A_tilde[:,:ctx.n_occupied] = np.einsum("rs,sk->rk",U.T, B[:,:ctx.n_occupied])
    ####A2_tilde = np.zeros((ctx.nmo, ctx.nmo))
    ###A_tilde = np.einsum("rs,sk->rk",U.T, B)
    # A2_tilde = np.zeros((ctx.nmo, ctx.nmo))
    # for r in range(ctx.nmo):
    #    for k in range(ctx.n_occupied):
    #        for s in range(ctx.nmo):
    #            A2_tilde[r][k] += U[s][r] * B[s][k]
    # print("bxvc", np.allclose(A2_tilde,A_tilde, rtol=1e-14,atol=1e-14))

    A3_tilde = A_tilde + A_tilde.T

    hessian_tilde = np.zeros((ctx.n_occupied, ctx.n_occupied, ctx.nmo, ctx.nmo))

    hessian_tilde -= 0.5 * np.einsum(
        "kl,rs->klrs", np.eye(ctx.n_occupied), A3_tilde
    )
    hessian_tilde -= 0.5 * np.einsum(
        "rs,kl->klrs",
        np.eye(ctx.nmo),
        A3_tilde[: ctx.n_occupied, : ctx.n_occupied],
    )
    hessian_tilde[:, :, : ctx.n_occupied, :] += 0.5 * np.einsum(
        "rl,ks->klrs", np.eye(ctx.n_occupied), A3_tilde[: ctx.n_occupied, :]
    )
    hessian_tilde[:, :, :, : ctx.n_occupied] += 0.5 * np.einsum(
        "ks,rl->klrs", np.eye(ctx.n_occupied), A3_tilde[:, : ctx.n_occupied]
    )

    sigma_total = np.zeros((num_states, ctx.nmo, ctx.n_occupied))

    temp1 = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    temp2 = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    temp1 = np.einsum("qs,isl->iql", U, R_total)
    temp2 = np.einsum("klpq,iql->ipk", G, temp1)
    sigma_total = np.einsum("ipk,pr->irk", temp2, U)
    sigma_total[:, : ctx.n_occupied, :] -= np.einsum(
        "ipr,pk->irk", temp2, U[:, : ctx.n_occupied]
    )

    temp1 = np.einsum(
        "ql,isl->iqs", U[:, : ctx.n_occupied], R_total[:, : ctx.n_occupied, :]
    )
    temp2 = np.einsum("kspq,iqs->ipk", G, temp1)
    sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
    sigma_total[:, : ctx.n_occupied, :] += np.einsum(
        "ipr,pk->irk", temp2, U[:, : ctx.n_occupied]
    )
    sigma_total += np.einsum("klrs,isl->irk", hessian_tilde, R_total)

    for i in range(num_states):
        for j in range(ctx.index_map_size):
            r = ctx.index_map[j][0]
            k = ctx.index_map[j][1]
            sigma_reduced[i][j + pointer] = sigma_total[i][r][k]


def build_sigma_reduced3(
    ctx, U, hessian_tilde, G, R_reduced, sigma_reduced, num_states, pointer
):
    R_total = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    print("weqeqw", num_states)
    for i in range(num_states):
        for j in range(ctx.index_map_size):
            r = ctx.index_map[j][0]
            k = ctx.index_map[j][1]
            R_total[i][r][k] = R_reduced[i][j + pointer]

    sigma_total = np.zeros((num_states, ctx.nmo, ctx.n_occupied))

    temp1 = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    temp2 = np.zeros((num_states, ctx.nmo, ctx.n_occupied))
    temp1 = np.einsum("qs,isl->iql", U, R_total, optimize="optimal")
    temp1 -= np.einsum(
        "ql,isl->iqs",
        U[:, : ctx.n_occupied],
        R_total[:, : ctx.n_occupied, :],
        optimize="optimal",
    )
    temp2 = np.einsum("klpq,iql->ipk", G, temp1, optimize="optimal")
    sigma_total = np.einsum("ipk,pr->irk", temp2, U, optimize="optimal")
    sigma_total[:, : ctx.n_occupied, :] -= np.einsum(
        "ipr,pk->irk", temp2, U[:, : ctx.n_occupied], optimize="optimal"
    )

    # temp1 = np.einsum("ql,isl->iqs", U[:,:ctx.n_occupied], R_total[:,:ctx.n_occupied,:])
    # temp2 = np.einsum("kspq,iqs->ipk", G, temp1, optimize = "optimal")
    # sigma_total -= np.einsum("ipk,pr->irk", temp2, U)
    # sigma_total[:,:ctx.n_occupied,:] += np.einsum("ipr,pk->irk", temp2, U[:,:ctx.n_occupied])
    sigma_total += np.einsum(
        "klrs,isl->irk", hessian_tilde, R_total, optimize="optimal"
    )
    for i in range(num_states):
        for j in range(ctx.index_map_size):
            r = ctx.index_map[j][0]
            k = ctx.index_map[j][1]
            sigma_reduced[i][j + pointer] = sigma_total[i][r][k]


def build_sigma_reduced4(
    U,
    A_tilde,
    index_map,
    G,
    R_reduced,
    sigma_reduced,
    num_states,
    pointer,
    nmo,
    index_map_size,
    n_occupied,
):

    assert U.shape == (nmo, nmo)
    assert G.shape == (nmo * n_occupied, nmo * n_occupied)
    assert A_tilde.shape == (nmo, nmo)
    R_total = np.zeros((num_states, nmo, n_occupied))
    print("num_states", num_states)
    #for j in nb.prange(index_map_size):
    for j in range(index_map_size):
        r = index_map[j][0]
        k = index_map[j][1]
        for i in range(num_states):
            R_total[i][r][k] = R_reduced[i][j + pointer]

    sigma_total = np.zeros((num_states, nmo, n_occupied))
    temp1 = np.zeros((num_states, nmo, n_occupied))
    temp2 = np.zeros((num_states, nmo, n_occupied))

    R1 = R_total.transpose(1, 0, 2)
    R1 = np.ascontiguousarray(R1)
    R1 = np.reshape(R1, (nmo, num_states * n_occupied))
    temp2 = np.dot(U, R1)
    temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
    R1 = np.ascontiguousarray(R_total[:, :n_occupied, :].transpose(2, 0, 1))
    # R_total1 = np.ascontiguousarray(R_total1)
    R1 = np.reshape(R1, (n_occupied, num_states * n_occupied))
    U1 = np.ascontiguousarray(U[:, :n_occupied])
    temp2 = np.dot(U1, R1)
    temp1 -= temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)

    ##sigma_total1 = np.zeros((num_states, nmo, n_occupied))
    ##temp11 = np.einsum("qs,isl->iql", U, R_total)
    # for q in nb.prange(nmo):
    #    for i in range(num_states):
    #        for l in range(n_occupied):
    #            #a = 0.0
    #            a = np.float64(0)
    #            for s in range(nmo):
    #                a += U[q,s] * R_total[i,s,l]
    #                #temp1[i,q,l] += U[q,s] * R_total[i,s,l]
    #            temp1[i,q,l] = a
    #

    # for q in nb.prange(nmo):
    #    for i in range(num_states):
    #        for s in range(n_occupied):
    #            a = np.float64(0)
    #            for l in range(n_occupied):
    #                a -= U[q,l] * R_total[i,s,l]
    #                #temp1[i,q,s] -= U[q,l] * R_total[i,s,l]
    #            temp1[i,q,s] += a
    R1 = np.ascontiguousarray(temp1)
    R1 = np.reshape(R1, (num_states, nmo * n_occupied))
    
    # R2 = np.zeros((num_states, nmo * n_occupied))
    ##for i in nb.prange(num_states):
    ##    for r in range(nmo):
    ##        for k in range(n_occupied):
    ##            a = R1[i][r*n_occupied+k] - temp1[i,r,k]
    ##            if np.abs(a) > 1e-14: print("LARGE error")
    #G_tensor = G.reshape(nmo, n_occupied, nmo, n_occupied)

    temp1 = np.dot(R1, G)
    # temp2 = temp1.reshape(num_states, nmo, n_occupied)
    # for r in nb.prange(nmo):
    #    for i in range(num_states):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for p in range(nmo):
    #                a += U[p,r] * temp2[i,p,k]
    #                #sigma_total[i,r,k] += U[p,r] * temp2[i,p,k]
    #            temp1[i,r,k] = a
    #            sigma_total[i,r,k] = a
    #

    # for r in nb.prange(n_occupied):
    #    for i in range(num_states):
    #        for k in range(n_occupied):
    #            sigma_total[i,r,k] -= temp1[i,k,r]

    #######for i in nb.prange(num_states):
    #######    R3 = np.dot(R1[i,:], G)
    #######    R2[i,:] = R3

    temp2 = temp1.reshape(num_states, nmo, n_occupied)
    #temp2 = oe.contract("irk,rksl->isl", R1, G_tensor, optimize = "optimal") 
    temp2 = temp2.transpose(0, 2, 1)
    temp2 = np.ascontiguousarray(temp2)
    temp2 = np.reshape(temp2, (num_states * n_occupied, nmo))
    temp1 = np.dot(temp2, U)
    temp2 = temp1.reshape(num_states, n_occupied, nmo)
    sigma_total[:, :, :] = temp2.transpose(0, 2, 1)[:, :, :]
    sigma_total[:, :n_occupied, :] -= temp2[:, :, :n_occupied]

    A3_tilde = A_tilde + A_tilde.T
    # temp1 = np.zeros((num_states, nmo, n_occupied))
    # temp2 = np.zeros((num_states, nmo, n_occupied))

    ##hessian_tilde -= 0.5 * np.einsum("kl,rs->klrs", np.eye(ctx.n_occupied), A3_tilde)
    ##hessian_tilde[:,:,:ctx.n_occupied,:] += 0.5 * np.einsum("rl,ks->klrs", np.eye(ctx.n_occupied), A3_tilde[:ctx.n_occupied,:])
    # for i in nb.prange(num_states):
    #    for r in range(nmo):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for s in range(nmo):
    #                a += A3_tilde[r,s] * R_total[i,s,k]
    #            temp1[i,r,k] = a
    #            sigma_total[i,r,k] -= 0.5 * a
    #
    # for i in nb.prange(num_states):
    #    for r in range(n_occupied):
    #        for k in range(n_occupied):
    #            sigma_total[i,r,k] += 0.5 * temp1[i,k,r]
    #
    ##hessian_tilde -= 0.5 * np.einsum("rs,kl->klrs", np.eye(ctx.nmo), A3_tilde[:ctx.n_occupied,:ctx.n_occupied])
    # for i in nb.prange(num_states):
    #    for r in range(nmo):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for l in range(n_occupied):
    #                a += A3_tilde[k,l] * R_total[i,r,l]
    #            sigma_total[i,r,k] -= 0.5 * a

    ##hessian_tilde[:,:,:,:ctx.n_occupied] += 0.5 * np.einsum("ks,rl->klrs", np.eye(ctx.n_occupied), A3_tilde[:,:ctx.n_occupied])
    # for i in nb.prange(num_states):
    #    for r in range(nmo):
    #        for k in range(n_occupied):
    #            a = np.float64(0)
    #            for l in range(n_occupied):
    #                a += A3_tilde[r,l] * R_total[i,k,l]
    #            sigma_total[i,r,k] += 0.5 * a
    R1 = R_total.transpose(1, 0, 2)
    R1 = np.ascontiguousarray(R1)
    R1 = np.reshape(R1, (nmo, num_states * n_occupied))
    temp2 = np.dot(A3_tilde, R1)
    temp1 = temp2.reshape(nmo, num_states, n_occupied).transpose(1, 0, 2)
    sigma_total[:, :, :] -= 0.5 * temp1[:, :, :]
    sigma_total[:, :n_occupied, :] += (
        0.5 * temp1.transpose(0, 2, 1)[:, :, :n_occupied]
    )
    A3 = A3_tilde[:n_occupied, :n_occupied].T
    A3 = np.ascontiguousarray(A3)
    R1 = np.ascontiguousarray(R_total)
    R1 = np.reshape(R1, (num_states * nmo, n_occupied))
    temp2 = np.dot(R1, A3)
    sigma_total[:, :, :] -= (
        0.5 * temp2.reshape(num_states, nmo, n_occupied)[:, :, :]
    )
    A3 = A3_tilde[:, :n_occupied].T
    A3 = np.ascontiguousarray(A3)
    R1 = np.ascontiguousarray(R_total[:, :n_occupied, :])
    R1 = np.reshape(R1, (num_states * n_occupied, n_occupied))
    temp2 = np.dot(R1, A3)
    temp1 = temp2.reshape(num_states, n_occupied, nmo)
    sigma_total[:, :, :] += 0.5 * temp1.transpose(0, 2, 1)[:, :, :]

    for j in range(index_map_size):
        r = index_map[j][0]
        k = index_map[j][1]
        for i in range(num_states):
            sigma_reduced[i][j + pointer] = sigma_total[i][r][k]


def orbital_sigma(
    ctx, U, A_tilde, G, R_reduced, sigma_reduced, num_states, pointer
):
    """Dense-G1 sigma build.  SUPERSEDED by orbital_sigma3/build_sigma_reduced7,
    which take the (G_ij, G_ti, G_tu) blocks instead.  No live caller remains:
    the full (nmo*n_occupied) square G1 this needs is no longer materialized
    anywhere in microiteration_optimization6.  Kept for reference only --
    build_sigma_reduced4 below is otherwise identical to build_sigma_reduced7.
    """
    nmo = ctx.nmo
    index_map = ctx.index_map
    index_map_size = ctx.index_map_size
    n_occupied = ctx.n_occupied
    return build_sigma_reduced4(ctx, 
        U,
        A_tilde,
        index_map,
        G,
        R_reduced,
        sigma_reduced,
        num_states,
        pointer,
        nmo,
        index_map_size,
        n_occupied,
    )


def print_matrix_nice(ctx, matrix, precision=6, width=12, cols_per_line=6, threshold=1e-10):
    """
    Print matrix with custom formatting
    """
    rows, cols = matrix.shape

    for start_col in range(0, cols, cols_per_line):
        end_col = min(start_col + cols_per_line, cols)

        print(f"\nColumns {start_col} to {end_col-1}:")
        print("-" * (width * (end_col - start_col)))

        for i in range(rows):
            row_str = ""
            for j in range(start_col, end_col):
                value = matrix[i, j]
                if abs(value) < threshold:  # Treat very small numbers as zero
                    value = 0.0
                row_str += f"{value:{width}.{precision}f}"
            print(f"Row {i}: {row_str}")


def solve_pcg_trust_region(ctx, U, A_tilde, G_blocks, g, M_diag, trust_radius, tol=1e-8, max_iter=1000):
    """
    Solves the trust-region subproblem using Preconditioned Conjugate Gradient (Steihaug-CG).

    This solver finds a step 'p' that minimizes the quadratic model:
        m(p) = g^T p + 0.5 * p^T H p
    subject to:
        ||p|| <= trust_radius

    It is "matrix-free" and handles the key cases:
    1. Finds the Newton step if H is positive definite and the step is inside the trust region.
    2. Stops if H is found to be non-positive-definite (negative curvature).
    3. Stops if the step exceeds the trust radius.

    Arguments:
    Hess_vec_product (callable): A function that takes a vector 'v' 
                                 and returns the Hessian-vector product H*v.
    g (np.ndarray):              The current gradient vector.
    M_diag (np.ndarray):         A vector of the diagonal entries of the preconditioner M.
    trust_radius (float):        The current trust radius (Delta).
    tol (float):                 Tolerance for convergence (based on residual norm).
    max_iter (int):              Maximum number of CG iterations.

    Returns:
    (np.ndarray, str): A tuple containing the computed step 'p' and a string 
                       indicating the reason for termination.
    """
    M_diag = np.abs(M_diag)
    # Prevent division by zero (or extremely large steps) for near-zero diagonals
    epsilon = 1e-6
    M_diag = np.maximum(M_diag, epsilon)
    # --- 1. Initialization ---
    p = np.zeros_like(g) # Step (starts at 0)
    r = -g               # Residual (r = -g - H*p, but p=0)
    
    # --- This is where the diagonal preconditioner is used ---
    # z = M_inv * r. For a diagonal M, M_inv * r is just r / M_diag
    z = r / M_diag
    
    d = z.copy()         # Search direction
    r_dot_z = np.dot(r, z)
    print(r)
    print(z)
    print("r_dot_z", r_dot_z) 
    # Check for a trivial solution (gradient is already zero)
    if np.sqrt(r_dot_z) < tol:
        return p, "success_gradient_zero"

    # --- 2. PCG Iteration Loop ---
    for j in range(max_iter):
        
        # --- This is the main matrix-vector product ---
        
        Hd = ctx.mv2(U, A_tilde, G_blocks, d, 1, 0, 0)
        # --- This is the Positive-Definite (Curvature) Check ---
        d_H_d = np.dot(d, Hd)
        print(d_H_d)
        if d_H_d <= 0:
            # The Hessian is not positive definite (negative or zero curvature).
            # We must find the step 'p' that extends to the trust boundary
            # along the current direction 'd'.
            # We need to find tau > 0 such that ||p + tau*d|| = trust_radius
            
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - trust_radius**2
            
            # Solve a*tau^2 + b*tau + c = 0 for the positive root tau
            discriminant = b**2 - 4*a*c
            tau = (-b + np.sqrt(discriminant)) / (2 * a)
            
            return p + tau * d, "stop_negative_curvature"
        
        # --- Standard PCG Step ---
        alpha = r_dot_z / d_H_d
        p_new = p + alpha * d
        print("norm of p", np.linalg.norm(p_new)) 
        # --- This is the Trust-Radius Boundary Check ---
        if np.linalg.norm(p_new) > trust_radius:
            # The step has crossed the boundary. We must find the
            # step 'p' that stops exactly *on* the boundary.
            # We use the same quadratic formula as in the negative curvature case.
            
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - trust_radius**2
            
            discriminant = b**2 - 4*a*c
            tau = (-b + np.sqrt(discriminant)) / (2 * a)
            
            return p + tau * d, "stop_trust_boundary"
        
        # --- Update for next iteration (no problems found) ---
        p = p_new
        r = r - alpha * Hd
        
        # --- Check for Convergence (Newton step found) ---
        print(j, np.linalg.norm(r))
        if np.linalg.norm(r) < tol:
            return p, "success_solution_found" # This is the internal Newton step
            
        # --- Apply preconditioner again ---
        z = r / M_diag
        
        r_dot_z_new = np.dot(r, z)
        
        # Beta calculation (Fletcher-Reeves)
        beta = r_dot_z_new / r_dot_z
        
        # Update search direction
        d = z + beta * d
        r_dot_z = r_dot_z_new

    # If the loop finishes, we ran out of iterations
    return p, "stop_max_iterations"   


def solve_pcg_with_operator(ctx, B_op, g, M_diag, trust_radius, tol=1e-8, max_iter=1000):
    """
    Solves the trust-region subproblem using Preconditioned Conjugate Gradient (Steihaug-CG).

    This solver finds a step 'p' that minimizes the quadratic model:
        m(p) = g^T p + 0.5 * p^T H p
    subject to:
        ||p|| <= trust_radius

    It is "matrix-free" and handles the key cases:
    1. Finds the Newton step if H is positive definite and the step is inside the trust region.
    2. Stops if H is found to be non-positive-definite (negative curvature).
    3. Stops if the step exceeds the trust radius.

    Arguments:
    Hess_vec_product (callable): A function that takes a vector 'v' 
                                 and returns the Hessian-vector product H*v.
    g (np.ndarray):              The current gradient vector.
    M_diag (np.ndarray):         A vector of the diagonal entries of the preconditioner M.
    trust_radius (float):        The current trust radius (Delta).
    tol (float):                 Tolerance for convergence (based on residual norm).
    max_iter (int):              Maximum number of CG iterations.

    Returns:
    (np.ndarray, str): A tuple containing the computed step 'p' and a string 
                       indicating the reason for termination.
    """
    M_diag = np.abs(M_diag)
    # Prevent division by zero (or extremely large steps) for near-zero diagonals
    epsilon = 1e-6
    M_diag = np.maximum(M_diag, epsilon)
    # --- 1. Initialization ---
    p = np.zeros_like(g) # Step (starts at 0)
    r = -g               # Residual (r = -g - H*p, but p=0)
    
    # --- This is where the diagonal preconditioner is used ---
    # z = M_inv * r. For a diagonal M, M_inv * r is just r / M_diag
    z = r / M_diag
    
    d = z.copy()         # Search direction
    r_dot_z = np.dot(r, z)
    #print(r)
    #print(z)
    print("r_dot_z", r_dot_z) 
    # Check for a trivial solution (gradient is already zero)
    if np.sqrt(r_dot_z) < tol:
        return p, "success_gradient_zero"

    # --- 2. PCG Iteration Loop ---
    for j in range(max_iter):
        
        # --- This is the main matrix-vector product ---
        
        Hd = B_op(d) 
        # --- This is the Positive-Definite (Curvature) Check ---
        d_H_d = np.dot(d, Hd)
        print(d_H_d)
        if d_H_d <= 0:
            # The Hessian is not positive definite (negative or zero curvature).
            # We must find the step 'p' that extends to the trust boundary
            # along the current direction 'd'.
            # We need to find tau > 0 such that ||p + tau*d|| = trust_radius
            
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - trust_radius**2
            
            # Solve a*tau^2 + b*tau + c = 0 for the positive root tau
            discriminant = b**2 - 4*a*c
            tau = (-b + np.sqrt(discriminant)) / (2 * a)
            
            return p + tau * d, "stop_negative_curvature"
        
        # --- Standard PCG Step ---
        alpha = r_dot_z / d_H_d
        p_new = p + alpha * d
        print("norm of p", np.linalg.norm(p_new)) 
        # --- This is the Trust-Radius Boundary Check ---
        if np.linalg.norm(p_new) > trust_radius:
            # The step has crossed the boundary. We must find the
            # step 'p' that stops exactly *on* the boundary.
            # We use the same quadratic formula as in the negative curvature case.
            
            a = np.dot(d, d)
            b = 2 * np.dot(p, d)
            c = np.dot(p, p) - trust_radius**2
            
            discriminant = b**2 - 4*a*c
            tau = (-b + np.sqrt(discriminant)) / (2 * a)
            
            return p + tau * d, "stop_trust_boundary"
        
        # --- Update for next iteration (no problems found) ---
        p = p_new
        r = r - alpha * Hd
        
        # --- Check for Convergence (Newton step found) ---
        print(j, np.linalg.norm(r))
        if np.linalg.norm(r) < tol:
            return p, "success_solution_found" # This is the internal Newton step
            
        # --- Apply preconditioner again ---
        z = r / M_diag
        
        r_dot_z_new = np.dot(r, z)
        
        # Beta calculation (Fletcher-Reeves)
        beta = r_dot_z_new / r_dot_z
        
        # Update search direction
        d = z + beta * d
        r_dot_z = r_dot_z_new

    # If the loop finishes, we ran out of iterations
    return p, "stop_max_iterations"   