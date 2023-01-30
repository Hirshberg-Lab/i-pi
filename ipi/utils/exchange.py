"""Contains all methods to evalaute potential energy and forces for indistinguishable particles.
Used in /engine/normalmodes.py
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.
from ipi.utils import units
from ipi.utils.depend import *

import numpy as np


def kth_diag_indices(a, k):
    """
    Indices to access matrix k-diagonals in numpy.
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

def tril_first_axes(arr, k):
    return np.transpose(np.tril(np.transpose(arr), k=k))

class ExchangePotential(dobject):
    def __init__(self, boson_identities, all_particle_bead_positions,
                 nbeads, bead_mass,
                 spring_freq_squared, betaP):
        assert len(boson_identities) != 0

        self._N = len(boson_identities)
        self._P = nbeads
        self._betaP = betaP
        self._spring_freq_squared = spring_freq_squared
        self._particle_mass = bead_mass

        self._q = self._init_bead_position_array(boson_identities, all_particle_bead_positions)

        # self._bead_diff_intra[j] = [r^{j+1}_0 - r^{j}_0, ..., r^{j+1}_{N-1} - r^{j}_{N-1}]
        self._bead_diff_intra = np.diff(self._q, axis=0)
        # self._bead_dist_inter_first_last_bead[l][m] = r^0_{l} - r^{P-1}_{m}
        self._bead_diff_inter_first_last_bead = self._q[0, :, np.newaxis, :] - self._q[self._P - 1, np.newaxis, :, :]

        # self._E_from_to[l1, l2] is the spring energy of the cycle on particle indices l1,...,l2
        self._E_from_to = self._evaluate_cycle_energies()
        # self._V[l] = V^[1,l+1]
        self._V = self._evaluate_V_forward()

        # TODO: remove
        self._V_backward = self._evaluate_V_backward()

    def _init_bead_position_array(self, boson_identities, qall):
        q = np.empty((self._P, self._N, 3), float)
        # Stores coordinates just for bosons in separate arrays with new indices 1,...,Nbosons
        # q[j,:] stores 3*natoms xyz coordinates of all atoms.
        # Index of bead #(j+1) of atom #(l+1) is [l,3*l]
        for ind, boson in enumerate(boson_identities):
            q[:, ind, :] = qall[:, 3 * boson: (3 * boson + 3)]

        return q

    def V_all(self):
        """
        The forward/backward potential on all particles: V^[1,N]
        """
        return self._V[self._N]

    def get_vspring_and_fspring(self):
        """
        Calculates spring forces and potential for bosons.
        """
        F = self.evaluate_dVB_from_VB()

        return [self._V[-1], F]

    def evaluate_dVB_from_VB(self):
        F = np.zeros((self._P, self._N, 3), float)

        dexpVall = -1 / (self._betaP * np.exp(- self._betaP * self.V_all()))

        dexpVm = np.empty(self._N)
        dexpVm[-1] = dexpVall
        for m in range(self._N - 2, -1, -1):
            # val = 0.0
            # for v in range(m+1, self._N):
            #     val += dexpVm[v] * 1/(v+1) * np.exp(- self._betaP * self._E_from_to[m+1, v])
            dexpVm[m] = np.sum(
                dexpVm[m+1:] * # recursion
                np.reciprocal(np.arange(m + 2, self._N + 1, dtype=float)) *
                np.exp(-self._betaP * self._E_from_to[m+1, (m+1):])
            )

        dexpEuv = np.zeros((self._N, self._N))
        # for u in range(self._N):
        #     for v in range(u, self._N):
        #         dexpEuv[u, v] = dexpVm[v] * 1/(v+1) * np.exp(- self._betaP * self._V[u]) # exp(-beta V[1,u-1])
        dexpEuv[:, :] = (dexpVm[np.newaxis, :] *
                         np.reciprocal(np.arange(1.0, self._N + 1, dtype=float))[np.newaxis, :] *
                         np.exp(- self._betaP * self._V[:-1, np.newaxis])
                         )

        dEuv = np.zeros((self._N, self._N), order='F')
        # for u in range(self._N):
        #     for v in range(u, self._N):
        #         dEuv[u, v] = dexpEuv[u, v] * (-self._betaP) * np.exp(- self._betaP * self._E_from_to[u, v]) \
        #                      + (0 if u == 0 else dEuv[u-1, v]) # recursion
        for u in range(self._N):
            dEuv[u, u:] = (
                dexpEuv[u, u:] * (-self._betaP) * np.exp(- self._betaP * self._E_from_to[u, u:])
                + (0 if u == 0 else dEuv[u - 1, u:])  # recursion
            )

        # dEint = np.empty(self._N)
        # for l in range(self._N):
        #     dEint[l] = np.sum(dEuv[l,l:]) # should always be 1.0
        dEint = np.sum(dEuv, axis=1) # should be [1.0, 1.0, ...]

        # force on intermediate beads
        # for l in range(self._N):
        #     F[1:-1, l, :] = dEint[l] * self._spring_force_prefix() * (-self._bead_diff_intra[1:, l] +
        #                                                np.roll(self._bead_diff_intra, axis=0, shift=1)[1:, l])
        F[1:-1, :, :] = dEint[:, np.newaxis] * self._spring_force_prefix() * (-self._bead_diff_intra[1:, :] +
                                            np.roll(self._bead_diff_intra, axis=0, shift=1)[1:, :])

        # force on endpoint beads
        #
        # for l in range(self._N):
        #     acc = np.zeros(3)
        #     acc += dEuv[l, l] * (- self._bead_diff_inter_first_last_bead[l, l])
        #     for v in range(l + 1, self._N):
        #         acc += dEuv[l, v] * (- self._bead_diff_inter_first_last_bead[l + 1, l])
        #     for u in range(l):
        #         acc += dEuv[u, l] * (self._bead_diff_inter_first_last_bead[u + 1, l] +
        #                              (- self._bead_diff_inter_first_last_bead[u, l]))
        #     acc += dEint[l] * self._bead_diff_intra[-1, l]
        #     F[-1, l, :] = self._spring_force_prefix() * acc
        diagonal_force = self._spring_force_prefix() * \
                       (np.diagonal(dEuv) * (np.diagonal(self._bead_diff_inter_first_last_bead, axis1=0, axis2=1))).T
        F[-1, :, :] += -diagonal_force
        F[-1, :-1, :] += self._spring_force_prefix() * \
                         (np.sum(tril_first_axes(
                             dEuv[:-1, :, np.newaxis] *
                             (- np.diagonal(np.transpose(self._bead_diff_inter_first_last_bead, axes=(1,0,2)),
                                            offset=1, axis1=0, axis2=1).T)[:, np.newaxis, :],
                             k=-1),
                             axis=1))
        F[-1, :, :] += self._spring_force_prefix() * \
                       np.sum(tril_first_axes(dEuv[:, :, np.newaxis] *
                                              (- self._bead_diff_inter_first_last_bead +
                                                np.roll(self._bead_diff_inter_first_last_bead, axis=0, shift=-1)),
                                              k=-1),
                              axis=0)
        F[-1, :, :] += self._spring_force_prefix() * \
                       dEint[:, np.newaxis] * self._bead_diff_intra[-1]

        # for l in range(self._N):
        #     acc = np.zeros(3)
        #     acc += dEuv[l, l] * self._bead_diff_inter_first_last_bead[l, l]
        #     for v in range(l + 1, self._N):
        #         acc += dEuv[l, v] * self._bead_diff_inter_first_last_bead[l, v]
        #     if l > 0:
        #         for v in range(l, self._N):
        #             acc += dEuv[l - 1, v] * (self._bead_diff_inter_first_last_bead[l, l - 1] +
        #                                  (- self._bead_diff_inter_first_last_bead[l, v]))
        #     acc += dEint[l] * (- self._bead_diff_intra[0, l])
        #     F[0, l, :] = self._spring_force_prefix() * acc
        F[0, :, :] += diagonal_force
        F[0, :, :] += self._spring_force_prefix() * \
                         (np.sum(tril_first_axes(
                             dEuv[:, :, np.newaxis] * self._bead_diff_inter_first_last_bead,
                             k=-1),
                             axis=1))
        F[0, 1:, :] += self._spring_force_prefix() * \
                       np.sum(tril_first_axes(np.roll(dEuv, axis=0, shift=1)[1:, :, np.newaxis] *
                                              (- self._bead_diff_inter_first_last_bead[1:, :, :] +
                                               np.diagonal(
                                                   self._bead_diff_inter_first_last_bead,
                                                   offset=-1, axis1=0, axis2=1).T[:, np.newaxis, :]),
                                              k=0),
                              axis=1)
        F[0, :, :] += self._spring_force_prefix() * \
                       dEint[:, np.newaxis] * (- self._bead_diff_intra[0])

        return F
    
    def _spring_force_prefix(self):
        return (-1.0) * self._particle_mass * self._spring_freq_squared

    def _spring_potential_prefix(self):
        return 0.5 * self._particle_mass * self._spring_freq_squared

    def _evaluate_cycle_energies(self):
        # using column-major (Fortran order) because uses of the array are mostly within the same column
        Emks = np.zeros((self._N, self._N), dtype=float, order='F')

        intra_spring_energies = np.sum(self._bead_diff_intra ** 2, axis=(0, -1))
        spring_energy_first_last_bead_array = np.sum(self._bead_diff_inter_first_last_bead ** 2, axis=-1)

        # for m in range(self._N):
        #     Emks[m][m] = coefficient * (intra_spring_energies[m] + spring_energy_first_last_bead_array[m, m])
        Emks[np.diag_indices_from(Emks)] = self._spring_potential_prefix() * (intra_spring_energies +
                                                                              np.diagonal(spring_energy_first_last_bead_array))

        for s in range(self._N - 1 - 1, -1, -1):
            # for m in range(s + 1, self._N):
            #     Emks[s][m] = Emks[s + 1][m] + coefficient * (
            #             - spring_energy_first_last_bead_array[s + 1, m]
            #             + intra_spring_energies[s]
            #             + spring_energy_first_last_bead_array[s + 1, s]
            #             + spring_energy_first_last_bead_array[s, m])
            Emks[s, (s + 1):] = Emks[s + 1, (s + 1):] + self._spring_potential_prefix() * (
                    - spring_energy_first_last_bead_array[s + 1, (s + 1):]
                    + intra_spring_energies[s]
                    + spring_energy_first_last_bead_array[s + 1, s]
                    + spring_energy_first_last_bead_array[s, (s + 1):])

        return Emks

    def _evaluate_V_forward(self):
        """
        Evaluate VB_m, m = {0,...,N}. VB0 = 0.0 by definition.
        Evaluation of each VB_m is done using Equation 5 of arXiv:1905.0905.
        Returns all VB_m and all E_m^{(k)} which are required for the forces later.
        """
        V = np.zeros(self._N + 1, float)

        for m in range(1, self._N + 1):
            # For numerical stability. See SI of arXiv:1905.0905
            Elong = min(V[m-1] + self._E_from_to[m - 1, m - 1], V[0] + self._E_from_to[0, m - 1])

            # sig = 0.0
            # for u in range(m):
            #   sig += np.exp(- self._betaP *
            #                (V[u] + self._Ek_N[u, m - 1] - Elong) # V until u-1, then cycle from u to m
            #                 )
            sig = np.sum(np.exp(- self._betaP *
                                (V[:m] + self._E_from_to[:m, m - 1] - Elong)
                                ))
            assert sig != 0.0
            V[m] = Elong - np.log(sig / m) / self._betaP

        return V

    def _evaluate_V_backward(self):
        RV = np.zeros(self._N + 1, float)

        for l in range(self._N - 1, 0, -1):
            # For numerical stability
            Elong = min(self._E_from_to[1, l] + RV[l + 1], self._E_from_to[l, self._N - 1])

            # sig = 0.0
            # for p in range(l, self._N):
            #     sig += 1 / (p + 1) * np.exp(- self._betaP * (self._Ek_N[l, p] + RV[p + 1]
            #                                                 - Elong))
            sig = np.sum(np.reciprocal(np.arange(l + 1.0, self._N + 1)) *
                         np.exp(- self._betaP * (self._E_from_to[l, l:] + RV[l + 1:]
                                                 - Elong)))
            assert sig != 0.0
            RV[l] = Elong - np.log(sig) / self._betaP

        # V^[1,N]
        RV[0] = self._V[-1]

        return RV
