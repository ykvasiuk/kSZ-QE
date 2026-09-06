"""kSZ velocity-reconstruction quadratic estimator (Deutsch et al. 2018 formalism).

* Wigner-3j symbols use ``math.lgamma`` so the numba kernels compile without numba_scipy.
* The reconstruction noise 1/N_L^{vv} is computed with an exact Gauss-Legendre quadrature
  (``nvv_inv``) instead of the brute-force double sum over (l1, l2) of Wigner-3j symbols
  (``Nvv_inv``, kept for reference / cross-checks). See README.md for the derivation.
* ``hp.map2alm`` is called with ``iter=0`` (full-sky, unmasked maps; the iterative quadrature
  refinement changes C_l at the 1e-5 level and costs 7 SHTs instead of 1).
"""
import math  # numba supports math.lgamma natively (no numba_scipy needed)
import numba

import healpy as hp
import numpy as np
from scipy.special import roots_legendre


# --------------------------------------------------------------------------------------- #
# reference implementation: brute-force Wigner-3j double sum, O(L_max * lmax^2)
# --------------------------------------------------------------------------------------- #
@numba.njit
def w3j(j1, j2, j3):
    """(j1 j2 j3; 0 0 0) Wigner-3j symbol."""
    if not ((j1+j2+j3) % 2 == 0) or j3 > j1+j2 or j3 < np.abs(j2-j1):
        return 0
    else:
        g = (j1+j2+j3)/2
    phase = np.power(-1, g)
    return phase*np.exp(0.5*(math.lgamma(2*g-2*j1+1)+math.lgamma(2*g-2*j2+1)+math.lgamma(2*g-2*j3+1)
            - math.lgamma(2*g+2))+math.lgamma(g+1)-(math.lgamma(g-j1+1)+math.lgamma(g-j2+1)+math.lgamma(g-j3+1)))


@numba.njit(parallel=True)
def Nvv_inv(nvv, l_, c_tilde_T, c_tilde_g, cl_tau_g, j1_max, j2_max):
    """1/N_L^{vv} at the multipoles ``l_`` by direct summation. ``nvv`` must be a float array
    (an int array silently truncates the result). Slow: prefer ``nvv_inv``."""
    for k in range(len(nvv)):
        nvv_l = 0
        for j1 in numba.prange(0, j1_max):
            for j2 in numba.prange(0, j2_max):
                nvv_l += np.power(c_tilde_T[j1], -1)*np.power(cl_tau_g[j2], 2)*np.power(c_tilde_g[j2], -1)\
                    * (2*j1+1)*(2*j2+1)*(2*l_[k]+1)/(4*np.pi)*w3j(j1, j2, l_[k])**2
        nvv[k] = nvv_l
    return nvv/(2*l_+1)


# --------------------------------------------------------------------------------------- #
# fast implementation: Gauss-Legendre quadrature, O(nodes * lmax) and exact
# --------------------------------------------------------------------------------------- #
@numba.njit(parallel=True, fastmath=True)
def _legendre_sums(x, w, fT, fg, Lmax):
    """out[L] = sum_i w_i xi_T(x_i) xi_g(x_i) P_L(x_i), with xi(x) = sum_l f_l P_l(x)."""
    n = x.shape[0]
    lmax = fT.shape[0] - 1
    out = np.zeros((n, Lmax + 1))
    for i in numba.prange(n):
        xi = x[i]
        p0 = 1.0
        p1 = xi
        sT = fT[0]*p0 + fT[1]*p1
        sg = fg[0]*p0 + fg[1]*p1
        PL = np.empty(Lmax + 1)
        PL[0] = p0
        PL[1] = p1
        for l in range(2, lmax + 1):
            p0, p1 = p1, ((2*l - 1)*xi*p1 - (l - 1)*p0)/l
            sT += fT[l]*p1
            sg += fg[l]*p1
            if l <= Lmax:
                PL[l] = p1
        for L in range(Lmax + 1):
            out[i, L] = w[i]*sT*sg*PL[L]
    return out.sum(axis=0)


_GL_NODES = {}


def gl_nodes(n):
    """Cached Gauss-Legendre nodes/weights (scipy's roots_legendre; numpy's leggauss is ~5x slower)."""
    if n not in _GL_NODES:
        _GL_NODES[n] = roots_legendre(n)
    return _GL_NODES[n]


def nvv_inv(cl_tilde_T, cl_tilde_g, cl_tau_g, Lmax):
    """1/N_L^{vv} for every L = 0..Lmax.

    1/N_L = sum_{l1 l2} (2l1+1)(2l2+1)/(4pi) (l1 l2 L; 0 0 0)^2 (C^{tau g}_{l2})^2 / (C~^TT_{l1} C~^gg_{l2})
          = 2 pi int_{-1}^{1} dx  xi_T(x) xi_g(x) P_L(x),
    xi_T(x) = sum_l (2l+1)/(4pi) P_l(x) / C~^TT_l,   xi_g(x) = sum_l (2l+1)/(4pi) P_l(x) (C^{tau g}_l)^2 / C~^gg_l,
    using  int_{-1}^{1} P_l1 P_l2 P_L dx = 2 (l1 l2 L; 0 0 0)^2.  The integrand is a polynomial of
    degree 2 lmax + Lmax, so Gauss-Legendre with lmax + Lmax/2 + 2 nodes is exact.
    """
    cl_tilde_T = np.asarray(cl_tilde_T, dtype=float)
    ells = np.arange(len(cl_tilde_T))
    fT = (2*ells + 1)/(4*np.pi)/cl_tilde_T
    fg = (2*ells + 1)/(4*np.pi)*np.divide(np.asarray(cl_tau_g, dtype=float)**2, cl_tilde_g,
                                          out=np.zeros(len(ells)), where=np.asarray(cl_tilde_g) != 0)
    x, w = gl_nodes(len(ells) + Lmax//2 + 2)
    return 2*np.pi*_legendre_sums(x, w, fT, fg, Lmax)


class Estimator():
    def __init__(self, alm_cmb, alm_ksz, alm_cmb_noise, cl_cmb, cl_ksz, cl_cmb_noise, nside, nside_out):
        self.alm_cmb = alm_cmb
        self.alm_ksz = alm_ksz
        self.alm_cmb_noise = alm_cmb_noise
        self.cl_cmb = cl_cmb
        self.cl_ksz = cl_ksz
        self.cl_cmb_noise = cl_cmb_noise
        self.nside = nside
        self.nside_out = nside_out

    @property
    def A_T(self):
        """Inverse-variance filtered temperature map, sum_lm T_lm / C~^TT_l Y_lm (cached)."""
        if self._A_T is None:
            a_lm = self.alm_cmb + self.alm_ksz + self.alm_cmb_noise
            hp.almxfl(a_lm, np.power(self.cl_tilde_T, -1), inplace=True)
            self._A_T = hp.alm2map(a_lm, self.nside)
        return self._A_T

    @property
    def cl_tilde_T(self):
        if self._cl_tilde_T is None:
            self._cl_tilde_T = self.cl_cmb + self.cl_ksz + self.cl_cmb_noise
        return self._cl_tilde_T

    @property
    def alm_cmb_noise(self):
        return self._alm_cmb_noise

    @alm_cmb_noise.setter
    def alm_cmb_noise(self, value):
        self._alm_cmb_noise = value
        self._A_T = None
        self._cl_tilde_T = None

    @property
    def cl_cmb_noise(self):
        return self._cl_cmb_noise

    @cl_cmb_noise.setter
    def cl_cmb_noise(self, value):
        self._cl_cmb_noise = value
        self._A_T = None
        self._cl_tilde_T = None

    def rec_vr(self, delta_g_lm, cl_tilde_g, cl_tau_g):
        """Unnormalized estimate  v_LM^bare = int A_T B_g Y*_LM  and the noise N_L^{vv}
        (L = 0..3 nside_out); the normalized estimate is almxfl(v_bare, N_L)."""
        delta_g_lm_ = hp.almxfl(delta_g_lm, np.divide(cl_tau_g, cl_tilde_g,
                                                      out=np.zeros_like(cl_tau_g), where=cl_tilde_g != 0))
        B_g = hp.alm2map(delta_g_lm_, self.nside)

        Lmax = 3*self.nside_out
        rec_noise = 1.0/nvv_inv(self.cl_tilde_T, cl_tilde_g, cl_tau_g, Lmax)
        vrec_lm_bare = hp.map2alm(self.A_T*B_g, lmax=Lmax, iter=0)
        return vrec_lm_bare, rec_noise
