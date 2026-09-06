# kSZ-QE

Quadratic estimator (QE) for reconstructing the large-scale radial ("remote dipole") velocity
field from the kinetic Sunyaev-Zel'dovich (kSZ) effect, cross-correlating a CMB temperature map
with a galaxy / halo density map. Implemented by Yurii Kvasiuk following the formalism of
[Deutsch et al. 2018](https://arxiv.org/abs/1707.08129)
(*Reconstruction of the remote dipole and quadrupole fields from the kSZ and pSZ effects*).

## Estimator

With the inverse-variance filtered temperature and the tau-weighted galaxy field

$$
A_T(\hat n) = \sum_{\ell m} \frac{T_{\ell m}}{\tilde C^{TT}_\ell} Y_{\ell m}(\hat n), \qquad
B_g(\hat n) = \sum_{\ell m} \frac{C^{\tau g}_\ell}{\tilde C^{gg}_\ell}\, g_{\ell m} Y_{\ell m}(\hat n),
$$

the radial velocity in a redshift bin is

$$
\hat v_{LM} = N^{vv}_L \int d^2\hat n \, A_T(\hat n) B_g(\hat n) Y^*_{LM}(\hat n),
\qquad
\frac{1}{N^{vv}_L} = \sum_{\ell_1 \ell_2}
\frac{(C^{\tau g}_{\ell_2})^2}{\tilde C^{TT}_{\ell_1}\tilde C^{gg}_{\ell_2}}
\frac{(2\ell_1+1)(2\ell_2+1)}{4\pi}
\begin{pmatrix}\ell_1 & \ell_2 & L\\ 0 & 0 & 0\end{pmatrix}^2 .
$$

Here $\tilde C^{TT} = C^{\rm CMB} + C^{\rm kSZ} + N^{TT}$ is the total temperature power,
$\tilde C^{gg}$ the (measured, smoothed) galaxy auto-spectrum including shot noise, and
$C^{\tau g}$ the cross-spectrum of the optical depth with the galaxies in the bin (in map units,
i.e. multiplied by $-T_{\rm CMB}$ if $\hat v$ should come out as $v_r/c$).
$N^{vv}_L$ is both the normalization and the reconstruction-noise power spectrum.

## Files

| file | content |
|---|---|
| `estim.py` | `Estimator` class (`A_T` cached; `rec_vr(g_lm, cl_tilde_g, cl_tau_g)` returns the unnormalized $\hat v_{LM}$ and $N^{vv}_L$), the fast `nvv_inv` and the reference `Nvv_inv` / `w3j` |
| `utils.py` | power-spectrum helpers: smoothed $\tilde C^{gg}$, $C^{\tau g}$ from maps or CAMB (Limber), electron-density prefactors |
| `mdpl2maps.py` | loaders for the Agora / MDPL2 density, velocity and halo shells |
| `working_qe-agora_example.ipynb` | end-to-end example on the Agora simulation |

Dependencies: `numpy`, `scipy`, `healpy`, `numba` (and `camb` for `utils.py`).

## Usage

```python
from estim import Estimator

es = Estimator(alm_cmb, alm_ksz, alm_noise, cl_cmb, cl_ksz, cl_noise, nside=2048, nside_out=128)
vrec_lm_bare, N_L = es.rec_vr(alm_g, cl_tilde_g, cl_tau_g)   # N_L for L = 0 .. 3*nside_out
vrec = hp.alm2map(hp.almxfl(vrec_lm_bare, N_L), nside_out)   # normalized v_r estimate
```

`A_T` depends only on the temperature side and is computed once and cached; call `rec_vr` for
each galaxy redshift bin.

## Reconstruction noise via Gauss-Legendre quadrature

The double sum over $(\ell_1,\ell_2)$ with an explicit Wigner-3j symbol for every $L$ costs
$O(L_{\max}\,\ell_{\max}^2)$ evaluations of `lgamma`-heavy 3j symbols (about 8 s per redshift
bin for $\ell_{\max}=5000$, $L_{\max}=384$, even with numba and only every third $L$ evaluated
and interpolated). `estim.nvv_inv` instead uses the Legendre-polynomial identity

$$
\int_{-1}^{1} P_{\ell_1}(x) P_{\ell_2}(x) P_L(x)\, dx
= 2 \begin{pmatrix}\ell_1 & \ell_2 & L\\ 0 & 0 & 0\end{pmatrix}^2 ,
$$

which turns the double sum into a one-dimensional integral of two Legendre transforms,

$$
\frac{1}{N^{vv}_L} = 2\pi \int_{-1}^{1} dx\; \xi_T(x)\, \xi_g(x)\, P_L(x), \qquad
\xi_T(x) = \sum_\ell \frac{2\ell+1}{4\pi}\frac{P_\ell(x)}{\tilde C^{TT}_\ell}, \quad
\xi_g(x) = \sum_\ell \frac{2\ell+1}{4\pi}\frac{(C^{\tau g}_\ell)^2}{\tilde C^{gg}_\ell} P_\ell(x).
$$

The integrand is a polynomial of degree $2\ell_{\max}+L_{\max}$, so Gauss-Legendre quadrature
with $\ell_{\max} + L_{\max}/2 + 2$ nodes is **exact** (not an approximation). Both Legendre
transforms and the $P_L(x_i)$ are built with the three-term recurrence in a single numba kernel
parallel over the quadrature nodes, cost $O(n_{\rm nodes}\,\ell_{\max})$.

Result for $\ell_{\max}=5000$, $L_{\max}=384$: agreement with the brute-force sum to
$\sim 10^{-6}$ (relative); 7 ms per call after the quadrature nodes are cached (the nodes are
computed once with `scipy.special.roots_legendre`, ~0.5 s), versus ~8 s before. $N^{vv}_L$ is
now returned for every $L$, so no interpolation across $L$ is needed. The brute-force
`Nvv_inv` / `w3j` are kept for cross-checks.

Other changes in the same update:

* `hp.map2alm(..., iter=0)` for the full-sky product map $A_T B_g$: the default `iter=3`
  runs seven spherical harmonic transforms instead of one and changes $C_L$ at the $10^{-5}$
  level for unmasked full-sky maps.
* Wigner-3j symbols use `math.lgamma` (numba-native) instead of `scipy.special.loggamma`, so no
  `numba_scipy` extension is needed.
