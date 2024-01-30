import numba_scipy
import numba
from scipy.special import loggamma
from scipy.interpolate import interp1d
import healpy as hp
import numpy as np

@numba.njit
def w3j(j1,j2,j3):
    if not ((j1+j2+j3) % 2 == 0) or j3 > j1+j2 or j3 < np.abs(j2-j1):
        return 0
    else:
        g = (j1+j2+j3)/2
    phase = np.power(-1,g)   
    return phase*np.exp( 0.5*(loggamma(2*g-2*j1+1)+loggamma(2*g-2*j2+1)+loggamma(2*g-2*j3+1) \
            -loggamma(2*g+2))+loggamma(g+1)-(loggamma(g-j1+1)+loggamma(g-j2+1)+loggamma(g-j3+1)) )

@numba.njit(parallel=True)
def Nvv_inv(nvv,l_, c_tilde_T,c_tilde_g,cl_tau_g,j1_max,j2_max):
    for k in range(len(nvv)):
        nvv_l = 0
        for j1 in numba.prange(0,j1_max):
            for j2 in numba.prange(0,j2_max):
                nvv_l += np.power(c_tilde_T[j1],-1)*np.power(cl_tau_g[j2],2)*np.power(c_tilde_g[j2],-1)\
                *(2*j1+1)*(2*j2+1)*(2*l_[k]+1)/(4*np.pi)*w3j(j1,j2,l_[k])**2
        nvv[k] = nvv_l        
    return nvv/(2*l_+1)

class Estimator():
    def __init__(self,alm_cmb,alm_ksz,alm_cmb_noise,cl_cmb,cl_ksz,cl_cmb_noise,nside,nside_out):
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
        if self._A_T is None:
            a_lm = self.alm_cmb + self.alm_ksz + self.alm_cmb_noise
            hp.almxfl(a_lm,np.power(self.cl_tilde_T,-1),inplace=True)
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
    
    
    def rec_vr(self,delta_g_lm,cl_tilde_g,cl_tau_g):
        
        delta_g_lm_ = hp.almxfl(delta_g_lm,(np.divide(cl_tau_g, cl_tilde_g, out=np.zeros_like(cl_tau_g), where=cl_tilde_g!=0)))
        B_g = hp.alm2map(delta_g_lm_, self.nside)
        
        
        l_ = np.arange(0,3*self.nside_out+5,3)
        l_max = self.cl_tilde_T.shape[0]-1
        nvv_i = Nvv_inv(np.zeros_like(l_),l_,self.cl_tilde_T,cl_tilde_g,cl_tau_g,l_max,l_max)
        rec_noise_inv_interp = interp1d(l_,nvv_i,bounds_error=False)
        
        rec_noise = np.power(rec_noise_inv_interp(np.arange(3*self.nside_out+1)),-1)
        vrec_lm_bare = hp.map2alm(self.A_T*B_g,lmax=3*self.nside_out)
        return vrec_lm_bare, rec_noise