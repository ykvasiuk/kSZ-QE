import camb
from camb import model
import healpy as hp
import numpy as np
import multiprocessing as mp
from scipy.special import spherical_jn
from scipy.interpolate import interp2d


from scipy.signal import savgol_filter
from dataclasses import dataclass

## a bunch of utilities to estimate power spectra
## estimation is done from maps and from camb. 
@dataclass
class Cl_tilde_gs():
    """Estimates lss angular ps from the map or from CAMB"""
    step: int
    @classmethod    
    def from_alms(cls,alms,step):
        instance = cls(step)
        Cls = hp.alm2cl(alms)
        Cls_smoo = savgol_filter(Cls, 41, 3)
        instance.Cls = Cls_smoo
        return instance
    
    @classmethod    
    def from_map(cls,mapp,step,lmax):
        instance = cls(step)
        Cls = hp.anafast(mapp,lmax=lmax)
        Cls_smoo = savgol_filter(Cls, 41, 3)
        instance.Cls = Cls_smoo
        return instance
    
    @classmethod
    def from_camb(cls,ls,noisecoef,step):
        instance = cls(step)
        cps = CAMBPowerSpectrum()
        z_bin = cambresults.redshift_at_comoving_radial_distance(np.array(s2chi[step])/0.675)
        cps.calculate_ps(z_bin)
        instance.Cls = cps.cl_m_Limber(ls)*b_z(np.mean(z_bin))**2 + np.ones_like(ls)/noisecoef
        return instance
@dataclass    
class Cl_tau_gs():
    """Estimates tau x lss ps from lss and density map. Assumes that $\delta_e \approx delta_m$"""
    step: int
    @classmethod
    def from_maps(cls,halomap,taumap,lmax,step):
        instance = cls(step)
        Cls = hp.anafast(halomap,taumap,lmax=lmax)
        Cls_smoo = savgol_filter(Cls, 41, 3)
        instance.Cls = Cls_smoo
        return instance
    
    @classmethod
    def from_alms(cls,haloalms,densalms,step):
        instance = cls(step)
        prefactor = get_prefactor(cambresults.redshift_at_comoving_radial_distance(np.array(s2chi[step])/0.675))
        Cls = hp.alm2cl(haloalms,densalms)*prefactor
        Cls_smoo = savgol_filter(Cls, 41, 3)
        instance.Cls = Cls_smoo
        return instance
    
    @classmethod
    def from_camb(cls,ls,step):
        instance = cls(step)
        cps = CAMBPowerSpectrum()
        z_bin = cambresults.redshift_at_comoving_radial_distance(np.array(s2chi[step])/0.675)
        cps.calculate_ps(z_bin)
        instance.Cls = cps.cl_tau_g_Limber(ls)*b_z(np.mean(z_bin))
        return instance


class _BaseCosmology():
    def __init__(self):
        self.h = 0.675
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=100*self.h, ombh2=0.022, omch2=0.122)
        self.pars.InitPower.set_params(ns=0.965)
        self.k = np.logspace(-5,1,10_000)
        self.results = camb.get_results(self.pars)

class CAMBPowerSpectrum(_BaseCosmology):
    def __init__(self): 
        super().__init__()   
    
    def calculate_ps(self,z_bin):
        self.z = np.linspace(*z_bin,30)
        self.pars.set_matter_power(redshifts=self.z, kmax=np.max(self.k))
        self.pars.NonLinear = model.NonLinear_both
        self.results = camb.get_results(self.pars)
        _ = self.results.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)
        PK = self.results.get_matter_power_interpolator(hubble_units=False, k_hunit=False);
        self.pk_i = PK.P(self.z,self.k)
        self.log_pk_interpolator = interp2d_stateful(np.log10(self.k),self.z,np.log10(self.pk_i))
        self.r = self.results.comoving_radial_distance(self.z)

        
    def cl_m(self,ls):
        Jl = np.array(parallel_executor(jl(self.k,self.r,prime=False),ls))
        I_l = np.trapz(Jl*np.sqrt(self.pk_i.T),self.z)/(np.max(self.z)-np.min(self.z))
        C_l = (2/np.pi)*np.trapz((np.atleast_2d(self.k)*I_l)**2,self.k)
        return C_l
    
    def cl_v(self,ls,scaled=False):
        a_z = 1/(1+self.z)
        H_z = self.results.hubble_parameter(self.z)
        f_z = (self.results.get_Omega('baryon',self.z)+self.results.get_Omega('cdm',self.z))**0.55

        if not scaled:
            factor = np.outer(np.power(self.k,-1),(H_z*f_z*a_z))
        else:
            factor = np.outer(np.power(self.k,-1),(H_z*f_z/(self.r*self.h)**2))
        Jl_primes = np.array(parallel_executor(jl(self.k,self.r,prime=True),ls))
        I_l_prime = np.trapz(Jl_primes*factor*np.sqrt(self.pk_i.T),self.z)/(np.max(self.z)-np.min(self.z))
        
        Cv_l = (2/np.pi)*np.trapz((np.atleast_2d(self.k)*I_l_prime)**2,self.k)
        return Cv_l
    
    def cl_tau_g(self,ls):
        Jl = np.array(parallel_executor(jl(self.k,self.r,prime=False),ls))
        I_l = np.trapz(Jl*np.sqrt(self.pk_i.T),self.z)/(np.max(self.z)-np.min(self.z))
        Ie_l = np.trapz(Jl*np.sqrt(self.pk_i.T)*electronfactor(self.z)/self.results.hubble_parameter(self.z),self.z)/(np.max(self.z)-np.min(self.z))
        C_l = (2/np.pi)*np.trapz(np.atleast_2d(self.k)**2*I_l*Ie_l*3*1e5,self.k)
        return C_l

    def cl_tau_g_Limber(self,ls):
        p_zl = np.array(parallel_executor(p_of_zl(self.z,self.r,self.log_pk_interpolator),ls))
        #C_l = np.trapz(self.results.hubble_parameter(self.z)/self.r**2*electronfactor(self.z)*10**p_zl,self.z)
        C_l = np.trapz(1/self.r**2*electronfactor(self.z)*10**p_zl,self.z)
        C_l *= 1/((np.max(self.z)-np.min(self.z))**2)
        #C_l *= 1/((np.max(self.z)-np.min(self.z))**2*3*1e5)
        return C_l
    
    
    
    def cl_tau_tau(self,ls):
        Jl = np.array(parallel_executor(jl(self.k,self.r,prime=False),ls))
        #I_l = np.trapz(Jl*np.sqrt(self.pk_i.T),self.z)/(np.max(self.z)-np.min(self.z))
        Ie_l = np.trapz(Jl*np.sqrt(self.pk_i.T)*electronfactor(self.z)/self.results.hubble_parameter(self.z),self.z)/(np.max(self.z)-np.min(self.z))
        C_l = (2/np.pi)*np.trapz(np.atleast_2d(self.k)**2*(Ie_l*3*1e5)**2,self.k)
        return C_l

    def cl_tau_tau_Limber(self,ls):
        p_zl = np.array(parallel_executor(p_of_zl(self.z,self.r,self.log_pk_interpolator),ls))

        C_l = np.trapz(1/self.r**2*electronfactor(self.z)**2*3*1e5/self.results.hubble_parameter(self.z)*10**p_zl,self.z)
        C_l *= 1/((np.max(self.z)-np.min(self.z))**2)
        #C_l *= 1/((np.max(self.z)-np.min(self.z))**2*3*1e5)
        return C_l
    
    def cl_m_Limber(self,ls):
        p_zl = np.array(parallel_executor(p_of_zl(self.z,self.r,self.log_pk_interpolator),ls))
        C_l = np.trapz(self.results.hubble_parameter(self.z)/self.r**2*10**p_zl,self.z)
        C_l *= 1/((np.max(self.z)-np.min(self.z))**2*3*1e5)
        return C_l

def electronfactor(zs):
    thompson_SI = 6.6524e-29
    meterToMegaparsec = 3.241e-23
    aas = 1./(1.+zs)
    factor = thompson_SI*ne0z()*aas**(-2.)/meterToMegaparsec#*2.725*10.**6
    return factor

def ne0z():
    # from Mortiz's code, true for z < 3 (we have z<3 in Yuukis sims)
    G_SI = 6.674e-11   
    mProton_SI = 1.673e-27
    H100_SI = 3.241e-18
    chi = 0.86
    me = 1.14
    gasfrac = 0.9
    ombh2 = 0.022
    omgh2 = gasfrac*ombh2
    ne0_SI = chi*omgh2 * 3.*(H100_SI**2.)/mProton_SI/8./np.pi/G_SI/me                   
    return ne0_SI    

class p_of_zl():
    def __init__(self,zs,rs,intrp_obj):
        self.zs = zs
        self.rs = rs
        self.intrp_obj = intrp_obj
    def __call__(self,l):
        return np.fliplr(self.intrp_obj(np.log10(l/self.rs),self.zs)).diagonal()
    
class jl():
    def __init__(self,k,r,prime):
        self.k = k
        self.r = r
        self.prime = prime
        
    def __call__(self,index):
        return spherical_jn(index,np.outer(self.k,self.r),self.prime)

def parallel_executor(func,iterables):    
    with mp.Pool() as p:
        result = p.map(func,iterables)
    return result

class interp2d_stateful:
    """
    scipy.interp2d wrapper that enables picklability
    for saving or parallel execution
    """
    def __init__(self, xs, ys, zs, **kwargs):
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.other_kwargs = kwargs
        self.func = interp2d(xs, ys, zs, **kwargs)

    def __call__(self, x, y):
        return self.func(x,y)

    def __getstate__(self):
        return self.xs, self.ys, self.zs, self.other_kwargs

    def __setstate__(self, state):
        self.func = interp2d(state[0], state[1], state[2], **state[3])