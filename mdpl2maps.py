#A bunch of utilities to work with Y. Omori's mdpl2 skysim maps (Agora)

import numpy as np
import healpy as hp
from utils import _BaseCosmology, electronfactor
from dataclasses import dataclass
import re
import pickle

basecosmo = _BaseCosmology()
cambresults = basecosmo.results

with open('/scratch/yurii/s2chi.pkl','rb') as fL:
    s2chi = pickle.load(fL)

def distance_lims(mapname):
    lims = re.findall('_[\d]{1,4}_[\d]{1,4}'+r'\.',mapname)
    return list(map(int,lims[0][1:-1].split('_')))
def find_step(mapname):
    step = re.findall('_[\d]{1,3}_',mapname)
    return int(step[0][1:-1])

@dataclass
class mdpl2_density_map():
    step: int
    dirpath: str
    def __post_init__(self):
        self.chi_bin = np.array(s2chi[self.step]) # in MPc/h
        self.z_bin = cambresults.redshift_at_comoving_radial_distance(self.chi_bin/0.675)
        self.map = hp.read_map(self.dirpath+'/'+f'mdpl2_density_{self.step}_{self.chi_bin[0]}_{self.chi_bin[1]}.fits',dtype=np.float32)
    def overdmap(self):
        return self.map/self.map.mean()-1
    
    def taumap(self):
        factor = np.mean(electronfactor(self.z_bin)/cambresults.hubble_parameter(self.z_bin))*3*1e5*np.diff(self.z_bin)
        return factor*(1+self.overdmap())
@dataclass    
class mdpl2_velocity_map():
    step: int
    dirpath: str
    def __post_init__(self):
        self.chi_bin = np.array(s2chi[self.step]) # in MPc/h
        self.z_bin = cambresults.redshift_at_comoving_radial_distance(self.chi_bin/0.675)
        self.map = hp.read_map(self.dirpath+'/'+f'mdpl2_vlos_{self.step}_{self.chi_bin[0]}_{self.chi_bin[1]}.fits',dtype=np.float32)
    def velmap(self,dens):
        #density is needed here because Agora velocities are actuallly vel*dens, i.e. momenta
        factor = np.mean(cambresults.angular_diameter_distance(self.z_bin))**2
        return factor*np.divide(self.map, dens.map, out=np.zeros_like(self.map), where=dens.map!=0)/(3*1e8) # unscaled unitless
    
@dataclass
class mdpl2_halo_map():
    step: int
    dirpath: str
    def __post_init__(self):
        self.chi_bin = np.array(s2chi[self.step]) # in MPc/h
        self.z_bin = cambresults.redshift_at_comoving_radial_distance(self.chi_bin/0.675)
        self.map = np.load(self.dirpath+'/'+f'halomap_{self.step}.npy')
    def overdmap(self):
        return self.map/self.map.mean()-1
    
def b_z(x):
    a,b,c = [0.24429152, 0.38941906, 1.07659483]
    return a*x**2+b*x+c    
        