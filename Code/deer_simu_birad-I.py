# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:25:31 2024

@author: as836
"""

import os
import numpy as np
from deer_newkernel import *
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
import seaborn as sns
import time

def scmm(data):    
    return (data - data.min()) / (data.max() - data.min())

def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')

# background corrected experimental data directory
_epath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'
# calculated P(theta) directory
_thpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Ptheta_Calculation'

# load experimental data
_expt = np.loadtxt(os.path.join(_epath,'HQ_30MHz_dlfit.txt'))
_expt = _expt[_expt[:,0]>0.4]
_expt[:,0] -= _expt[0,0]
_datexp = np.c_[_expt[:,0], _expt[:,1] - _expt[:,2]] # DEER background correction

# define some constants
_uB = 9.274*1E-24
_h = 6.626*1E-34
_g = 2.0023

# molecular parameters

## define relaxation parameters for the two molecules and their proportions

## P(r) for the two molecules
_rrange = np.arange(1.0, 3.0, 0.01)
_mu1, _sm1 = 1.55, 0.03 # in nm
_pr1 = np.exp(-(_rrange - _mu1)**2/(2*_sm1**2))
_pr1 /= _pr1.sum()
_prf = interp1d(_rrange, _pr1)

# define the variables for all the 
_seps = [20, 25, 30, 50, 100]
_thmus = [1.05, 1.05, 1.05, 1.05, 0.95] 
_thsms = [0.22, 0.25, 0.35, 0.45, 0.30]
_wps = [10., 12.5, 14.71, 16.67, 7.58]
_tps = [50, 40, 34, 30, 32, 66]

for k in range(5):
    _start = time.time()

    # other parameters
    _thtab = np.linspace(0, np.pi/2, 100)
    _thmu, _thsm = _thmus[k], _thsms[k] 
    _pth = np.exp(-(_thtab - _thmu)**2/(2*_thsm**2))
    _pth /= _pth.sum()
    
    # pulse parameters
    _sep = _seps[k]
    _wid = 10  
    _mult = 4
    _omtab = np.linspace(-_mult*_wid, _mult*_wid, 100)
    _pom = np.exp(-_omtab**2/(2*_wid**2))
    _pom /= _pom.sum()
    
    
    _omtab2 = np.linspace(-_mult*_wid - _sep, _mult*_wid - _sep, 100)
    _pom2 = np.exp(-(_omtab2 + _sep)**2/(2*_wid**2))
    _pom2 /= _pom2.sum()
    
    _wp2, _wp = _wps[k], _wps[k] # pulse field in rad.MHz
    _tp2, _tp = _tps[k]/2*1E-3, _tps[k]*1E-3 # pulse length in us
    
    # pulse separations in time-domain
    _tau1 = 0.4
    _tau2 = 1.8
    _dt = 0.008
    _tmin = 0.28
    _ttab = _datexp[:,0].copy()
    
    # simulate the signal
    _mmax = 3*1E4
    
    _res = 0
    _m = 1
    while _m <= _mmax:
    
        _om1, _om2 = np.random.choice(_omtab, p = _pom), np.random.choice(_omtab2, p = _pom2)
        
        if _om1==_om2:
            _om2 += 1E-4
    
        _r = np.random.choice(_rrange, p = _pr1)
        _th = np.random.choice(_thtab, p = _pth)
        
        _dip = 2*np.pi*52.04/_r**3 * (1-3*np.cos(_th)**2)
    
        _sig1 = deer4p(_om1, _om2, _dip, _wp2, _tp2, _wp, 
                       _tp, _sep, _tau1, _tau2, _ttab)
        
        _res -= _sig1
        _m += 1
        
    _res /= _mmax
    
    _end = time.time() - _start
    
    print("SimuLation with %sk iterations took %s sec." %(int(_mmax*1E-3), np.round(_end,2)))
    
    plt.plot(_ttab,scmm(movav(_res)), 'r')
    plt.plot(_datexp[:,0], scmm(_datexp[:,1]), 'b')
    plt.show()
    
    # save the simulation
    _wpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Simulations'
    _fname = 'HQ_'+str(_sep)+'MHz_'+str(int(_tps[k]))+'ns.txt'
    
    np.savetxt(os.path.join(_wpath,_fname), np.c_[_ttab, _res])