# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:02:38 2024

@author: as836
"""

import os
import numpy as np
from deer_newkernel import *
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.optimize import nnls, lsq_linear
import deerlab as dl
from TIKR_methods import *
import time

def scmm(data):    
    return (data - data[-1]) / (data.max() - data[-1])

def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')

def ptkn_deriv(X, Y, alpha):
    n = X.shape[1]
    L = np.identity(n)

    A1 = np.concatenate((X, alpha*L))
    b1 = np.concatenate((Y, np.zeros(n)))

    theta, rnorm = nnls(A1, b1)

    rho = np.linalg.norm(Y - np.dot(X, theta))
    eta = np.linalg.norm(np.dot(np.identity(n), theta))

    return theta, rnorm, rho, eta

def fftdl(data, tdat, op = 2):
    _nu, _spec = dl.fftspec(data - data.mean(), tdat, mode = 'real')
    
    if op == 1:
        return _nu, _spec
    else:
        return _spec

# set path for fetching experimental data 
_epath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'
_sep = 30 # observation and pump pulse separation in MHz
_mol = 'HQ' # select molecule tags from HQ (biradical-I), AF (biradical-II) and AQ (biradical-III)
_tag = _mol+'_'+str(_sep)+'MHz'
_expt = np.loadtxt(os.path.join(_epath, _tag+'_dlfit.txt'))


## pre-processing experimental data
_pos = np.where(_expt[:,1] == _expt[:,1].max())[0][0]
_expt = _expt[_pos:]
_expt[:,0] -= _expt[0,0]

_start = time.time()

# define distance domain
_rrange = np.arange(0.5, 2.5, 0.015)

# define P(theta)
_thtab = np.linspace(0, np.pi/2, 100)

## for biradical-I with long pulses
_thmu, _thsm = 1.08, 0.30
_pth = np.exp(-(_thtab - _thmu)**2/(2*_thsm**2))

## for all the other cases
#_pth = np.sin(_thtab)
_pth /= _pth.sum()

# define resonance offset range with a gaussian probability distribution
_wid = 15 # max FWHM for pi-pulse excitation bandwidth (30 ns or longer)
_mult = 4 # range of reson. offsets = _mult x _wid
_omtab = np.linspace(-_mult*_wid, _mult*_wid, 100)
_pom = np.exp(-_omtab**2/(2*_wid**2))
_pom /= _pom.sum()


_omtab2 = np.linspace(-_mult*_wid - _sep, _mult*_wid - _sep, 100)
_pom2 = np.exp(-(_omtab2 + _sep)**2/(2*_wid**2))
_pom2 /= _pom2.sum()

# define pi-pulse magnetic field and duration
_wp2, _wp = 14.71, 14.71 # pulse field in MHz
_tp2, _tp = 17.*1E-3, 34.*1E-3 # pi/2 and pi pulse length in us

# define time-domain parameters
_tau1 = 0.4
_tau2 = 1.8
_ttab = _expt[:,0].copy()

_tdat, _rdat = np.meshgrid(_ttab, _rrange, indexing = "ij")
_tdat, _rdat = np.ndarray.flatten(_tdat), np.ndarray.flatten(_rdat)

# set maximum number of iterations
_mmax = 3*1E3 

# calculate the design matrix
_res = np.zeros((_ttab.shape[0], _rrange.shape[0]))
_m = 1
while _m <= _mmax:
    _om1, _om2 = np.random.choice(_omtab, p = _pom), np.random.choice(_omtab2, p = _pom2)
    
    if _om1==_om2:
        _om2 += 1E-4
    _th = np.random.choice(_thtab, p = _pth)
    
    _dip = 2*np.pi*52.04/_rdat**3 * (1-3*np.cos(_th)**2)

    _sig1 = deer4p(_om1, _om2, _dip, _wp2, _tp2, _wp, _tp, _sep, 
                   _tau1, _tau2, _tdat)
    
    _res -= _sig1.reshape(_ttab.shape[0], _rrange.shape[0])
    _m += 1
    
_res /= int(_mmax)

_end = time.time() - _start

print("SimuLation with %sk iterations took %s sec." %(int(_mmax*1E-3), np.round(_end,2)))

_exptdat = _expt[:,1] / _expt[:,2]
_daty1 = scmm(_exptdat)

# calculate P(r) using Tikhonov regularization
_datx1 = _res.copy()
_dim = _datx1.shape[1]
_dat1 = np.array([scmm(_datx1[:,x]) for x in range(_dim)]).T

_rho, _eta, _prs = TIKR(_dat1, _daty1, 200)
_alphas = np.logspace(-4,2,200)
# change this number to find the L-curve corner
_num = 105
plt.loglog(_rho, _eta)
plt.loglog(_rho[_num], _eta[_num], marker = "o")
plt.show()

_alp = _alphas[_num]
_pr = ptkn_deriv(_dat1, _daty1, _alp)[0]

# calculate the P(w) from the frequency-domain data
_fdat, _daty2 = fftdl(_daty1, _expt[:,0], op = 1)
_daty2 = scmm(_daty2)

_datf1 = np.array([scmm(fftdl(_dat1[:,x], _expt[:,0])) for x in range(_dim)]).T
_rho, _eta, _prs = TIKR(_datf1, _daty2, 200)
# change this number to find the L-curve corner
_num = 125
plt.loglog(_rho, _eta)
plt.loglog(_rho[_num], _eta[_num], marker = "o")
plt.show()

_alp = _alphas[_num]
_pw = ptkn_deriv(_datf1, _daty2, _alp)[0]

# plot the optimal P(r)
plt.plot(_rrange, scmm(movav(_pr)*movav(_pw)))
plt.show()

# save data
_dpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Pr'
np.savetxt(os.path.join(_dpath, _tag + '_Pr-PTH.txt'), np.c_[_rrange, scmm(movav(_pr)*movav(_pw))])
np.savetxt(os.path.join(_dpath, _tag + '_nkfit-PTH.txt'), np.c_[_ttab, scmm(_dat1@(movav(_pr)*movav(_pw)))])