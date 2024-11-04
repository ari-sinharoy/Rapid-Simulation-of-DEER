
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 03:18:16 2024

@author: as836
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def scmm(data):
    return (data - data.min()) / (data.max() - data.min())

def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')

# euler rotation matrix in R3
def rotmat(alp, bet, gam):
    return np.array([[np.cos(alp)*np.cos(gam) - np.cos(bet)*np.sin(alp)*np.sin(gam),
                      np.cos(gam)*np.sin(alp) + np.cos(alp)*np.cos(bet)*np.sin(gam),
                      np.sin(bet)*np.sin(gam)],
                     [-np.cos(bet)*np.cos(gam)*np.sin(alp) - np.cos(alp)*np.sin(gam), 
                      np.cos(alp)*np.cos(bet)*np.cos(gam) - np.sin(alp)*np.sin(gam),
                      np.cos(gam)*np.sin(bet)],
                     [np.sin(alp)*np.sin(bet), -np.cos(alp)*np.sin(bet), np.cos(bet)]])
    
# approximate DEER transition energy expression
def energy(ms, mI, B0, w0, bet, gam, tht, phi):
    _Be = 9.274*1E-24 # in J/T
    _Bn = 5.051*1E-27 # in J/T
    _h = 6.626*1E-34 # in J-s
    _g = np.array([[2.0084, 0, 0],[0, 2.0057, 0], [0, 0, 2.0023]]) # g values
    _A = np.array([[12.6, 0, 0],[0, 12.6, 0], [0, 0, 102.6]]) # A values

    _sx, _sy, _sz = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])

    _R1 = rotmat(gam, bet, 0)
    _R2 = rotmat(phi, tht, 0)
    _gzz = _sz@((_R2@((_R1@_g)@np.linalg.inv(_R1)))@np.linalg.inv(_R2))@_sz
    _geff = (_Be/_h*B0*_gzz - w0)*1E-6
    
    _Anew = ((_R2@((_R1@_A)@np.linalg.inv(_R1)))@np.linalg.inv(_R2))
    
    _Axz = (_sz@_Anew)@_sx
    _Ayz = (_sz@_Anew)@_sy
    _Azz = (_sz@_Anew)@_sz
    
    _wn = _Bn/_h*B0*0.4037*1E-6
    _cms = _Azz - 4*ms*_wn*1E-6
    _lms = np.sqrt(_Axz**2 + _Ayz**2 + _cms**2)
    _energy = ms*(_geff + mI*_lms)

    return _energy

# dipolar Hamiltonian for ms(pum) = +/-1/2
def enD(r, tht):
    return 52.04/r**3*(1-3*np.cos(tht)**2)/2


_wrf1 = 34040.0*1E6 # pump frequency in MHz 
_b0 = 1.2114 # external magnetic field in T
_r = 1.5 # average interspin distance
_seps = [20, 25, 30, 50, 70, 100] # Delta-nu in MHz

# 2D grid of psi and phi
_size = 25
_psi, _phi = np.meshgrid(np.linspace(0, np.pi/2,_size),
                         np.linspace(0, np.pi, _size))

_psi, _phi = np.ndarray.flatten(_psi), np.ndarray.flatten(_phi)
_gam = 0.

_tht = np.linspace(0, np.pi/2, 50) # 1D grid of theta
_lims = [7, 8, 14, 14, 14, 6] # FWHM of excitation bandwidth of the pulses

_tag = r'HQ-ptheta_Dnu-'

_res1 = np.zeros((_tht.shape[0], len(_seps) + 1))

for j in range(_tht.shape[0]):
    _th = _tht[j]
    _w = _wrf1 # _wrf1 for pump, _wrf2 for obs.
    # nitroxide energy of ms=1/2 --> ms=-1/2 transition
    ## for mI=0
    _em0 = np.array(list(
        map(lambda x,y: (energy(-1/2, 0, _b0, _w, x, _gam, _th, y) - 
                         energy(1/2, 0, _b0, _w, x, _gam, _th, y))*np.sin(x), 
            _psi, _phi)))
    
    # for mI = +1
    _em1 = np.array(list(
        map(lambda x,y: (energy(-1/2, 1, _b0, _w, x, _gam, _th, y) - 
                         energy(1/2, 1, _b0, _w, x, _gam, _th, y))*np.sin(x), 
            _psi, _phi)))
    
    # for mI = -1
    _em1m = np.array(list(
        map(lambda x,y: (energy(-1/2, -1, _b0, _w, x, _gam, _th, y) - 
                         energy(1/2, -1, _b0, _w, x, _gam, _th, y))*np.sin(x), 
            _psi, _phi)))
    
    _energies = np.concatenate((np.concatenate((_em0, _em1)), _em1m))
    
    #include the dipolar energy
    _E0 = np.concatenate((_em0 + enD(_r, _th),_em0 - enD(_r, _th)))
    _E1 = np.concatenate((_em1 + enD(_r, _th),_em1 - enD(_r, _th)))
    _E1m = np.concatenate((_em1m + enD(_r, _th),_em1m - enD(_r, _th)))

    # count all the entries that fall within the excitation bandwidth 
    # of the pulse
    
    _res1[j, 0] = _th
    
    for i in range(len(_seps)):
        _lim = _lims[i]
        _cnt = (_E0[abs(_E0)<_lim].shape[0] + 
                _E1[abs(_E1)<_lim].shape[0] + 
                 _E1m[abs(_E1m)<_lim].shape[0])

        _res1[j, i+1] = _cnt    


    # save the data
    _wpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Ptheta_Calculation'
    np.savetxt(os.path.join(_wpath,_tag + 'PUM.txt'), _res1)

# P(theta) for observation pulses
for i in range(len(_seps)):
    _sep = _seps[i]
    _wrf2 = _wrf1 - _sep*1E6 # observation frequency in MHz
    _lim = _lims[i]

    _res2 = np.zeros((_tht.shape[0], 2))
    
    _energies = np.array([])

    for j in range(_tht.shape[0]):
        _th = _tht[j]
        _w = _wrf2 # _wrf1 for pump, _wrf2 for obs.
        # nitroxide energy of ms=1/2 --> ms=-1/2 transition
        ## for mI=0
        _em0 = np.array(list(
            map(lambda x,y: (energy(-1/2, 0, _b0, _w, x, _gam, _th, y) - 
                             energy(1/2, 0, _b0, _w, x, _gam, _th, y))*np.sin(x), 
                _psi, _phi)))
        
        # for mI = +1
        _em1 = np.array(list(
            map(lambda x,y: (energy(-1/2, 1, _b0, _w, x, _gam, _th, y) - 
                             energy(1/2, 1, _b0, _w, x, _gam, _th, y))*np.sin(x), 
                _psi, _phi)))
        
        # for mI = -1
        _em1m = np.array(list(
            map(lambda x,y: (energy(-1/2, -1, _b0, _w, x, _gam, _th, y) - 
                             energy(1/2, -1, _b0, _w, x, _gam, _th, y))*np.sin(x), 
                _psi, _phi)))
        
        _energies = np.concatenate((np.concatenate((_em0, _em1)), _em1m))
        
        #include the dipolar energy
        _E0 = np.concatenate((_em0 + enD(_r, _th),_em0 - enD(_r, _th)))
        _E1 = np.concatenate((_em1 + enD(_r, _th),_em1 - enD(_r, _th)))
        _E1m = np.concatenate((_em1m + enD(_r, _th),_em1m - enD(_r, _th)))

        # count all the entries that fall within the excitation bandwidth 
        # of the pulse
        _cnt = (_E0[abs(_E0)<_lim].shape[0] + 
                _E1[abs(_E1)<_lim].shape[0] + 
                _E1m[abs(_E1m)<_lim].shape[0])
    
        _res2[j, 0] = _th
        _res2[j, 1] = _cnt    
    

    # save the data
    _wpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Ptheta_Calculation'
    np.savetxt(os.path.join(_wpath,_tag + str(_sep) + 'MHz.txt'), _res2)