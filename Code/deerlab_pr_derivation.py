# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:12:44 2024

@author: as836
"""

import deerlab as dl
import numpy as np
import os

# mother directory
_mpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER'

# write data
_wpath = os.path.join(_mpath,'Data\DEERLab_Analysis')

# read data
_dpaths = ['Data\DEER Data_biradical-I',
           'Data\DEER Data_biradical-II',
           'Data\DEER Data_biradical-III']

# molecular tags associated with the files
_mtags = ['HQ', 'AF', 'AQ']

# DEER data recorded with observation-pump frequency separation of
_seps = [[20, 25, 30, 50, 70, 100],
        [20, 25, 30, 50, 70, 100],
        [20, 50, 70, 100]]

# set the distance domain
_rmins = [1.0, 1.5, 2.0]
_rmaxs = [3.0, 3.0, 4.0]
_delrs = [0.01, 0.015, 0.015]

# set pulse separation time parameters
_tmin = 0.28
_tau1 = 0.40 
_tau2s = [1.80, 2.00, 4.00]

# run deerlab to get P(r) and fit background
for i in range(3):
    _dpath = os.path.join(_mpath,_dpaths[i])
    _mtag = _mtags[i]
    _rmin, _rmax, _delr = _rmins[i], _rmaxs[i], _delrs[i] 
    _tau2 = _tau2s[i]

    _jrange = len(_seps[i])
    for j in range(_jrange):
        
        _fname = _mtag+'_'+str(_seps[i][j])+'MHz.DTA'
        _t, _Vexp = dl.deerload(os.path.join(_dpath, _fname))

        # pre-processing
        _Vexp = dl.correctphase(_Vexp) # Phase correction
        _Vexp = _Vexp/np.max(_Vexp)     # Rescaling (aesthetic)
        _t = _t - _t[0]                 # Account for zerotime
        _t = _t + _tmin
    
        # distance vector
        _r = np.arange(_rmin, _rmax, _delr) # nm

        # construct the model
        _Vmodel = dl.dipolarmodel(_t, _r, 
                                  experiment = dl.ex_4pdeer(_tau1, _tau2, 
                                                            pathways = [1]))
        
        _compactness = dl.dipolarpenalty(Pmodel = None, r = _r, type = 'compactness')

        # fit the model to the data
        _results = dl.fit(_Vmodel, _Vexp, penalties = _compactness)

        # background correction
        _Bfcn = lambda mod,conc,reftime: _results.P_scale*(1-mod)*dl.bg_hom3d(
            _t-reftime,conc,mod)
        _Bfit = _results.evaluate(_Bfcn)
        _Vcorr = 1 - _results.mod + _Vexp - _Bfit

        # obtain P(r) 
        _Pr = _results.P
    
        # save the data
        np.savetxt(os.path.join(_wpath, _mtag+'_'+str(_seps[i][j])+'MHz_Pr.txt'), 
                   np.c_[_r, _Pr])
        np.savetxt(os.path.join(_wpath, _mtag+'_'+str(_seps[i][j])+'MHz_dlfit.txt'), 
                   np.c_[_t, _Vexp, _Bfit, _results.model])