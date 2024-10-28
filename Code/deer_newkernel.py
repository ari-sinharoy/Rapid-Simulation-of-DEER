# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 04:18:11 2024

@author: as836
"""

import os
import numpy as np
from scipy.interpolate import interp1d 

def maxscale(data):
    return data / max(data)

def Cs(om1, om2, b12):
    _omN, _omP = (om1 - om2), (om1 + om2) 
    _bet = np.arctan(b12/_omN)
    _kappa = np.sqrt(_omN**2 + b12**2)
    
    _c1, _c2 = 2*np.pi/2*(_kappa + _omP), 2*np.pi/2*(-_kappa + _omP)
    
    return _c1, _c2, _bet

def pulpi(om, w1, tp):
    return 1/2*(1 - (om**2 + w1**2*np.cos(2*np.pi*tp*np.sqrt(om**2 + w1**2))) / (om**2 + w1**2))

def pulpi2(om, w1, tp):
    #return w1*np.sin(2*np.pi*tp*np.sqrt(om**2 + w1**2)) / np.sqrt(om**2 + w1**2)
    _mz = (om**2 + w1**2*np.cos(2*np.pi*tp*np.sqrt(om**2 + w1**2))) / (om**2 + w1**2)
    _mx = -om*w1*(-1 + np.cos(2*np.pi*tp*np.sqrt(om**2 + w1**2))) / (om**2 + w1**2)
    return 1/2*(1 - _mz - abs(_mx))

def c3coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, tau1):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)
    
    # the observation pi/2-pulse
    _dpi21 = pulpi2(om1, wpi2, tpi2)
    _dpi22 = pulpi2(om2, wpi2, tpi2)
    
    # the observation pi-pulse
    _dpi1 = pulpi(om1, wpi, tpi)
    _dpi2 = pulpi(om2, wpi, tpi)
    
    _c3dqy = -4*(_dpi21-_dpi22)*(_dpi1-_dpi2)*np.sin((_c1-_c2)*tau1)*np.sin(_bet)
    _c3dqx = 4*(_dpi21-_dpi22)*(-_dpi2+_dpi1*(-1 + 2*_dpi2))*np.sin(1/2*(_c1-_c2)*tau1)**2*np.sin(2*_bet)
    
    _c3xy, _c3yx, _c3xx, _c3yy = _c3dqy, _c3dqy, _c3dqx, -_c3dqx
    
    _c3x1 = 2*_dpi1*(2*_dpi21*np.cos(a12*tau1/2)*(
        np.cos(_bet/2)**2*np.sin(_c1*tau1) + np.sin(_c2*tau1)*np.sin(_bet/2)**2) + 
        _dpi22*(-np.cos(_c1*tau1) + np.cos(_c2*tau1))*np.sin(a12*tau1/2)*np.sin(_bet))
    
    _c3y1 = 2*_dpi1*(2*_dpi21*np.cos(a12*tau1/2)*(
        np.cos(_bet/2)**2*np.cos(_c1*tau1) + np.cos(_c2*tau1)*np.sin(_bet/2)**2) + 
        _dpi22*(np.sin(_c1*tau1) - np.sin(_c2*tau1))*np.sin(a12*tau1/2)*np.sin(_bet))
    
    _c3xz = -4*_dpi1*(-1 + 2*_dpi2)*(
        2*_dpi21*np.sin(a12*tau1/2)*(
            np.cos(_c1*tau1)*np.cos(_bet/2)**2 + np.cos(_c2*tau1)*np.sin(_bet/2)**2) + 
        _dpi22*np.cos(a12*tau1/2)*(-np.sin(_c1*tau1) + np.sin(_c2*tau1))*np.sin(_bet))
    
    _c3yz = 4*_dpi1*(-1 + 2*_dpi2)*(
        2*_dpi21*np.sin(a12*tau1/2)*(
            np.sin(_c1*tau1)*np.cos(_bet/2)**2 + np.sin(_c2*tau1)*np.sin(_bet/2)**2) + 
        _dpi22*np.cos(a12*tau1/2)*(np.cos(_c1*tau1) - np.cos(_c2*tau1))*np.sin(_bet))
    
    _c3x2 = 2*_dpi2*(2*_dpi22*np.cos(a12*tau1/2)*(
        np.cos(_bet/2)**2*np.sin(_c2*tau1) + np.sin(_c1*tau1)*np.sin(_bet/2)**2) + 
        _dpi21*(-np.cos(_c1*tau1) + np.cos(_c2*tau1))*np.sin(a12*tau1/2)*np.sin(_bet))
    
    _c3y2 = 2*_dpi2*(2*_dpi22*np.cos(a12*tau1/2)*(
        np.cos(_bet/2)**2*np.cos(_c2*tau1) + np.cos(_c1*tau1)*np.sin(_bet/2)**2) + 
        _dpi21*(np.sin(_c1*tau1) - np.sin(_c2*tau1))*np.sin(a12*tau1/2)*np.sin(_bet))

    _c3zx = -4*_dpi2*(-1 + 2*_dpi1)*(
        2*_dpi22*np.sin(a12*tau1/2)*(
            np.cos(_c2*tau1)*np.cos(_bet/2)**2 + np.cos(_c1*tau1)*np.sin(_bet/2)**2) + 
        _dpi21*np.cos(a12*tau1/2)*(-np.sin(_c1*tau1) + np.sin(_c2*tau1))*np.sin(_bet))

    _c3zy = 4*_dpi2*(-1 + 2*_dpi1)*(
        2*_dpi22*np.sin(a12*tau1/2)*(
            np.sin(_c1*tau1)*np.sin(_bet/2)**2 + np.sin(_c2*tau1)*np.cos(_bet/2)**2) + 
        _dpi21*np.cos(a12*tau1/2)*(np.cos(_c1*tau1) - np.cos(_c2*tau1))*np.sin(_bet))

    return _c3x1, _c3y1, _c3xz, _c3yz, _c3x2, _c3y2, _c3zx, _c3zy, _c3xx, _c3xy, _c3yx, _c3yy


def c4coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, tau1, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    _c3x1, _c3y1, _c3xz, _c3yz, _c3x2, _c3y2, _c3zx, _c3zy, _c3xx, _c3xy, _c3yx, _c3yy = c3coef(
        om1, om2, a12, wpi2, tpi2, wpi, tpi, tau1)
    
    _c4x1 = 1/4*(
        -np.sin(1/2*a12*(t+tau1))*(
            2*np.sin(_c1*(t+tau1))*(_c3xz*np.cos(_bet/2)**2 - _c3x2*np.sin(_bet)) + 
            2*np.sin(_c2*(t+tau1))*(_c3xz*np.sin(_bet/2)**2 + _c3x2*np.sin(_bet)) + 
            2*np.cos(_c1*(t+tau1))*(_c3yz*np.cos(_bet/2)**2 - _c3y2*np.sin(_bet)) + 
            2*np.cos(_c2*(t+tau1))*(_c3yz*np.sin(_bet/2)**2 + _c3y2*np.sin(_bet))) + 
        np.cos(1/2*a12*(t+tau1))*(
            np.sin(_c1*(t+tau1))*(-4*_c3y1*np.cos(_bet/2)**2 + _c3zy*np.sin(_bet)) + 
            np.cos(_c1*(t+tau1))*(4*_c3x1*np.cos(_bet/2)**2 - _c3zx*np.sin(_bet)) + 
            np.cos(_c2*(t+tau1))*(4*_c3x1*np.sin(_bet/2)**2 + _c3zx*np.sin(_bet)) + 
            np.sin(_c2*(t+tau1))*(-4*_c3y1*np.sin(_bet/2)**2 - _c3zy*np.sin(_bet))))

    _c4y1 = 1/4*(
        np.sin(1/2*a12*(t+tau1))*(
            2*np.sin(_c1*(t+tau1))*(-_c3yz*np.cos(_bet/2)**2 + _c3y2*np.sin(_bet)) - 
            2*np.sin(_c2*(t+tau1))*(_c3yz*np.sin(_bet/2)**2 + _c3y2*np.sin(_bet)) + 
            2*np.cos(_c1*(t+tau1))*(_c3xz*np.cos(_bet/2)**2 - _c3x2*np.sin(_bet)) + 
            2*np.cos(_c2*(t+tau1))*(_c3xz*np.sin(_bet/2)**2 + _c3x2*np.sin(_bet))) + 
        np.cos(1/2*a12*(t+tau1))*(
            np.sin(_c1*(t+tau1))*(4*_c3x1*np.cos(_bet/2)**2 - _c3zx*np.sin(_bet)) + 
            np.cos(_c1*(t+tau1))*(4*_c3y1*np.cos(_bet/2)**2 - _c3zy*np.sin(_bet)) + 
            np.cos(_c2*(t+tau1))*(4*_c3y1*np.sin(_bet/2)**2 + _c3zy*np.sin(_bet)) + 
            np.sin(_c2*(t+tau1))*(4*_c3x1*np.sin(_bet/2)**2 + _c3zx*np.sin(_bet))))    
    
    _c4xz = 1/2*(np.cos(1/2*a12*(t+tau1))*(
        -2*_c3yz*np.cos(_bet/2)**2*np.sin(_c1*(t+tau1)) - 2*_c3yz*np.sin(_bet/2)**2*np.sin(_c2*(t+tau1)) + 
        2*_c3y2*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
        2*np.cos(_c1*(t+tau1))*(_c3xz*np.cos(_bet/2)**2 - _c3x2*np.sin(_bet)) + 
        2*np.cos(_c2*(t+tau1))*(_c3xz*np.sin(_bet/2)**2 + _c3x2*np.sin(_bet))) - 
        np.sin(1/2*a12*(t+tau1))*(
            np.sin(_c1*(t+tau1))*(4*_c3x1*np.cos(_bet/2)**2 - _c3zx*np.sin(_bet)) + 
            np.sin(_c2*(t+tau1))*(4*_c3x1*np.sin(_bet/2)**2 + _c3zx*np.sin(_bet)) +
            np.cos(_c1*(t+tau1))*(4*_c3y1*np.cos(_bet/2)**2 - _c3zy*np.sin(_bet)) + 
            np.cos(_c2*(t+tau1))*(4*_c3y1*np.sin(_bet/2)**2 + _c3zy*np.sin(_bet))))
    
    _c4yz = 1/2*(np.cos(1/2*a12*(t+tau1))*(
        2*_c3xz*np.cos(_bet/2)**2*np.sin(_c1*(t+tau1)) + 2*_c3xz*np.sin(_bet/2)**2*np.sin(_c2*(t+tau1)) + 
        2*_c3x2*(-np.sin(_c1*(t+tau1)) + np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
        2*np.cos(_c1*(t+tau1))*(_c3yz*np.cos(_bet/2)**2 - _c3y2*np.sin(_bet)) + 
        2*np.cos(_c2*(t+tau1))*(_c3yz*np.sin(_bet/2)**2 + _c3y2*np.sin(_bet))) - 
        np.sin(1/2*a12*(t+tau1))*(
            np.sin(_c1*(t+tau1))*(-4*_c3y1*np.cos(_bet/2)**2 + _c3zy*np.sin(_bet)) + 
            np.sin(_c2*(t+tau1))*(-4*_c3y1*np.sin(_bet/2)**2 - _c3zy*np.sin(_bet)) +
            np.cos(_c1*(t+tau1))*(4*_c3x1*np.cos(_bet/2)**2 - _c3zx*np.sin(_bet)) + 
            np.cos(_c2*(t+tau1))*(4*_c3x1*np.sin(_bet/2)**2 + _c3zx*np.sin(_bet))))    
    
    _c4x2 = 1/4*(np.cos(1/2*a12*(t+tau1))*(
        -4*_c3y2*(np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + np.sin(_c1*(t+tau1))*np.sin(_bet/2)**2) + 
        _c3yz*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) - 
        np.cos(_c1*(t+tau1))*(-4*_c3x2*np.sin(_bet/2)**2 + _c3xz*np.sin(_bet)) + 
        np.cos(_c2*(t+tau1))*(4*_c3x2*np.cos(_bet/2)**2 + _c3xz*np.sin(_bet))) - 
        2*np.sin(1/2*a12*(t+tau1))*(
            _c3zx*np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + 
            (_c3zy*np.cos(_c1*(t+tau1)) + _c3zx*np.sin(_c1*(t+tau1)))*np.sin(_bet/2)**2 - 
            (_c3y1*np.cos(_c1*(t+tau1)) + _c3x1*np.sin(_c1*(t+tau1)) - 
             _c3x1*np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
            1/2*np.cos(_c2*(t+tau1))*(2*_c3zy*np.cos(_bet/2)**2 + 2*_c3y1*np.sin(_bet))))
    
    _c4y2 = 1/4*(np.sin(1/2*a12*(t+tau1))*(
        -2*_c3zy*(np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + np.sin(_c1*(t+tau1))*np.sin(_bet/2)**2) + 
        2*_c3y1*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
        2*np.cos(_c1*(t+tau1))*(_c3zx*np.sin(_bet/2)**2 - _c3x1*np.sin(_bet)) + 
        2*np.cos(_c2*(t+tau1))*(_c3zx*np.cos(_bet/2)**2 + _c3x1*np.sin(_bet))) + 
        np.cos(1/2*a12*(t+tau1))*(
            4*_c3x2*(np.cos(_bet/2)**2*np.cos(_c2*(t+tau1)) + np.sin(_c1*(t+tau1))*np.sin(_bet/2)**2) + 
            _c3xz*(-np.sin(_c1*(t+tau1)) + np.sin(_c2*(t+tau1)))*np.sin(_bet) - 
            np.cos(_c1*(t+tau1))*(-4*_c3y2*np.sin(_bet/2)**2 + _c3yz*np.sin(_bet)) + 
            np.cos(_c2*(t+tau1))*(4*_c3y2*np.cos(_bet/2)**2 + _c3yz*np.sin(_bet))))    
    
    _c4zx = 1/2*(np.cos(1/2*a12*(t+tau1))*(
        -2*_c3zy*np.sin(_bet/2)**2*np.sin(_c1*(t+tau1)) - 2*_c3zy*np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + 
        2*_c3y1*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
        2*np.cos(_c1*(t+tau1))*(_c3zx*np.sin(_bet/2)**2 - _c3x1*np.sin(_bet)) + 
        2*np.cos(_c2*(t+tau1))*(_c3zx*np.cos(_bet/2)**2 + _c3x1*np.sin(_bet))) + 
        np.sin(1/2*a12*(t+tau1))*(
            -4*_c3x2*(np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + np.sin(_c1*(t+tau1))*np.sin(_bet/2)**2) + 
            _c3xz*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) + 
            np.cos(_c1*(t+tau1))*(-4*_c3y2*np.sin(_bet/2)**2 + _c3yz*np.sin(_bet)) - 
            np.cos(_c2*(t+tau1))*(4*_c3y2*np.cos(_bet/2)**2 + _c3yz*np.sin(_bet))))
    
    _c4zy = 1/2*(np.cos(1/2*a12*(t+tau1))*(
        _c3zx*np.sin(_c1*(t+tau1)) + 2*np.cos(_bet/2)**2*np.sin(_c2*(t+tau1))*(
            _c3zx*np.cos(_bet/2) + 2*_c3x1*np.sin(_bet/2)) - 
        np.sin(_c1*(t+tau1))*(_c3zx*np.cos(_bet) + 2*_c3x1*np.sin(_bet)) + 
        2*np.cos(_c1*(t+tau1))*(_c3zy*np.sin(_bet/2)**2 - _c3y1*np.sin(_bet)) + 
        2*np.cos(_c2*(t+tau1))*(_c3zy*np.cos(_bet/2)**2 + _c3y1*np.sin(_bet))) + 
        np.sin(1/2*a12*(t+tau1))*(
            -4*_c3y2*(np.cos(_bet/2)**2*np.sin(_c2*(t+tau1)) + np.sin(_c1*(t+tau1))*np.sin(_bet/2)**2) + 
            _c3yz*(np.sin(_c1*(t+tau1)) - np.sin(_c2*(t+tau1)))*np.sin(_bet) - 
            np.cos(_c1*(t+tau1))*(-4*_c3x2*np.sin(_bet/2)**2 + _c3xz*np.sin(_bet)) - 
            np.cos(_c2*(t+tau1))*(4*_c3x2*np.cos(_bet/2)**2 + _c3xz*np.sin(_bet))))
    
    _c4xx = 1/4*(_c3xx+_c3yy + (_c3xx+_c3yy)*np.cos((_c1-_c2)*(t+tau1)) + 
                 2*(_c3xx-_c3yy)*np.cos((_c1+_c2)*(t+tau1)) - 
                 2*(_c3xx+_c3yy)*np.cos(2*_bet)*np.sin(1/2*(_c1-_c2)*(t+tau1))**2 + 
                 2*(_c3xy-_c3yx)*np.cos(_bet)*np.sin((_c1-_c2)*(t+tau1)) - 
                 2*(_c3xy+_c3yx)*np.sin((_c1+_c2)*(t+tau1)))
    
    _c4yy = 1/4*(_c3xx+_c3yy + (_c3xx+_c3yy)*np.cos((_c1-_c2)*(t+tau1)) + 
                 2*(-_c3xx+_c3yy)*np.cos((_c1+_c2)*(t+tau1)) - 
                 2*(_c3xx+_c3yy)*np.cos(2*_bet)*np.sin(1/2*(_c1-_c2)*(t+tau1))**2 + 
                 2*(_c3xy-_c3yx)*np.cos(_bet)*np.sin((_c1-_c2)*(t+tau1)) + 
                 2*(_c3xy+_c3yx)*np.sin((_c1+_c2)*(t+tau1)))
    
    _c4xy = 1/2*(2*_c3xy*np.cos(_c1*(t+tau1))*np.cos(_c2*(t+tau1)) - 
                 (_c3xx+_c3yy)*np.cos(_bet)*np.sin((_c1-_c2)*(t+tau1)) - 
                 2*_c3yx*np.sin(_c1*(t+tau1))*np.sin(_c2*(t+tau1)) + 
                 (_c3xx-_c3yy)*np.sin((_c1+_c2)*(t+tau1)))
    
    _c4yx = 1/2*(2*_c3yx*np.cos(_c1*(t+tau1))*np.cos(_c2*(t+tau1)) + 
                 (_c3xx+_c3yy)*np.cos(_bet)*np.sin((_c1-_c2)*(t+tau1)) - 
                 2*_c3xy*np.sin(_c1*(t+tau1))*np.sin(_c2*(t+tau1)) + 
                 (_c3xx-_c3yy)*np.sin((_c1+_c2)*(t+tau1)))    
    
    return maxscale(_c4x1), maxscale(_c4y1), maxscale(_c4xz), maxscale(_c4yz), maxscale(_c4x2), maxscale(_c4y2), maxscale(_c4zx), maxscale(_c4zy), maxscale(_c4xx), maxscale(_c4xy), maxscale(_c4yx), maxscale(_c4yy)
    

def c5coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, sep, tau1, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    _c4x1, _c4y1, _c4xz, _c4yz, _c4x2, _c4y2, _c4zx, _c4zy, _c4xx, _c4xy, _c4yx, _c4yy = c4coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, tau1, t)
    
    # the pump pi-pulse
    _dpio1 = pulpi(om1 + sep, wpi, tpi)
    _dpip2 = pulpi(om2 + sep, wpi, tpi)

    _c5x1 = 2*(_c4x1 - _c4x1*_dpio1 + _c4x1*_dpio1*np.cos(2*sep) - _c4y1*_dpio1*np.sin(2*sep))

    _c5y1 = -2*(-_c4y1 + _c4y1*_dpio1 + _c4y1*_dpio1*np.cos(2*sep) + _c4x1*_dpio1*np.sin(2*sep))

    _c5xz = -2*(-1+2*_dpip2)*(_c4xz-_c4xz*_dpio1+_c4xz*_dpio1*np.cos(2*sep) - _c4yz*_dpio1*np.sin(2*sep))
    
    _c5yz = 2*(-1+2*_dpip2)*(-_c4yz+_c4yz*_dpio1+_c4yz*_dpio1*np.cos(2*sep) + _c4xz*_dpio1*np.sin(2*sep))

    _c5x2 = 2*(_c4x2 - _c4x2*_dpip2 + _c4x2*_dpip2*np.cos(2*sep) - _c4y2*_dpip2*np.sin(2*sep))

    _c5y2 = -2*(-_c4y2 + _c4y2*_dpip2 + _c4y2*_dpip2*np.cos(2*sep) + _c4x2*_dpip2*np.sin(2*sep))    
    
    _c5zx = -2*(-1+2*_dpio1)*(_c4zx-_c4zx*_dpip2+_c4zx*_dpip2*np.cos(2*sep) - _c4zy*_dpip2*np.sin(2*sep))
    
    _c5zy = 2*(-1+2*_dpio1)*(-_c4zy+_c4zy*_dpip2+_c4zy*_dpip2*np.cos(2*sep) + _c4zx*_dpip2*np.sin(2*sep))
    
    _c5xx = (_c4yy*_dpio1*_dpip2 + _c4xx*(2-2*_dpip2+_dpio1*(-2+3*_dpip2)) + 
             2*_c4xx*(_dpio1 + _dpip2 - 2*_dpio1*_dpip2)*np.cos(2*sep) + 
             (_c4xx-_c4yy)*_dpio1*_dpip2*np.cos(4*sep) + 2*(
                 _c4yx*_dpio1*(-1+_dpip2) + _c4xy*(-1+_dpio1)*_dpip2)*np.sin(2*sep) - 
             (_c4xy + _c4yx)*_dpio1*_dpip2*np.sin(4*sep))
    
    _c5yy = (_c4xx*_dpio1*_dpip2 + _c4yy*(2-2*_dpip2+_dpio1*(-2+3*_dpip2)) + 
             2*_c4yy*(_dpip2 + _dpio1*(-1+2*_dpip2))*np.cos(2*sep) + 
             (-_c4xx+_c4yy)*_dpio1*_dpip2*np.cos(4*sep) + 2*(
                 _c4xy*_dpio1*(-1+_dpip2) + _c4yx*(-1+_dpio1)*_dpip2)*np.sin(2*sep) + 
             (_c4xy + _c4yx)*_dpio1*_dpip2*np.sin(4*sep))
    
    _c5xy = (_c4xy*(2+_dpio1*(-2+_dpip2) - 2*_dpip2) + _c4yx*_dpio1*_dpip2 + 
             2*_c4xy*(_dpio1 - _dpip2)*np.cos(2*sep) - (_c4xy + _c4yx)*_dpio1*_dpip2*np.cos(4*sep) + 
             2*(_c4yy*_dpio1*(-1+_dpip2) + _c4xx*(-1+_dpio1)*_dpip2)*np.sin(2*sep) + 
             (-_c4xx+_c4yy)*_dpio1*_dpip2*np.sin(4*sep))
    
    _c5yx = (_c4yx*(2+_dpio1*(-2+_dpip2) - 2*_dpip2) + _c4xy*_dpio1*_dpip2 + 
             2*_c4yx*(-_dpio1 + _dpip2)*np.cos(2*sep) - (_c4xy + _c4yx)*_dpio1*_dpip2*np.cos(4*sep) + 
             2*(_c4xx*_dpio1*(-1+_dpip2) + _c4yy*(-1+_dpio1)*_dpip2)*np.sin(2*sep) + 
             (-_c4xx+_c4yy)*_dpio1*_dpip2*np.sin(4*sep))
    
    
    return _c5x1, _c5y1, _c5xz, _c5yz, _c5x2, _c5y2, _c5zx, _c5zy, _c5xx, _c5xy, _c5yx, _c5yy



def c7coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, sep, tau1, tau2, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)
    
    # the observation pi-pulse
    _dpi1 = pulpi(om1, wpi, tpi)
    _dpi2 = pulpi(om2, wpi, tpi) 
    
    _c5x1, _c5y1, _c5xz, _c5yz, _c5x2, _c5y2, _c5zx, _c5zy, _c5xx, _c5xy, _c5yx, _c5yy = c5coef(
        om1, om2, a12, wpi2, tpi2, wpi, tpi, sep, tau1, t)
    
    _c7x1 = 1/4*(np.cos(1/2*a12*(-t+tau2))*(
        4*_c5y1*(np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2) + 
        _c5zy*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
        np.cos(_c1*(-t+tau2))*(4*_c5x1*np.cos(_bet/2)**2 - _c5zx*np.sin(_bet)) + 
        np.cos(_c2*(-t+tau2))*(4*_c5x1*np.sin(_bet/2)**2 + _c5zx*np.sin(_bet))) + 
        np.sin(1/2*a12*(-t+tau2))*(
            2*np.cos(_c2*(-t+tau2))*(-_c5yz*np.sin(_bet/2)**2 - _c5y2*np.sin(_bet)) - 
            2*np.cos(_c1*(-t+tau2))*(_c5yz*np.cos(_bet/2)**2 - _c5y2*np.sin(_bet)) + 
            2*np.sin(_c1*(t-tau2))*(_c5xz*np.cos(_bet/2)**2 - _c5x2*np.sin(_bet)) +
            2*np.sin(_c2*(t-tau2))*(_c5xz*np.sin(_bet/2)**2 + _c5x2*np.sin(_bet))))
    
    _c7y1 = -1/4*(-1+2*_dpi1)*(np.cos(1/2*a12*(-t+tau2))*(
        4*_c5x1*(np.cos(_bet/2)**2*np.sin(_c1*(-t+tau2)) - np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2) + 
        _c5zx*(np.sin(_c1*(t-tau2)) + np.sin(_c2*(-t+tau2)))*np.sin(_bet) + 
        np.cos(_c1*(-t+tau2))*(4*_c5y1*np.cos(_bet/2)**2 - _c5zy*np.sin(_bet)) + 
        np.cos(_c2*(-t+tau2))*(4*_c5y1*np.sin(_bet/2)**2 + _c5zy*np.sin(_bet))) + 
        np.sin(1/2*a12*(-t+tau2))*(
            2*np.cos(_c2*(-t+tau2))*(_c5xz*np.sin(_bet/2)**2 + _c5x2*np.sin(_bet)) + 
            2*np.cos(_c1*(-t+tau2))*(_c5xz*np.cos(_bet/2)**2 - _c5x2*np.sin(_bet)) + 
            2*np.sin(_c1*(t-tau2))*(_c5yz*np.cos(_bet/2)**2 - _c5y2*np.sin(_bet)) +
            2*np.sin(_c2*(t-tau2))*(_c5yz*np.sin(_bet/2)**2 + _c5y2*np.sin(_bet))))
    
    _c7xz = -1/2*(-1+2*_dpi2)*(np.cos(1/2*a12*(-t+tau2))*(
        2*_c5yz*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + 2*_c5yz*np.sin(_bet/2)**2*np.sin(_c2*(t-tau2)) + 
        2*_c5y2*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
        2*np.cos(_c1*(-t+tau2))*(_c5xz*np.cos(_bet/2)**2 - _c5x2*np.sin(_bet)) + 
        2*np.cos(_c2*(-t+tau2))*(_c5xz*np.sin(_bet/2)**2 + _c5x2*np.sin(_bet))) + 
        np.sin(1/2*a12*(-t+tau2))*(
            4*_c5x1*(np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2) + 
            _c5zx*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
            np.cos(_c2*(-t+tau2))*(-4*_c5y1*np.sin(_bet/2)**2 - _c5zy*np.sin(_bet)) + 
            np.cos(_c1*(-t+tau2))*(-4*_c5y1*np.cos(_bet/2)**2 + _c5zy*np.sin(_bet))))

    _c7yz = 1/2*(-1+2*_dpi1)*(-1+2*_dpi2)*(np.cos(1/2*a12*(-t+tau2))*(
        2*_c5xz*np.cos(_bet/2)**2*np.sin(_c1*(-t+tau2)) - 2*_c5xz*np.sin(_bet/2)**2*np.sin(_c2*(t-tau2)) + 
        2*_c5x2*(np.sin(_c1*(t-tau2)) + np.sin(_c2*(-t+tau2)))*np.sin(_bet) + 
        2*np.cos(_c1*(-t+tau2))*(_c5yz*np.cos(_bet/2)**2 - _c5y2*np.sin(_bet)) + 
        2*np.cos(_c2*(-t+tau2))*(_c5yz*np.sin(_bet/2)**2 + _c5y2*np.sin(_bet))) + 
        np.sin(1/2*a12*(-t+tau2))*(
            4*_c5y1*(np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2) + 
            _c5zy*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
            np.cos(_c1*(-t+tau2))*(4*_c5x1*np.cos(_bet/2)**2 - _c5zx*np.sin(_bet)) + 
            np.cos(_c2*(-t+tau2))*(4*_c5x1*np.sin(_bet/2)**2 + _c5zx*np.sin(_bet))))    
    
    _c7x2 = 1/2*(np.cos(1/2*a12*(-t+tau2))*(
        2*np.cos(_c1*(-t+tau2))*_c5x2*np.sin(_bet/2)**2 + 
        2*np.cos(_bet/2)**2*(_c5x2*np.cos(_c2*(-t+tau2)) + _c5y2*np.sin(_c2*(t-tau2))) + 
        2*_c5y2*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 + 
        np.sin(1/2*(_c1-_c2)*(t-tau2))*(
            -_c5yz*np.cos(1/2*(_c1+_c2)*(t-tau2)) + _c5xz*np.sin(1/2*(_c1+_c2)*(t-tau2)))*np.sin(_bet)) + 
        np.sin(1/2*a12*(-t+tau2))*(
            -np.cos(_bet/2)**2*(_c5zy*np.cos(_c2*(-t+tau2)) + _c5zx*np.sin(_c2*(-t+tau2))) + 
            (-_c5zy*np.cos(_c1*(t-tau2)) + _c5zx*np.sin(_c1*(t-tau2)))*np.sin(_bet/2)**2 - 
            2*np.sin(1/2*(_c1-_c2)*(t-tau2))*(
                _c5x1*np.cos(1/2*(_c1+_c2)*(t-tau2)) + _c5y1*np.sin(1/2*(_c1+_c2)*(t-tau2)))*np.sin(_bet)))
    
    _c7y2 = 1/4*(-1+2*_dpi2)*(np.sin(1/2*a12*(-t+tau2))*(
        -2*_c5zx*np.cos(_c1*(-t+tau2))*np.sin(_bet/2)**2 - 2*_c5zx*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 - 
        2*_c5zy*(np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2) + 
        4*np.sin(1/2*(_c1-_c2)*(t-tau2))*(
            _c5y1*np.cos(1/2*(_c1+_c2)*(t-tau2)) - _c5x1*np.sin(1/2*(_c1+_c2)*(t-tau2)))*np.sin(_bet)) + 
        np.cos(1/2*a12*(-t+tau2))*(
            4*_c5x2*(np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2) + 
            _c5xz*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
            np.cos(_c1*(-t+tau2))*(-4*_c5y2*np.sin(_bet/2)**2 + _c5yz*np.sin(_bet)) - 
            np.cos(_c2*(-t+tau2))*(4*_c5y2*np.cos(_bet/2)**2 + _c5yz*np.sin(_bet))))
    
    _c7zx = 1/2*(-1+2*_dpi1)*(np.cos(1/2*a12*(-t+tau2))*(
        _c5zy*(np.cos(_bet)*np.sin(_c1*(t-tau2)) + np.sin(_c1*(-t+tau2)) + 
               2*np.cos(_bet/2)**2*np.sin(_c2*(-t+tau2))) + 
        2*_c5y1*(np.sin(_c1*(t-tau2)) + np.sin(_c2*(-t+tau2)))*np.sin(_bet) + 
        2*np.cos(_c1*(-t+tau2))*(-_c5zx*np.sin(_bet/2)**2 + _c5x1*np.sin(_bet)) - 
        2*np.cos(_c2*(-t+tau2))*(_c5zx*np.cos(_bet/2)**2 + _c5x1*np.sin(_bet))) - 
        np.sin(1/2*a12*(-t+tau2))*(
            4*_c5x2*(np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2) + 
            _c5xz*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
            np.cos(_c1*(-t+tau2))*(-4*_c5y2*np.sin(_bet/2)**2 + _c5yz*np.sin(_bet)) - 
            np.cos(_c2*(-t+tau2))*(4*_c5y2*np.cos(_bet/2)**2 + _c5yz*np.sin(_bet))))
    
    _c7zy = -1/2*(-1+2*_dpi1)*(-1+2*_dpi2)*(np.sin(1/2*a12*(-t+tau2))*(
        2*_c5y2*(np.cos(_bet)*np.sin(_c1*(t-tau2)) + np.sin(_c1*(-t+tau2)) + 
                 2*np.cos(_bet/2)**2*np.sin(_c2*(-t+tau2))) + 
        _c5yz*(np.sin(_c1*(t-tau2)) + np.sin(_c2*(-t+tau2)))*np.sin(_bet) + 
        np.cos(_c1*(-t+tau2))*(-4*_c5x2*np.sin(_bet/2)**2 + _c5xz*np.sin(_bet)) - 
        np.cos(_c2*(-t+tau2))*(4*_c5x2*np.cos(_bet/2)**2 + _c5xz*np.sin(_bet))) + 
        np.cos(1/2*a12*(-t+tau2))*(
            2*_c5zx*(np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2) + 
            2*_c5x1*(np.sin(_c2*(t-tau2)) + np.sin(_c1*(-t+tau2)))*np.sin(_bet) + 
            2*np.cos(_c1*(-t+tau2))*(-_c5zy*np.sin(_bet/2)**2 + _c5y1*np.sin(_bet)) - 
            2*np.cos(_c2*(-t+tau2))*(_c5zy*np.cos(_bet/2)**2 + _c5y1*np.sin(_bet))))

    return _c7x1, _c7y1, _c7xz, _c7yz, _c7x2, _c7y2, _c7zx, _c7zy 

def deer4p(om1, om2, a12, wpi2, tpi2, wpi, tpi, sep, tau1, tau2, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    #_del = (om1**2 + w1**2*np.cos(tp1*np.sqrt(om1**2 + w1**2))) / (om1**2 + w1**2)
    #_delp = (om2**2 + w2**2*np.cos(tp2*np.sqrt(om2**2 + w2**2))) / (om2**2 + w2**2)
    
    _c7x1, _c7y1, _c7xz, _c7yz, _c7x2, _c7y2, _c7zx, _c7zy = c7coef(om1, om2, a12, wpi2, tpi2, wpi, tpi, sep, tau1, tau2, t)
    
    _sig = 1/4*(np.sin(a12*tau2/2)*(
        np.cos(_c1*tau2)*(_c7xz + _c7zx + (_c7xz - _c7zx)*np.cos(_bet) - 2*(_c7x1 + _c7x2)*np.sin(_bet)) + 
        np.cos(_c2*tau2)*(_c7xz + _c7zx + (-_c7xz + _c7zx)*np.cos(_bet) + 2*(_c7x1 + _c7x2)*np.sin(_bet)) - 
        np.sin(_c1*tau2)*(_c7yz + _c7zy + (_c7yz - _c7zy)*np.cos(_bet) - 2*(_c7y1 + _c7y2)*np.sin(_bet)) - 
        np.sin(_c2*tau2)*(_c7yz + _c7zy + (-_c7yz + _c7zy)*np.cos(_bet) + 2*(_c7y1 + _c7y2)*np.sin(_bet))) + 
        np.cos(a12*tau2/2)*(
            np.sin(_c1*tau2)*(2*(_c7x1 + _c7x2 + (_c7x1 - _c7x2)*np.cos(_bet)) - (_c7xz + _c7zx)*np.sin(_bet)) + 
            np.sin(_c2*tau2)*(2*(_c7x1 + _c7x2) + 2*(-_c7x1 + _c7x2)*np.cos(_bet) + (_c7xz + _c7zx)*np.sin(_bet)) + 
            np.cos(_c1*tau2)*(2*(_c7y1 + _c7y2 + (_c7y1 - _c7y2)*np.cos(_bet)) - (_c7yz+_c7zy)*np.sin(_bet)) + 
            np.cos(_c2*tau2)*(2*(_c7y1+_c7y2) + 2*(-_c7y1 + _c7y2)*np.cos(_bet) + (_c7yz+_c7zy)*np.sin(_bet))))
    
    return _sig