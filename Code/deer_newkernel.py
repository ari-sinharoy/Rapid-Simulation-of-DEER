# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 04:18:11 2024

@author: as836
"""

import numpy as np

def Cs(om1, om2, b12):
    _omN, _omP = (om1 - om2), (om1 + om2) 
    _bet = np.arctan(b12/_omN)
    _kappa = np.sqrt(_omN**2 + b12**2)
    
    _c1, _c2 = 1/2*(_kappa + _omP), 1/2*(-_kappa + _omP)
    
    return _c1, _c2, _bet

def c5coef(om1, om2, a12, w1, tp1, w2, tp2, sep, tau1, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    _del = (om1**2 + w1**2*np.cos(tp1*np.sqrt(om1**2 + w1**2))) / (om1**2 + w1**2)
    _delp = (om2**2 + w2**2*np.cos(tp2*np.sqrt(om2**2 + w2**2))) / (om2**2 + w2**2)
    
    _c5x1 = -1/2*_del*np.cos(a12*t/2)*(
        8*np.cos(_bet/2)**4*np.sin(_c1*t) + np.sin(_c1*t+_c1*tau1-_c2*tau1) + 
        np.sin(_c2*t-_c1*tau1+_c2*tau1) + np.cos(2*_bet)*(
            np.sin(_c2*tau1-_c1*(t+tau1)) + np.sin(_c1*tau1-_c2*(t+tau1))) + 
        8*np.sin(_c2*t)*np.sin(_bet/2)**4)
    
    _c5y1 = _del*np.cos(a12*t/2)*(
        4*np.cos(_c1*t)*np.cos(_bet/2)**4 + 4*np.cos(_c2*t)*np.sin(_bet/2)**4 + 
        (np.cos(_c2*tau1-_c1*(t+tau1)) + np.cos(_c1*tau1-_c2*(t+tau1)))*np.sin(_bet)**2)
    
    _c5xz = 2*_del*(-1+2*_delp)*np.sin(a12*t/2)*(
        4*np.cos(_c1*t)*np.cos(_bet/2)**4 + 4*np.cos(_c2*t)*np.sin(_bet/2)**4 + 
        (np.cos(_c2*tau1-_c1*(t+tau1)) + np.cos(_c1*tau1-_c2*(t+tau1)))*np.sin(_bet)**2)
    
    _c5yz = _del*(-1+2*_delp)*np.sin(a12*t/2)*(
        8*np.cos(_bet/2)**4*np.sin(_c1*t) + 2*np.cos(1/2*(_c1+_c2)*(t+2*tau1))*np.sin(1/2*(_c1+_c2)*t) + 
        np.cos(2*_bet)*(np.sin(_c2*tau1-_c1*(t+tau1)) + np.sin(_c1*tau1-_c2*(t+tau1))) + 8*np.sin(_c2*t)*np.sin(_bet/2)**4)
    
    _c5x2 = 1/2*_del*np.sin(a12*t/2)*np.sin(_bet)*(
        (2*np.cos(_c1*t) - 2*np.cos(_c2*t) + 2*np.cos(_c2*tau1 - _c1*(t+tau1)) - 
         2*np.cos(_c1*tau1-_c2*(t+tau1)) + np.cos(_c1*t-_bet) + np.cos(_c2*t-_bet) + 
         np.cos(_c1*t+_bet) + np.cos(_c2*t+_bet) - np.cos(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
         np.cos(_c2*t-_c1*tau1+_c2*tau1+_bet) - np.cos(_c2*tau1-_c1*(t+tau1)+_bet) - 
         np.cos(_c1*tau1-_c2*(t+tau1)+_bet))*(1-_delp+_delp*np.cos(2*sep)) - 
        _delp*(2*np.sin(_c1*t) - 2*np.sin(_c2*t) + 2*np.sin(_c1*t+_c1*tau1-_c2*tau1) - 
               2*np.sin(_c2*t-_c1*tau1+_c2*tau1) + np.sin(_c1*t-_bet) + np.sin(_c2*t-_bet) + 
               np.sin(_c1*t+_bet) + np.sin(_c2*t+_bet) - np.sin(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
               np.sin(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.sin(_c2*tau1-_c1*(t+tau1)+_bet) + 
               np.sin(_c1*tau1-_c2*(t+tau1)+_bet))*np.sin(2*sep))
    
    _c5y2 = -1/2*_del*np.sin(a12*t/2)*np.sin(_bet)*(
        (-1+_delp+_delp*np.cos(2*sep))*(
            2*np.sin(_c1*t) - 2*np.sin(_c2*t) + 2*np.sin(_c1*t+_c1*tau1-_c2*tau1) - 
            2*np.sin(_c2*t-_c1*tau1+_c2*tau1) + np.sin(_c1*t-_bet) + np.sin(_c2*t-_bet) + 
            np.sin(_c1*t+_bet) + np.sin(_c2*t+_bet) - np.sin(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
            np.sin(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.sin(_c2*tau1-_c1*(t+tau1)+_bet) + 
            np.sin(_c1*tau1-_c2*(t+tau1)+_bet)) + 
        _delp*(2*np.cos(_c1*t) - 2*np.cos(_c2*t) + 2*np.cos(_c2*tau1-_c1*(t+tau1)) - 
               2*np.cos(_c1*tau1-_c2*(t+tau1)) + np.cos(_c1*t-_bet) + np.cos(_c2*t-_bet) + 
               np.cos(_c1*t+_bet) + np.cos(_c2*t+_bet) - np.cos(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
               np.cos(_c2*t-_c1*tau1+_c2*tau1+_bet) - np.cos(_c2*tau1-_c1*(t+tau1)+_bet) - 
               np.cos(_c1*tau1-_c2*(t+tau1)+_bet))*np.sin(2*sep))
    
    _c5zx = -_del*np.cos(a12*t/2)*np.sin(_bet)*(
        -((1-_delp+_delp*np.cos(2*sep))*(
            2*np.sin(_c1*t) - 2*np.sin(_c2*t) + 2*np.sin(_c1*t+_c1*tau1-_c2*tau1) - 
            2*np.sin(_c2*t-_c1*tau1+_c2*tau1) + np.sin(_c1*t-_bet) + np.sin(_c2*t-_bet) + 
            np.sin(_c1*t+_bet) + np.sin(_c2*t+_bet) - np.sin(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
            np.sin(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.sin(_c2*tau1-_c1*(t+tau1)+_bet) + 
            np.sin(_c1*tau1-_c2*(t+tau1)+_bet))) + 
        _delp*(-2*np.cos(_c1*t) + 2*np.cos(_c2*t) - 2*np.cos(_c2*tau1-_c1*(t+tau1)) + 
               2*np.cos(_c1*tau1-_c2*(t+tau1)) - np.cos(_c1*t-_bet) - np.cos(_c2*t-_bet) - 
               np.cos(_c1*t+_bet) - np.cos(_c2*t+_bet) + np.cos(_c1*t+_c1*tau1-_c2*tau1+_bet) + 
               np.cos(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.cos(_c2*tau1-_c1*(t+tau1)+_bet) + 
               np.cos(_c1*tau1-_c2*(t+tau1)+_bet))*np.sin(2*sep))
    
    _c5zy = -_del*np.cos(a12*t/2)*np.sin(_bet)*(
        (-2*np.cos(_c1*t) + 2*np.cos(_c2*t) - 2*np.cos(_c2*tau1-_c1*(t+tau1)) + 
         2*np.cos(_c1*tau1-_c2*(t+tau1)) - np.cos(_c1*t-_bet) - np.cos(_c2*t-_bet) - 
         np.cos(_c1*t+_bet) - np.cos(_c2*t+_bet) + np.cos(_c1*t+_c1*tau1-_c2*tau1+_bet) + 
         np.cos(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.cos(_c2*tau1-_c1*(t+tau1)+_bet) + 
         np.cos(_c1*tau1-_c2*(t+tau1)+_bet))*(-1+_delp+_delp*np.cos(2*sep)) + 
        _delp*(2*np.sin(_c1*t) - 2*np.sin(_c2*t) - 2*np.sin(_c2*tau1-_c1*(t+tau1)) + 
               2*np.sin(_c2*tau1-_c2*(t+tau1)) + np.sin(_c1*t-_bet) + np.sin(_c2*t-_bet) + 
               np.sin(_c1*t+_bet) + np.sin(_c2*t+_bet) - np.sin(_c1*t+_c1*tau1-_c2*tau1+_bet) - 
               np.sin(_c2*t-_c1*tau1+_c2*tau1+_bet) + np.sin(_c2*tau1-_c1*(t+tau1)+_bet) + 
               np.sin(_c1*tau1-_c2*(t+tau1)+_bet))*np.sin(2*sep))
    
    return _c5x1, _c5y1, _c5xz, _c5yz, _c5x2, _c5y2, _c5zx, _c5zy



def c7coef(om1, om2, a12, w1, tp1, w2, tp2, sep, tau1, tau2, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    _del = (om1**2 + w1**2*np.cos(tp1*np.sqrt(om1**2 + w1**2))) / (om1**2 + w1**2)
    _delp = (om2**2 + w2**2*np.cos(tp2*np.sqrt(om2**2 + w2**2))) / (om2**2 + w2**2)
    
    _c5x1, _c5y1, _c5xz, _c5yz, _c5x2, _c5y2, _c5zx, _c5zy = c5coef(om1, om2, a12, w1, tp1, w2, tp2, sep, tau1, t)
    
    _c7x1 = 1/4*(-2*np.sin(1/2*a12*(-t+tau2))*(
        _c5yz*np.cos(_c1*(-t+tau2))*np.cos(_bet/2)**2 + -_c5xz*np.cos(_bet/2)**2*np.sin(_c1*(-t+tau2)) + 
        _c5yz*np.cos(_c2*(t-tau2))*np.sin(_bet/2)**2 - _c5xz*np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2 + 
        2*_c5x2*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) + 
        2*_c5y2*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)) + 
        np.cos(1/2*a12*(-t+tau2))*(
            4*_c5y1*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + 2*_c5y1*np.sin(_c2*(t-tau2)) - 
            2*_c5y1*np.cos(_bet)*np.sin(_c2*(t-tau2)) - _c5zy*np.sin(_c1*(t-tau2))*np.sin(_bet) + 
            _c5zy*np.sin(_c2*(t-tau2))*np.sin(_bet) + np.cos(_c1*(-t+tau2))*(
                2*_c5x1+2*_c5x1*np.cos(_bet)-_c5zx*np.sin(_bet)) + np.cos(_c2*(-t+tau2))*(
                    2*_c5x1-2*_c5x1*np.cos(_bet)+_c5zx*np.sin(_bet))))
    
    _c7y1 = -1/4*(np.cos(1/2*a12*(-t+tau2))*(
        4*_c5x1*np.cos(_bet/2)**2*np.sin(_c1*(-t+tau2)) + 2*_c5x1*np.sin(_c2*(-t+tau2)) - 
        2*_c5x1*np.cos(_bet)*np.sin(_c2*(-t+tau2)) + _c5zx*np.sin(_c1*(t-tau2))*np.sin(_bet) + 
        _c5zx*np.sin(_c2*(-t+tau2))*np.sin(_bet) + np.cos(_c1*(-t+tau2))*(
            2*_c5y1 + 2*_c5y1*np.cos(_bet) - _c5zy*np.sin(_bet)) + np.cos(_c2*(-t+tau2))*(
                2*_c5y1-2*_c5y1*np.cos(_bet)+_c5zy*np.sin(_bet))) + 
                np.sin(1/2*a12*(-t+tau2))*(
                    np.cos(_c1*(-t+tau2))*(_c5xz + _c5xz*np.cos(_bet) - 2*_c5x2*np.sin(_bet)) + 
                    2*(_c5yz*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + 
                       _c5yz*np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2 + _c5y2*np.sin(_c2*(t-tau2))*np.sin(_bet) + 
                       _c5y2*np.sin(_c1*(-t+tau2))*np.sin(_bet) + np.cos(_c2*(-t+tau2))*(
                           _c5xz*np.sin(_bet/2)**2 + _c5x2*np.sin(_bet)))))
                
    _c7xz = 1/2*(np.sin(1/2*a12*(-t+tau2))*(
        4*_c5x1*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + 4*_c5x1*np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2 + 
        _c5zx*np.sin(_c2*(t-tau2))*np.sin(_bet) + _c5zx*np.sin(_c1*(-t+tau2))*np.sin(_bet) + 
        np.cos(_c2*(-t+tau2))*(-2*_c5y1 + 2*_c5y1*np.cos(_bet) - _c5zy*np.sin(_bet)) + 
        np.cos(_c1*(-t+tau2))*(-2*_c5y1 - 2*_c5y1*np.cos(_bet) + _c5zy*np.sin(_bet))) + 
        np.cos(1/2*a12*(-t+tau2))*(
            np.cos(_c1*(-t+tau2))*(_c5xz+_c5xz*np.cos(_bet)-2*_c5x2*np.sin(_bet)) + np.cos(_c2*(-t+tau2))*(
                _c5xz-_c5xz*np.cos(_bet)+2*_c5x2*np.sin(_bet)) + 
            2*(_c5yz*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + _c5y2*np.sin(_c1*(-t+tau2))*np.sin(_bet) + 
               np.sin(_c2*(t-tau2))*(_c5yz*np.sin(_bet/2)**2 + _c5y2*np.sin(_bet)))))
                    
    _c7yz = -1/2*(np.sin(1/2*a12*(-t+tau2))*(
        4*_c5x1*np.cos(_c1*(t-tau2))*np.cos(_bet/2)**2 + 4*_c5y1*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) - 
        2*np.cos(_bet/2)**2*(_c5zx*np.cos(_c1*(-t+tau2)) - _c5zy*np.sin(_c1*(-t+tau2)))*np.sin(_bet/2) + 
        4*_c5x1*np.cos(_c2*(-t+tau2))*np.sin(_bet/2)**2 + 4*_c5y1*np.sin(_c2*(t-tau2))*np.sin(_bet/2)**2 - 
        2*_c5zx*np.cos(_c1*(-t+tau2))*np.cos(_bet/2)*np.sin(_bet/2)**3 + _c5zx*np.cos(_c2*(-t+tau2))*np.sin(_bet) + 
        _c5zy*np.sin(_c2*(t-tau2))*np.sin(_bet) + _c5zy*np.sin(_c1*(-t+tau2))*np.sin(_bet/2)**2*np.sin(_bet)) + 
        np.cos(1/2*a12*(-t+tau2))*(
            np.cos(_c1*(-t+tau2))*(_c5yz+_c5yz*np.cos(_bet)-2*_c5y2*np.sin(_bet)) + 
            2*(-_c5xz*np.cos(_bet/2)**2*np.sin(_c1*(t-tau2)) + _c5xz*np.sin(_c2*(-t+tau2))*np.sin(_bet/2)**2 + 
               _c5x2*np.sin(_c1*(t-tau2))*np.sin(_bet) + _c5x2*np.sin(_c2*(-t+tau2))*np.sin(_bet) + 
               np.cos(_c2*(-t+tau2))*(_c5yz*np.sin(_bet/2)**2 + _c5y2*np.sin(_bet)))))

    _c7x2 = -1/2*(np.cos(1/2*a12*(-t+tau2))*(
        -2*_c5x2*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 + _c5x2*np.cos(_c1*(-t+tau2))*(-1+np.cos(_bet)) - 
        2*_c5y2*np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) - 2*_c5y2*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 + 
        _c5yz*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) - 
        _c5xz*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)) + 
        np.sin(1/2*a12*(-t+tau2))*(
            _c5zy*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 + _c5zx*np.cos(_bet/2)**2*np.sin(_c2*(-t+tau2)) + 
            _c5zy*np.cos(_c1*(t-tau2))*np.sin(_bet/2)**2 - _c5zx*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 + 
            2*_c5x1*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) + 
            2*_c5y1*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)))

    _c7y2 = 1/4*(2*np.sin(1/2*a12*(-t+tau2))*(
        _c5zx*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 + _c5zy*np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + 
        _c5zx*np.cos(_c1*(-t+tau2))*np.sin(_bet/2)**2 + _c5zy*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 - 
        2*_c5y1*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) + 
        2*_c5x1*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)) + 
        np.cos(1/2*a12*(-t+tau2))*(
            4*_c5x2*np.cos(_bet/2)**2*np.sin(_c2*(-t+tau2)) - 4*_c5x2*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 + 
            _c5xz*np.sin(_c1*(t-tau2))*np.sin(_bet) + _c5xz*np.sin(_c2*(-t+tau2))*np.sin(_bet) - 
            np.cos(_c1*(-t+tau2))*(-2*_c5y2 + 2*_c5y2*np.cos(_bet) + _c5yz*np.sin(_bet)) + 
            np.cos(_c2*(-t+tau2))*(2*_c5y2 + 2*_c5y2*np.cos(_bet) + _c5yz*np.sin(_bet))))

    _c7zx = -(np.cos(1/2*a12*(-t+tau2))*(
        _c5zx*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 + _c5zy*np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) + 
        _c5zx*np.cos(_c1*(-t+tau2))*np.sin(_bet/2)**2 + _c5zy*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 - 
        2*_c5y1*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) + 
        2*_c5x1*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)) + 
        np.sin(1/2*a12*(-t+tau2))*(
            -2*_c5y2*np.cos(_c2*(-t+tau2))*np.cos(_bet/2)**2 + _c5y2*np.cos(_c1*(-t+tau2))*(-1+np.cos(_bet)) - 
            2*_c5x2*np.cos(_bet/2)**2*np.sin(_c2*(-t+tau2)) + 2*_c5x2*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 - 
            _c5xz*np.cos(1/2*(_c1+_c2)*(t-tau2))*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(_bet) - 
            _c5yz*np.sin(1/2*(_c1-_c2)*(t-tau2))*np.sin(1/2*(_c1+_c2)*(t-tau2))*np.sin(_bet)))

    _c7zy = 1/2*(np.sin(1/2*a12*(-t+tau2))*(
        -4*_c5y2*np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) - 4*_c5y2*np.sin(_c1*(t-tau2))*np.sin(_bet/2)**2 + 
        _c5yz*np.sin(_c1*(t-tau2))*np.sin(_bet) - _c5yz*np.sin(_c2*(t-tau2))*np.sin(_bet) + 
        np.cos(_c1*(-t+tau2))*(-2*_c5x2+2*_c5x2*np.cos(_bet)+_c5xz*np.sin(_bet)) - np.cos(_c2*(-t+tau2))*(
            2*_c5x2+2*_c5x2*np.cos(_bet)+_c5xz*np.sin(_bet))) +
        np.cos(1/2*a12*(-t+tau2))*(
            _c5zx*np.sin(_c1*(t-tau2)) - _c5zx*np.cos(_bet)*np.sin(_c1*(t-tau2)) + 
            2*_c5zx*np.cos(_bet/2)**2*np.sin(_c2*(t-tau2)) - 2*_c5x1*np.sin(_c1*(t-tau2))*np.sin(_bet) - 
            2*_c5x1*np.sin(_c2*(-t+tau2))*np.sin(_bet) + 
            np.cos(_c1*(-t+tau2))*(-_c5zy+_c5zy*np.cos(_bet)+2*_c5y1*np.sin(_bet)) - 
            np.cos(_c2*(-t+tau2))*(_c5zy+_c5zy*np.cos(_bet)+2*_c5y1*np.sin(_bet))))

    return _c7x1, _c7y1, _c7xz, _c7yz, _c7x2, _c7y2, _c7zx, _c7zy 

def deer4p(om1, om2, a12, w1, tp1, w2, tp2, sep, tau1, tau2, t):
    
    _b12 = -a12/2
    _c1, _c2, _bet = Cs(om1, om2, _b12)

    _del = (om1**2 + w1**2*np.cos(tp1*np.sqrt(om1**2 + w1**2))) / (om1**2 + w1**2)
    _delp = (om2**2 + w2**2*np.cos(tp2*np.sqrt(om2**2 + w2**2))) / (om2**2 + w2**2)
    
    _c7x1, _c7y1, _c7xz, _c7yz, _c7x2, _c7y2, _c7zx, _c7zy = c7coef(om1, om2, a12, w1, tp1, w2, tp2, sep, tau1, tau2, t)
    
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