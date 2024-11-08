# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:44:52 2024

@author: as836
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import nnls

def scmm(data):
    return (data - data.min()) / (data.max() - data.min())

def Lmat(n, op=0):
    if op == 0:
        L = np.identity(n)
    elif op == 1:
        L = np.zeros((n, n))
        L[:-1,1:] = np.identity(n-1)
        L -= np.identity(n)
        L = L[:-1,:]
        
    return L

# Tikhonov rho-eta calculator
def TIKR(A, b, grid = 200):
    # Compute the SVD of K
    U, s, Vh = svd(A, full_matrices = False)
    L = Lmat(A.shape[1], op=1)
    alpha_values = np.logspace(-4, 2, grid)
    
    residual_norms = []
    solution_norms = []
    solution = np.zeros((A.shape[1], grid))
    
    _m = 0
    for alpha in alpha_values:
        # Compute the regularized solution
        x_reg = np.dot(Vh.T, np.dot(np.diag(s / (s**2 + alpha**2)), np.dot(U.T, b)))
    
        # Compute the residual and solution norms
        residual_norms.append(np.linalg.norm(A @ x_reg - b))
        solution_norms.append(np.linalg.norm(L @ x_reg))
        solution[:,_m] = x_reg
        _m += 1
        
    return residual_norms, solution_norms, solution

# Tikhonov solution with a specific regularization parameter
def solTIKR(A, b, alpha):
    # Compute the SVD of K
    U, s, Vh = svd(A, full_matrices = False)
    L = Lmat(A.shape[1], op=1)
    # Compute the regularized solution
    _pr = np.dot(Vh.T, np.dot(np.diag(s / (s**2 + alpha**2)), np.dot(U.T, b)))
        
    return _pr

# Tikhonov with non-negativity constraint
def NNTIKR(K, f, epsilon, alpha):
    A_nn = np.vstack((K, alpha*np.identity(K.shape[1])))
    b_nn = np.hstack((f, np.zeros(K.shape[1])))
    
    A_nne = np.vstack((epsilon * A_nn, np.full(A_nn.shape[1], 1.0)))
    b_nne = np.hstack((epsilon * b_nn, 1.0))
    
    sol, residue = nnls(A_nne, b_nne)
    
    return sol, residue

# maximum curvature method to find an optimal regularization parameter
def maxcurve(A, b, grid = 1000):
    _alphas = np.logspace(-4, 2, grid)
    _rho, _eta, _prs = TIKR(A, b, grid)
    _rp, _rpp = np.gradient(_rho), np.gradient(np.gradient(_rho))
    _ep, _epp = np.gradient(_eta), np.gradient(np.gradient(_eta))
    
    _vals = (_rp*_epp - _rpp*_ep) / (_rp**2 + _ep**2)
    _pos = np.where(_vals == _vals.max())[0][0]
    
    return _alphas[_pos]

# cross-validation method to find an optimal regularization parameter
def CV(A, b, grid = 1000):
    _rho, _eta, _prs = TIKR(A, b, grid)
    _alphas = np.logspace(-4, 2, grid)
    _nt = b.shape[0]
    _n = A.shape[1]
    L = Lmat(A.shape[1], op=1)
    
    _vals = []
    for j in range(grid):
        _alp = _alphas[j]
        _kaL = np.linalg.inv(A.T @ A + _alp**2*L.T @ L) @ A.T
        _HaL = A @ _kaL
        _SaL = scmm(A @ _prs[:,j])
        #_SaL /= _SaL.sum()
        _val = 0
        for i in range(b.shape[0]):
            _val += ((b[i] - _SaL[i])/(1 - _HaL[i,i]))**2
        _vals += [_val]
        
    _pos = np.where(np.array(_vals) == min(_vals))[0][0]

    return _alphas[_pos]    

# general cross-validation method to find an optimal regularization parameter
def GCV(A, b, grid = 1000):
    _rho, _eta, _prs = TIKR(A, b, grid)
    _alphas = np.logspace(-4, 2, grid)
    _nt = b.shape[0]
    _n = A.shape[1]
    L = Lmat(A.shape[1], op=1)
    
    _vals = []
    for j in range(grid):
        _alp = _alphas[j]
        _kaL = np.linalg.inv(A.T @ A + _alp**2*L.T @ L) @ A.T
        _HaL = A @ _kaL
        _SaL = scmm(A @ _prs[:,j])
        #_SaL /= _SaL.sum()
        _vals += [np.linalg.norm(b - _SaL)**2/(1 - _HaL.trace()/_nt)**2]
        
    _pos = np.where(np.array(_vals) == min(_vals))[0][0]

    return _alphas[_pos]    

r'''
# cross-validation methods to find an optimal regularization parameter
def rGCV(A, b, _gam = 0.9, grid = 1000):
    _rho, _eta, _prs = TIKR(A, b, grid)
    _alphas = np.logspace(-4, 2, grid)
    _nt = b.shape[0]
    _n = A.shape[1]
    L = Lmat(A.shape[1], op=1)
    
    _vals = []
    for j in range(grid):
        _alp = _alphas[j]
        _kaL = np.linalg.inv(A.T @ A + _alp**2*L.T @ L) @ A.T
        _HaL = A @ _kaL
        _SaL = scmm(A @ _prs[:,j])
        _vals += [sum(b**2 - _SaL**2)/(1 - _HaL.trace()/_nt)**2 * (
            _gam + (1-_gam)*(_HaL.T@_HaL).trace()/_nt)]
        
    _pos = np.where(np.array(_vals) == min(_vals))[0][0]

    return _alphas[_pos]    
'''

# Akaike information criterion method to find an optimal regularization parameter
def AIC(A, b, grid = 1000):
    _rho, _eta, _prs = TIKR(A, b, grid)
    _alphas = np.logspace(-4, 2, grid)
    _nt = b.shape[0]
    _n = A.shape[1]
    L = Lmat(A.shape[1], op=0)
    
    _vals = []
    for j in range(grid):
        _alp = _alphas[j]
        _kaL = np.linalg.inv(A.T @ A + _alp**2*L.T @ L) @ A.T
        _HaL = A @ _kaL
        _SaL = A @ _prs[:,j]
        _SaL /= _SaL.sum()
        _vals += [_nt*np.log(np.linalg.norm(b - scmm(_SaL))**2/_nt) + 2*_HaL.trace()]
        
    _pos = np.where(np.array(_vals) == min(_vals))[0][0]

    return _alphas[_pos]
    
# residual method to find an optimal regularization parameter
def RM(A, b, grid = 1000):
    _rho, _eta, _prs = TIKR(A, b, grid)
    _alphas = np.logspace(-4, 2, grid)
    #_prs = np.zeros((A.shape[1], grid))
    #for j in range(grid):
    #    _prs[:, j] = NNTIKR(A, b, 0.001, _alphas[j])[0]

    _nt = b.shape[0]
    _n = A.shape[1]
    L = Lmat(A.shape[1], op=0)
    
    _vals = []
    for j in range(grid):
        _alp = _alphas[j]
        _kaL = np.linalg.inv(A.T @ A + _alp**2*L.T @ L) @ A.T
        _HaL = A @ _kaL
        _B = A.T @ (np.identity(_nt) - _HaL)
        _BTB = _B.T @ _B
        _SaL = scmm(A @ _prs[:,j])
        _vals += [np.linalg.norm(b - _SaL)**2 / np.sqrt(_BTB.trace())]
        #_vals += [np.linalg.norm(b - _SaL)**2]
        
    _pos = np.where(np.array(_vals) == min(_vals))[0][0]
    
    return _alphas[_pos]