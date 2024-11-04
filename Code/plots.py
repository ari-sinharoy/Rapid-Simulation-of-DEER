# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:26:27 2024

@author: as836
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d 

from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

def my_formatter(x, pos):
    if x.is_integer():
        return str(int(x))
    else:
        return str(x)
    
def scmm(data):
    
    return (data - data.min()) / (data.max() - data.min())

def scmm2(data):
    
    return (data - data[-1]) / (data.max() - data[-1])

def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')
    
formatter = FuncFormatter(my_formatter)

_impath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Images'

r'''
## Fig.2
sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 1.8,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,2, sharex = 'col', sharey = True, figsize = (12,6), dpi = 400)

_seps = ['20', '50', '70', '100']
_mols = ['AF', 'AQ']

_xticks = [[16, 20, 24, 28], [26, 32, 38]]
_xlims = [[15, 30], [25, 40]]

_dlpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'
_nkpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Pr'

_colors = ['royalblue', 'orange', 'green', 'red']
_xpos = [0.33, 0.78]
_ypos = [0.36, 0.36]

for j in range(4):
    for i in range(2):
        _ax = axs[i]
        _mol = _mols[i]
        _fname = _mol+'_'+_seps[j]+'MHZ_Pr.txt'
        
        _dlpr = np.loadtxt(os.path.join(_dlpath,_fname))
        _nkpr = np.loadtxt(os.path.join(_nkpath,_fname))
        
        _dlpr[:,0] *= 10
        _nkpr[:,0] *= 10
        
        g1 = sns.lineplot(x = _dlpr[:,0], y = scmm(_dlpr[:,1]) + j*1.2, 
                          linewidth = 3, alpha = 0.7, ax = _ax)
       
        
        g1 = _ax.fill_between(_nkpr[:,0], j*1.2, scmm(_nkpr[:,1]) + j*1.2, 
                          alpha = 0.5)
        
        _ax.spines[['top', 'right']].set_visible(False)
        _ax.set(yticks = [], xticks = _xticks[i],
                xlim = _xlims[i])
        _ax.yaxis.labelpad = 20
        _ax.spines[['left','bottom']].set_linewidth(4)
        _ax.set_xlabel(r'Distance ($\mathrm{\AA}$)', weight = 'bold')
        _ax.set_ylabel('P(r)', weight = 'bold')
        
        
        fig.text(_xpos[i], _ypos[i] + 0.145*j, r'$\Delta\nu$ = '+_seps[j]+' MHz', 
                 color = _colors[j], fontsize = 22, fontweight = 'bold')
        
plt.tight_layout()
plt.savefig(os.path.join(_impath, 'Fig2-RAW.png'), bbox_inches = 'tight', transparent = True)
plt.show()


## Fig.3
sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 1.7,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,2, sharex = 'col', sharey = False, figsize = (9,7), 
                        gridspec_kw={'width_ratios': [1.2, 0.8]}, dpi = 400)

_seps = ['50', '70']

_xticks = [[0, 0.4, 0.8], [10, 15, 20]]
_xlims = [[-0.02, 0.9], [9, 24]]

_xlabels = [r'Time ($\mu\mathrm{s}$)', r'Distance ($\mathrm{\AA}$)']
_ylabels = ['DEER Signal', 'P(r)']

_dlpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'
_nkpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Pr'

_colors = np.array([['royalblue', 'navy',],
                    ['purple', 'purple'], 
                    ['magenta', 'red']])

_xpos = [0.33, 0.78]
_ypos = [0.36, 0.36]


for j in range(2):
    _fname1 = 'HQ_'+_seps[j]+'MHZ_Pr.txt'
    _fname2 = 'HQ_'+_seps[j]+'MHZ_nkfit.txt'
    _fname3 = 'HQ_'+_seps[j]+'MHZ_dlfit.txt'
    
    _dlpr = np.loadtxt(os.path.join(_dlpath,_fname1))
    _dlpr[:,0] *= 10
    _nkpr = np.loadtxt(os.path.join(_nkpath,_fname1))    
    _nkpr[:,0] *= 10
    _nktd = np.loadtxt(os.path.join(_nkpath,_fname2))
    _dldat = np.loadtxt(os.path.join(_dlpath,_fname3)) 
    _dldat = _dldat[_dldat[:,0]> 0.4]
    _dldat[:,0] -= _dldat[0,0]
    
    _expt = np.c_[_dldat[:,0], _dldat[:,1] - _dldat[:,2]]
    _dltd = np.c_[_dldat[:,0], _dldat[:,3]]

    _vshift1, _vshift2 = 1.2, 1.2    
        
    g1 = sns.lineplot(x = _expt[:,0], y = scmm2(_expt[:,1]) + j*_vshift1,
                      linewidth = 5, color = _colors[0,j], alpha = 0.8, 
                      ax = axs[0])
    
    g1 = sns.lineplot(x = _nktd[:,0], y = scmm2(_nktd[:,1]) + j*_vshift1,
                      linewidth = 5, linestyle = '-', color = _colors[2,j], alpha = 0.55, 
                      ax = axs[0])
    
    g2 = sns.lineplot(x = _dlpr[:,0], y = scmm(_dlpr[:,1]) + j*_vshift2, 
                      linewidth = 4, color = _colors[1,j], alpha = 0.7, 
                      ax = axs[1])

    g2 = axs[1].fill_between(_nkpr[:,0], j*_vshift2, scmm(_nkpr[:,1]) + j*_vshift2, 
                          color = _colors[2,j], edgecolor = _colors[2,j], 
                          linewidth = 3, alpha = 0.5)

    
    for i in range(2):
        _ax = axs[i]
        _ax.spines[['top', 'right']].set_visible(False)
        _ax.spines[['left','bottom']].set_linewidth(4)
        _ax.set(yticks = [], xticks = _xticks[i],
                xlim = _xlims[i])
        _ax.set_xlabel(_xlabels[i], weight = 'bold')
        _ax.set_ylabel(_ylabels[i], weight = 'bold')
        _ax.yaxis.labelpad = 20
        _ax.xaxis.set_major_formatter(formatter)
        
        
    #fig.text(_xpos[i], _ypos[i] + 0.145*j, r'$\Delta\nu$ = '+_seps[j]+' MHz', 
    #         color = _colors[j], fontsize = 22, fontweight = 'bold')
        
plt.tight_layout()
#plt.savefig(os.path.join(_impath, 'Fig3-RAW.png'), bbox_inches = 'tight', transparent = True)
plt.show()


## Fig.3
sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 1.7,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,2, sharex = 'col', sharey = False, figsize = (9,7), 
                        gridspec_kw={'width_ratios': [1.2, 0.8]}, dpi = 400)

_seps = ['50', '70']

_xticks = [[0, 0.4, 0.8], [10, 15, 20]]
_xlims = [[-0.02, 0.9], [9, 24]]

_xlabels = [r'Time ($\mu\mathrm{s}$)', r'Distance ($\mathrm{\AA}$)']
_ylabels = ['DEER Signal', 'P(r)']

_dlpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'
_nkpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Pr'

_colors = np.array([['royalblue', 'navy',],
                    ['purple', 'purple'], 
                    ['magenta', 'red']])

_xpos = [0.33, 0.78]
_ypos = [0.36, 0.36]


for j in range(2):
    _fname1 = 'HQ_'+_seps[j]+'MHZ_Pr.txt'
    _fname2 = 'HQ_'+_seps[j]+'MHZ_nkfit.txt'
    _fname3 = 'HQ_'+_seps[j]+'MHZ_dlfit.txt'
    
    _dlpr = np.loadtxt(os.path.join(_dlpath,_fname1))
    _dlpr[:,0] *= 10
    _nkpr = np.loadtxt(os.path.join(_nkpath,_fname1))    
    _nkpr[:,0] *= 10
    _nktd = np.loadtxt(os.path.join(_nkpath,_fname2))
    _dldat = np.loadtxt(os.path.join(_dlpath,_fname3)) 
    _dldat = _dldat[_dldat[:,0]> 0.4]
    _dldat[:,0] -= _dldat[0,0]
    
    _expt = np.c_[_dldat[:,0], _dldat[:,1] - _dldat[:,2]]
    _dltd = np.c_[_dldat[:,0], _dldat[:,3]]

    _vshift1, _vshift2 = 1.2, 1.2    
        
    g1 = sns.lineplot(x = _expt[:,0], y = scmm2(_expt[:,1]) + j*_vshift1,
                      linewidth = 5, color = _colors[0,j], alpha = 0.8, 
                      ax = axs[0])
    
    g1 = sns.lineplot(x = _nktd[:,0], y = scmm2(_nktd[:,1]) + j*_vshift1,
                      linewidth = 5, linestyle = '-', color = _colors[2,j], alpha = 0.55, 
                      ax = axs[0])
    
    g2 = sns.lineplot(x = _dlpr[:,0], y = scmm(_dlpr[:,1]) + j*_vshift2, 
                      linewidth = 4, color = _colors[1,j], alpha = 0.7, 
                      ax = axs[1])

    g2 = axs[1].fill_between(_nkpr[:,0], j*_vshift2, scmm(_nkpr[:,1]) + j*_vshift2, 
                          color = _colors[2,j], edgecolor = _colors[2,j], 
                          linewidth = 3, alpha = 0.5)

    
    for i in range(2):
        _ax = axs[i]
        _ax.spines[['top', 'right']].set_visible(False)
        _ax.spines[['left','bottom']].set_linewidth(4)
        _ax.set(yticks = [], xticks = _xticks[i],
                xlim = _xlims[i])
        _ax.set_xlabel(_xlabels[i], weight = 'bold')
        _ax.set_ylabel(_ylabels[i], weight = 'bold')
        _ax.yaxis.labelpad = 20
        _ax.xaxis.set_major_formatter(formatter)
        
        
    #fig.text(_xpos[i], _ypos[i] + 0.145*j, r'$\Delta\nu$ = '+_seps[j]+' MHz', 
    #         color = _colors[j], fontsize = 22, fontweight = 'bold')
        
plt.tight_layout()
plt.savefig(os.path.join(_impath, 'Fig3-RAW.png'), bbox_inches = 'tight', transparent = True)
plt.show()



# Fig4
_dpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Simulations'
_epath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'

def scmm(data):
    return (data - data[-1]) / (data.max() - data[-1])

sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 2.0,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,1, sharex = True, figsize = (12,6), dpi = 400)

_seps = ['20', '25', '30', '50', '100']
_pis = np.array(['50', '40', '34', '30', '66'])

_stag = 'HQ_'

_xticks = [0, 0.3, 0.6]

_labels = np.array([r'20 MHz, 50 ns',
                    r'25 MHz, 40 ns',
                    r'30 MHz, 34 ns',
                    r'50 MHz, 30 ns',
                    r'100 MHz, 66 ns'])

_vshifts = 0.8

for j in range(5):

    _sep = _seps[j]
    _pi = _pis[j]
    _fname = _stag+_sep+'MHz_'+_pi+'ns.txt'
    _ename = _stag+_sep+'MHz_dlfit.txt'
    
    _sdata = np.loadtxt(os.path.join(_dpath, _fname))
    _sdata[:,1] = scmm(_sdata[:,1])
    _edata = np.loadtxt(os.path.join(_epath, _ename))
    _edata = _edata[_edata[:,0] > 0.4]
    _edata[:,0] -= _edata[0,0]
    _edaty = scmm(_edata[:,1] - _edata[:,2])

    _gap = 5

    g1 = sns.lineplot(x = _sdata[:,0], y = _sdata[:,1] + _vshifts*j, 
                      linewidth = 3, alpha = 0.7, ax = axs,
                      label = _labels[j])

    g1 = sns.scatterplot(x = _edata[::_gap,0], y = _edaty[::_gap] + _vshifts*j,
                         marker = '$\circ$', ec = 'face',
                         alpha = 0.5, s = 400, ax = axs)


    g1.set_xlabel(r'Time ($\mathbf{\mu s}$)', weight = 'bold')
    g1.set_ylabel(ylabel = 'DEER Signal', weight = 'bold')
    g1.set(xticks = _xticks, yticks = [], xlim = [-0.04, 0.75])
    
    #axs.legend(frameon = False, prop = {'size': 24},
    #              labelcolor = 'linecolor')
    
axs.spines[['left','bottom']].set_linewidth(4)
handles, labels = axs.get_legend_handles_labels()
lgd = plt.legend(handles[::-1], labels[::-1], 
                 frameon = False, prop = {'size':26},
                 labelcolor = 'linecolor', handlelength = 0,
                 loc='upper center', bbox_to_anchor=(1.3, 0.9))

axs.spines[['top','right']].set_visible(False)

axs.xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig(os.path.join(_impath, 'Fig4-RAW.png'), bbox_inches = 'tight')
plt.show()
'''

# Fig5
_dpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\NewKernel_Pr'
_epath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\DEERLab_Analysis'

def scmm(data):
    return (data - data.min()) / (data.max() - data.min())

sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 2.0,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,1, sharex = True, figsize = (8,8), dpi = 400)

_seps = ['50', '70', '100']

_stag = 'HQ_'

#_xticks = [0, 0.3, 0.6]

_titles = np.array([r'$\mathbf{\Delta\nu}$ = 50 MHz',
                    r'$\mathbf{\Delta\nu}$ = 70 MHz',
                    r'$\mathbf{\Delta\nu}$ = 100 MHz'])

_vshifts = 1.2

for j in range(3):

    _sep = _seps[j]
    _fname = _stag+_sep+'MHz_Pr-PTH.txt'
    _dname = _stag+_sep+'MHz_Pr.txt'
    
    _sdata = np.loadtxt(os.path.join(_dpath, _fname))
    _sdata[:,0] *= 10
    _sfdata = interp1d(_sdata[:,0], _sdata[:,1])
    _ddata = np.loadtxt(os.path.join(_epath, _dname))
    _ddata[:,1] = scmm(_ddata[:,1])
    _ddata[:,0] *= 10

    _gap = 5

    g1 = axs.fill_between(_ddata[:,0], _vshifts*j,
                          _ddata[:,1] + _vshifts*j, color = 'gray', alpha = 0.4)

    g1 = axs.fill_between(_ddata[:,0], _vshifts*j,
                          scmm(_sfdata(_ddata[:,0])) + _vshifts*j, alpha = 0.7)


    axs.set_xlabel(r'Distance ($\mathbf{\AA}$)', weight = 'bold')
    axs.set_ylabel(ylabel = 'P(r)', weight = 'bold')
    axs.set(xticks = [15, 18, 21, 24], yticks = [], xlim = [10, 25])
    
    fig.text(0.16, 0.42 + j*0.24, _titles[j], fontsize = 22)
        
axs.spines[['left','bottom']].set_linewidth(4)
axs.spines[['top','right']].set_visible(False)
axs.xaxis.set_major_formatter(formatter)
axs.yaxis.labelpad = 12

plt.tight_layout()
plt.savefig(os.path.join(_impath, 'Fig5-RAW.png'), bbox_inches = 'tight')
plt.show()