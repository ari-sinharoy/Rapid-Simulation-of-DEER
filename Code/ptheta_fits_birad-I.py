# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:33:23 2024

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


def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')
    
formatter = FuncFormatter(my_formatter)

_thpath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Ptheta_Calculation'
_impath = r'C:\Users\as836\Documents\GitHub\Rapid-Simulation-of-DEER\Data\Images' 


_seps = ['20', '25', '30', '50', '70', '100']
_thmus = [1.05, 1.05, 1.06, 1.07, 1.07, 1.00] 
_thsms = [0.22, 0.25, 0.35, 0.38, 0.40, 0.28]
 
_thtab = np.linspace(0, np.pi/2, 100)

sns.set_theme(context='talk', style='white', 
              palette='bright', font_scale = 2.4,
              rc = {"font.weight": "bold"})
#              matplotlib.rc('text', usetex=True),
#              matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"])

fig, axs = plt.subplots(1,2, sharex = True, sharey = True, figsize = (12,12), dpi = 400)

_colors = ['cornflowerblue', 'blueviolet', 'magenta', 'purple', 'darkorange', 'red']

for j in range(6):
    i = j//3
    _ax = axs[i]

    _tag = 'HQ-ptheta_Dnu-'+_seps[j]+'MHz'
    _pthcalA = np.loadtxt(os.path.join(_thpath,_tag+'.txt'))
    _pthcalB = np.loadtxt(os.path.join(_thpath,'HQ-ptheta_Dnu-PUM.txt'))[:,[0,j+1]]
    
    # P(theta) fits
    _thmu, _thsm = _thmus[j], _thsms[j] 
    _pth = np.exp(-(_thtab - _thmu)**2/(2*_thsm**2))
    
    # P(theta) calculated scmm(_pthcalB[:,1])*
    _pthf = interp1d(_pthcalB[:,0], movav(scmm(_pthcalA[:,1])*scmm(_pthcalB[:,1])))

    _jm = j%3
    _vshift = 1.2    
    g1 = sns.lineplot(x = _thtab, y = scmm(np.sin(_thtab)) + _jm*_vshift,
                      linewidth = 14, color = 'gray', alpha = 0.8,
                      ax = _ax)
    
    g2 = _ax.fill_between(_thtab, _jm*_vshift, scmm(_pthf(_thtab)) + _jm*_vshift,
                          color = _colors[j], alpha = 0.4)
    
    #g2 = sns.lineplot(x = _thtab, y = scmm(_pthf(_thtab)) + j*_vshift,
    #                  linewidth = 8, alpha = 0.7, ax = axs)
    
    g3 = sns.lineplot(x = _thtab, y = scmm(_pth) + _jm*_vshift,
                  linewidth = 14, color = _colors[j], alpha = 0.8,
                  ax = _ax)
    _xpos = [0.12, 0.565]
    _ypos = 0.245
    fig.text(_xpos[i], 0.42 + _ypos*_jm, r'$\mathbf{\Delta\nu}$:'+_seps[j]+'$\,$MHz',
             fontsize = 30, fontweight = 'bold')  
    _ax.spines[['top','right']].set_visible(False)
    _ax.set(yticks = [], xticks = [0, np.pi/4, np.pi/2])
    _ax.set_ylabel(r'$\mathbf{P(\theta)}$', weight = 'bold')
    _ax.set_xlabel('Dipolar\nAngle '+r'$\mathbf{\theta\,(\pi)}$', weight = 'bold')
    _ax.spines[['left','bottom']].set_linewidth(4)
    _ax.set_xticklabels(['0', r'$\mathbf{\pi/4}$', r'$\mathbf{\pi/2}$'])

plt.tight_layout()
plt.savefig(os.path.join(_impath,'ptheta_fits.png'),
            bbox_inches = 'tight', transparent = True)
plt.show()


# nitroxide field-swept spectrum
sns.set_theme(context='talk', style='white', 
                  palette='bright', font_scale = 1.8,
                  rc = {"font.weight": "bold"})

fig, axs = plt.subplots(1,1, sharex = True, sharey = True, figsize = (12,2), dpi = 400)

_data = np.loadtxt(os.path.join(_thpath, 'nitroxide_FS.txt'))
_data = _data[((_data[:,0]<=80) & (_data[:,0]>-240))]
_data[:,0] += 34040
#_data[:,0] /= 1E3
_nitrof = interp1d(_data[:,0], _data[:,1])

_xpos = [34040, 34020, 34015, 34010, 33990, 33970, 33940]

g1 = sns.lineplot(x = _data[:,0], y = _data[:,1],
                  color = 'k', linewidth = 5, alpha = 0.4, 
                  ax = axs)


g2 = axs.plot([_xpos[0]], _nitrof(_xpos[0]) + 0.1,
              marker = 'v', markersize = 15,
                     color = 'green')

for j in range(1, 7):
    g2 = axs.plot([_xpos[j]], _nitrof(_xpos[j]),
                         marker = 'o', markersize = 15,
                         color = _colors[j-1])

axs.spines[['top','right','left']].set_visible(False)
axs.spines[['bottom']].set_linewidth(4)
g1.set(yticks = [], ylabel = None)
axs.set_xlabel('Magnetic Field (MHz)', weight = 'bold')

#fig.text(0.65, 0.97, 'Pump', fontsize = 24)

plt.savefig(os.path.join(_impath,'nitroxide_FS.png'),
            bbox_inches = 'tight', transparent = True)
plt.show()