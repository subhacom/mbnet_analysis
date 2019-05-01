# plot_ggn_deflection.py --- 
# Author: Subhasis  Ray
# Created: Mon Jul 30 18:01:46 2018 (-0400)
# Last-Updated: Tue Apr 30 16:58:22 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:
"""Plot the deflection of GGN Vm"""

from __future__ import print_function
import os
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import pint

plt.style.use('grayscale')
fontsize = 8
plt.rcParams['figure.figsize'] = (6/2.54, 3.5/2.54)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['lines.linewidth'] = fontsize / 12.
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
# plt.rcParams['text.usetex'] = True

_ur = pint.UnitRegistry()
Q_ = _ur.Quantity

def sectype(cid):
    """Get the SWC structure type (sid) from section name `cid`"""
    if isinstance(cid, bytes):
        cid = cid.decode('utf-8')
    if '.dend[' in cid:
        stype = 3
    elif '.dend_' in cid:
        stype = int(cid.split('_')[-1].split('[')[0])
    elif 'soma' in cid:
        stype = 1
    elif 'axon' in cid:
        stype = 2
    else:
        stype = 0
    return stype

fname =  '../simulated/Fig_2bc/Vm_single_syn_GGN_20170309_sc[0].dend_8[1].2018_09_06__18_14_04.JID0_PID7796.h5'

with h5.File(fname, 'r') as fd:
    alpha = [fd[node].value for node in fd if node.startswith('v_') and sectype(fd[node].attrs['section']) == 8]
    alpha = np.vstack(alpha)
    output_lca = [fd[node].value for node in fd if node.startswith('v_') and (sectype(fd[node].attrs['section']) == 5)]
    output_lca =  np.vstack(output_lca)
    output_mca = [fd[node].value for node in fd if node.startswith('v_') and (sectype(fd[node].attrs['section']) == 6)]
    output_mca = np.vstack(output_mca)
    # The colors are based on 15 color colorblind-friendly palette (neurograph.py).
    mec = {'alpha': '#6db6ff', 'lca': '#db6d00', 'mca': '#006ddb'}
    xpos = {'alpha': 1, 'lca': 2, 'mca': 2}
    fig, ax = plt.subplots()
    alpha_stable =  alpha[:,  -1] + 51
    ca_stable =  np.concatenate([output_lca[:,  -1],  output_mca[:,  -1]]) + 51
    boxplot_data =  [alpha_stable,  ca_stable]
    boxplot_pos = [1, 2]
    vp = ax.violinplot(boxplot_data, boxplot_pos, showmedians=True, points=100)
    plt.setp(vp['cmedians'], color='white')
    ax.tick_params(axis='y', which='right', length=0)
    ax.tick_params(axis='x', which='top', length=0)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.xaxis.set_ticks(boxplot_pos)
    ax.xaxis.set_ticklabels([r'$\alpha$ lobe', 'calyx'])
    [s.set_visible(False) for s in ax.spines.values()]
    fig.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2, wspace=0.1, hspace=0.1)
    fig.frameon = False
    fig.savefig('../figures/{}.deflection.svg'.format(os.path.basename(fname)), dpi=300, transparent=True)
    with plt.style.context(('default')):
        fig_vm,  ax_vm =  plt.subplots()
        time = fd['/time'].value
        vm_clamp = fd['v_GGN_20170309_sc[0].dend_8[1]'].value
        vm_ca =  fd['v_GGN_20170309_sc[0].dend_5[2700]'].value
        ax_vm.plot(time, vm_clamp, label='clamp')
        ax_vm.plot(time, vm_ca, label='calyx')
        ax_vm.tick_params(top=False, right=False)
        ax_vm.set_yticks([-51, -40])
        ax_vm.set_xticks([0, 100, 500])
        ax_vm.legend(bbox_to_anchor=(1.1, 1.0))
        [sp.set_visible(False) for sp in ax_vm.spines.values()]
        fig_vm.subplots_adjust(left=0.2, bottom=0.3, right=0.7)
        fig_vm.set_size_inches(5.0/2.54, 2/2.54)
        fig_vm.savefig('../figures/2018_09_06__18_14_04.JID0_PID7796_vclamp.svg',  transparent=True)
        fig_vm.frameon = False
        fig_vm.savefig('../figures/{}.vm.svg'.format(os.path.basename(fname)), dpi=300, transparent=True)
   
    plt.show()        



# 
# plot_ggn_deflection.py ends here
