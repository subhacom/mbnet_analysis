# figure_7efg.py --- 
# Author: Subhasis  Ray
# Created: Wed Dec  5 10:58:13 2018 (-0500)
# Last-Updated: Tue Apr 30 20:01:41 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:

"""Plot reproduction of hyperpolarization of GGN membrane potential.

Requires Python 3.7 or later
"""
import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import mbnetplot

plt.rc('font', size=11)

fig, ax = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='row')

jid = '21841553'   # new simulation 
fname = '../simulated/Fig_7e/fixed_net_UTC2019_03_05__23_50_23-PID17790-JID21841553.h5'
originals = []
with h5.File(fname, 'r') as fd:
    ax[0, 0].set_title('{}'.format(jid))
    print(fname)
    mbnetplot.plot_ggn_vm(ax[1, 0], fd, fd['/data/uniform/ggn_output/GGN_output_Vm'], 'LCA', 1, color='#009292', alpha=1.0)
    ax[1, 0].set_ylim(-60, -45)
    ax[1, 0].set_yticks([-55, -50])
    ig_vm = fd['/data/uniform/ig/IG_Vm']
    dt = ig_vm.attrs['dt']
    ig_vm = ig_vm[0, :]
    t = np.arange(len(ig_vm)) * dt
    ax[2, 0].plot(t, ig_vm, color='#b662ff')
    ax[2, 0].hlines(y=-60.0, xmin=500, xmax=1500, color='gray', lw=10)
    ax[2, 0].hlines(y=-60.0, xmin=1500, xmax=2000, color='lightgray', lw=10)
    for axis in ax.flat:
        [sp.set_visible(False) for sp in axis.spines.values()]
        axis.tick_params(right=False, top=False)    
    ax[1, 0].tick_params(bottom=False)
    xticks = [200.0, 1000.0, 2000.0, 3000.0]    
    ax[2, 0].set_xticks(xticks)
    ax[2, 0].set_xticklabels([x/1000.0 for x in xticks])
    ax[2, 0].set_xlabel('Time (s)')
    ax[2, 0].set_xlim(200.0, 3000.0)
    ax[2, 0].set_ylabel('Membrane potential (mV)')
    x = []
    y = []
    for pn, st in fd['/data/event/pn/pn_spiketime'].items():
        y.append([int(pn.split('_')[-1])] * st.shape[0])
        x.append(st.value)
    ax[0, 0].set_xlabel('Time (ms)')
    ax[0, 0].set_ylabel('PN #')
    ax[0, 0].plot(np.concatenate(x[::10]), np.concatenate(y[::10]), color='#fdb863', marker='s', ms=3, ls='none')
    ax[0, 0].set_xlim(200.0, 2500.0)

jid = '21841558'   # new simulation
fname = '../simulated/Fig_7fg/fixed_net_UTC2019_03_05__23_50_23-PID29025-JID21841558.h5'
with h5.File(fname, 'r') as fd:
    print(fname)
    mbnetplot.plot_ggn_vm(ax[1, 1], fd, fd['/data/uniform/ggn_output/GGN_output_Vm'], 'LCA', 1, color='#009292', alpha=1.0)
    ig_vm = fd['/data/uniform/ig/IG_Vm']
    dt = ig_vm.attrs['dt']
    ig_vm = ig_vm[0, :]
    t = np.arange(len(ig_vm)) * dt
    ax[2, 1].plot(t, ig_vm, color='#b662ff')
    ax[2, 1].hlines(y=-60.0, xmin=500, xmax=1500, color='gray', lw=10)
    ax[2, 1].hlines(y=-60.0, xmin=1500, xmax=2000, color='lightgray', lw=10)
    for axis in ax.flat:
        [sp.set_visible(False) for sp in axis.spines.values()]
        axis.tick_params(right=False, top=False)    
    ax[1, 1].tick_params(bottom=False)
    xticks = [200.0, 1000.0, 2000.0, 3000.0]    
    ax[2, 1].set_xticks(xticks)
    ax[2, 1].set_xticklabels([xt/1000.0 for xt in xticks])
    ax[2, 1].set_xlabel('Time (s)')
    ax[2, 1].set_xlim(200.0, 3000.0)
    ax[2, 1].set_ylabel('Membrane potential (mV)')
    x = []
    y = []
    for pn, st in fd['/data/event/pn/pn_spiketime'].items():
        y.append([int(pn.split('_')[-1])] * st.shape[0])
        x.append(st.value)
    ax[0, 1].set_title(str(jid))
    ax[0, 1].set_xlabel('Time (ms)')
    ax[0, 1].set_ylabel('PN #')
    ax[0, 1].plot(np.concatenate(x[::10]), np.concatenate(y[::10]), color='#fdb863', marker='s', ms=3, ls='none')
    ax[0, 1].set_xlim(200.0, 2500.0)
fig.subplots_adjust(top=0.95, bottom=0.15, left=0.2, right=0.95, hspace=0.1, wspace=0.1)
fig.set_frameon(False)
fig.set_size_inches(9/2.54, 12/2.54)
fig.savefig('../figures/Figure_7efg.svg')
plt.show()

# 
# figure_7efg.py ends here
