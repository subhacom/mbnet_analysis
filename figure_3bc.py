# figure_3bc.py --- 
# Author: Subhasis Ray
# Created: Thu Jun 14 14:15:40 2018 (-0400)
# Last-Updated: Tue Apr 30 20:07:49 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:
"""Plot the results of current injection test on KC with and without
feedback inhibition.

"""

import h5py as h5
import numpy as np
from analytic_wfm import peakdetect
from matplotlib import pyplot as plt
import mbnetplot

plt.rcParams['svg.fonttype'] = 'none'  # Text will be exported as text in SVG format

plt.style.use('grayscale')


nrows = 10
ncols = 2
istart = 0.013  # Starting current just before any spiking
iend = 0.029    # Ending current just when spiking is mangled


## With jitter KC->GGN gmax is regular, but 50,000 synapses with
## delays between 0-60 ms

fpath = '../simulated/Fig_3bc/kc_ggn_amp_sweep_UTC2018_05_04__22_01_15_PID36354_JID283051.h5'

fig_spikecount, axes_spikecount = plt.subplots(frameon=False)

fig_stim, axes_stim = plt.subplots(frameon=False)

with h5.File(fpath, 'r') as fd:
    kc_grp = fd['kc']    
    kcggn_grp = fd['kc_ggn']
    kc_amp_scount = []
    kcggn_amp_scount = []
    amp_nodes = []
    axes_stim.plot(fd['kc/amp_0'][2, :], fd['kc/amp_0'][0, :])    
    fig_stim.suptitle('Current injection')
    for name, node in kc_grp.items():
        fields = [fname.decode() for fname in node.attrs['fields']]
        tidx = fields.index('t')
        vidx = fields.index('kc')
        v = node[vidx, :]
        t = node[tidx, :]
        peaks, troughs = peakdetect(v, x_axis=t, lookahead=20, delta=0.1*np.std(v))
        kc_amp_scount.append((node.attrs['amp'], len(peaks)))
        if (node.attrs['amp'] >=  istart) and (node.attrs['amp'] <= iend):
            amp_nodes.append((node.attrs['amp'], name))
            print(amp_nodes[-1])
    amp_nodes = sorted(amp_nodes, key=lambda x: x[0])
    fig_vm, axes_vm = plt.subplots(nrows=len(amp_nodes), ncols=2, sharex='all',
                                   sharey='all',  frameon=False)
    print('Number of qualifying nodes for Vm plot', len(amp_nodes))
    # Plot the KC Vm
    for ii in range(0, len(amp_nodes)):
        amp, node = amp_nodes[ii]
        kc_ds = kc_grp[node]
        data = {}
        for kk, field in enumerate(kc_ds.attrs['fields']):
            data[field.decode()] = kc_ds[kk, :]
        print(data['t'].max())
        idx = (data['t'] > 400.0) & (data['t'] < 2000.0)
        axes_vm[ii, 0].plot(data['t'][idx], data['kc'][idx], label='KC Vm', lw=1.0)
        axes_vm[ii, 0].set_ylabel(str(amp))
        kcggn_ds = kcggn_grp[node]
        data = {}
        for kk, field in enumerate(kcggn_ds.attrs['fields']):            
            data[field.decode()] = kcggn_ds[kk, :]
        axes_vm[ii, 1].plot(data['t'][idx], data['kc'][idx], label='KC Vm', lw=1.0)
        # axes_vm[ii, 1].plot(data['t'], data['ca'], label='GGN Vm')
        txt = axes_vm[ii, 0].set_ylabel('{} pA'.format(int(1e3 * amp)))
        txt.set_rotation(0)
    for ax in axes_vm.flat:
        mbnetplot.despine(ax)
        ax.xaxis.set_visible(False)
        ax.set_yticks([])
    for ax in axes_vm[-1, :]:
        ax.xaxis.set_visible(True)
        
    # fig_vm.tight_layout()
    fig_vm.set_size_inches(5, 6)
    fname = '../figures/Figure_3b.svg'
    fig_vm.savefig(fname,
                   transparent=True)
    kc_amp_scount = sorted(kc_amp_scount, key=lambda x: x[0])
    for name, node in kcggn_grp.items():
        fields = [fname_.decode() for fname_ in node.attrs['fields']]
        tidx = fields.index('t')
        vidx = fields.index('kc')
        v = node[vidx, :]
        t = node[tidx, :]
        peaks, troughs = peakdetect(v, x_axis=t, lookahead=20, delta=0.1*np.std(v))
        kcggn_amp_scount.append((node.attrs['amp'], len(peaks)))
    kcggn_amp_scount = sorted(kcggn_amp_scount, key=lambda x: x[0])
    kc_amps, kc_scount = zip(*kc_amp_scount)
    kcggn_amps, kcggn_scount = zip(*kcggn_amp_scount)
    axes_spikecount.plot(np.array(kc_amps) * 1e3, kc_scount, 'o-', label='Only KC')
    axes_spikecount.plot(np.array(kcggn_amps) * 1e3, kcggn_scount, '^-', label='KC with GGN feedback')
    axes_spikecount.legend(bbox_to_anchor=(0., 1.0, 1.0, 0.102), loc='lower left')
    mbnetplot.despine(axes_spikecount)
    # axes_spikecount.set_xlabel('Current (pA)')
    # axes_spikecount.set_ylabel('Spike count')
    fig_spikecount.set_size_inches((4, 3))
    fig_spikecount.subplots_adjust(left=0.16, right=0.9, top=0.8, bottom=0.1, wspace=0.1, hspace=0.1)
    # fig_spikecount.tight_layout()
    fname = '../figures/Figure_3c.svg'
    fig_spikecount.savefig(fname, transparent=True)
    plt.show()

    
# 
# figure_3bc.py ends here
