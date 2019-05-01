# figure_5efgh.py --- 
# Author: Subhasis  Ray
# Created: Fri Dec 21 12:32:08 2018 (-0500)
# Last-Updated: Tue Apr 30 20:03:59 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:

"""This script plots shifting PN and GGN response.

"""
#* imports
from __future__ import print_function

import h5py as h5
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import mbnetplot
import network_data_analysis as nda

plt.rc('font', size=8)

#* setup figure and axes
gs = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[2, 1, 1], hspace=0.05)
fig = plt.figure()
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0, sharey=ax1)

#* open file descriptors
#** jid_shifting_lognorm for shifting PNs with lognorm distribution of synapses
jid_lognorm = '22087969'
fname_lognorm = '../simulated/Fig_5egh/mb_net_UTC2019_03_09__18_28_19-PID22056-JID22087969.h5'
fd_lognorm = h5.File(fname_lognorm, 'r')
#** jid_constant for only shifting PN, constant GGN->KC inhibition
jid_constant = '24211204'
fname_constant = '../simulated/Fig_5f/mb_net_UTC2019_04_11__19_27_17-PID25901-JID24211204.h5'
fd_constant = h5.File(fname_constant, 'r')

#* plot PN activity from jid_lognorm
stiminfo = nda.get_stimtime(fd_lognorm)
stiminfo_const = nda.get_stimtime(fd_constant)
pn_st = []
pn_id = []
for pn in fd_lognorm[nda.pn_st_path].values():
    pn_st.append(pn.value)
    pn_id.append([int(pn.name.rpartition('_')[-1])] * len(pn))    
ax0.plot(np.concatenate(pn_st[::10]), np.concatenate(pn_id[::10]),
         color='#fdb863', ls='', marker='s', ms=1)
ax0.xaxis.set_visible(False)
ax0.yaxis.set_visible(False)

#* plot ggn vm from both simulations
mbnetplot.plot_ggn_vm(ax1, fd_constant, fd_constant['/data/uniform/ggn_basal/GGN_basal_Vm'],
                   'dend_b', 1, color='#009292', alpha=1.0)
ax1.hlines(y=-53,
           xmin=stiminfo_const['onset'],
           xmax=stiminfo_const['onset'] + stiminfo_const['duration'],
           color='gray', lw=10)
ax1.hlines(y=-53,
           xmin=stiminfo_const['onset'] + stiminfo_const['duration'],
           xmax=stiminfo_const['onset'] + stiminfo_const['duration'] + stiminfo_const['offdur'],
           color='lightgray', lw=10)

mbnetplot.plot_ggn_vm(ax2, fd_lognorm, fd_lognorm['/data/uniform/ggn_basal/GGN_basal_Vm'],
                   'dend_b', 1, color='#009292', alpha=1.0)
ax2.set_ylim((-53, -40))
ax2.set_yticks([-50, -45])
ax2.set_xlim(200, 2500)
# ax2.set_xticks([200, 1000, 2000])
# ax2.set_xlabel('Time (ms)')
ax2.hlines(y=-53,
           xmin=stiminfo['onset'],
           xmax=stiminfo['onset'] + stiminfo['duration'],
           color='gray', lw=10)
ax2.hlines(y=-53,
           xmin=stiminfo['onset'] + stiminfo['duration'],
           xmax=stiminfo['onset'] + stiminfo['duration'] + stiminfo['offdur'],
           color='lightgray', lw=10)
# ax2.set_ylabel('Voltage (mV)')
ax0.set_ylabel('Cell no.')
fig.set_size_inches(5.5/2.54, 10/2.54)
for ax in [ax0, ax1, ax2]:
    ax.tick_params(top=False, right=False)
    ax.xaxis.set_visible(False)
    [sp.set_visible(False) for sp in ax.spines.values()]
fig.set_frameon(False)
# fig.subplots_adjust(left=0.18, right=0.97, top=0.95, bottom=0.1, hspace=0.05)
fig.tight_layout()
fig.savefig('../figures/Figure_5efg.svg', transparent=True, dpi=300)

#* show KC spike rate distribution for the shifting PN +weak KC inhibition simulation
kc_spike_counts = [len(v) for v in fd_lognorm[nda.kc_st_path].values()]
stimend = stiminfo['onset'] + stiminfo['duration'] + stiminfo['offdur']
rates = [spike_count * 1e3 / (stimend - stiminfo['onset']) for
         spike_count in kc_spike_counts]
max_rate = max(rates)
print('Max rate:', max_rate)
bins = np.arange(0, np.ceil(max_rate))
#** plot the histograms
# try a broken y axis
fig_kc_rates, ax_kc_rates = plt.subplots(nrows = 2,  sharex='all')

for kc_ax in ax_kc_rates.flat:
    hist, bins, patchs = kc_ax.hist(rates, bins=bins, color='#3d26a8')
    [sp.set_visible(False) for sp in kc_ax.spines.values()]
ax_kc_rates[0].set_ylim((35000,  50000))
ax_kc_rates[0].tick_params(top = False, bottom = False,  right = False)
ax_kc_rates[0].set_yticks([40000,  50000])
ax_kc_rates[1].set_xlabel('Firing rate (spikes/s)')
ax_kc_rates[1].set_ylabel('Number of cells')
ax_kc_rates[1].tick_params(top=False, right=False)
ax_kc_rates[1].set_ylim(0, 1000)
# ax_kc_rates.set_yticks([0, 10000, 20000, 30000, 40000])
ax_kc_rates[1].set_yticks([0, 500, 1000])
ax_kc_rates[1].annotate('Maximum', xy=(max_rate, 0),
                     xytext=(max_rate, 500),
                     arrowprops=dict(facecolor='black', width=0.5,
                                     headwidth=5.0, shrink=0.05),
                     horizontalalignment='right')
fig_kc_rates.set_frameon(False)
fig_kc_rates.set_size_inches(5.5/2.54, 5/2.54)
fig_kc_rates.subplots_adjust(left=0.25, right=0.95, top=0.98, bottom=0.2,  hspace = 0.0)
fig_kc_rates.savefig('../figures/Figure_5h.svg', transparent=True, dpi=300)

plt.show()

#* close the files
fd_lognorm.close()
fd_constant.close()

# 
# figure_5efgh.py ends here
