# mbnetplot.py --- 
# 
# Filename: pn_kc_ggn_plot.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Feb 16 13:08:41 2018 (-0500)
# Last-Updated: Tue Apr 30 19:41:57 2019 (-0400)
#           By: Subhasis Ray
#     Update #: 1120
# 
# Code:
from __future__ import print_function
import os
from timeit import default_timer as timer
import numpy as np
import random
import h5py as h5
from matplotlib import pyplot as plt
from pint import UnitRegistry
import neurograph as ng
import network_data_analysis as nda

ur = UnitRegistry()
Q_ = ur.Quantity

def despine(ax, locs='all'):
    """Remove the line marking axing bounder in locations specified in the list `locs`.
    The locations can be 'top', 'bottom', 'left', 'right' or 'all'
    """
    if locs == 'all':
        locs = ['top', 'bottom', 'left', 'right']
    for loc in locs:
        ax.spines[loc].set_visible(False)

        
def plot_population_psth(ax, spike_trains, ncell, bins, alpha=0.5, rate_sym='b-', cell_sym='r--'):
    """Plot the population PSTH on axis `ax`, where `spike_trains` is a
    list of arrays, each containing the spike times of one cell. ncell
    is the total number of cell and is used for computing rates per
    cell.

    """
    start = timer()
    cell_counts = np.zeros(len(bins) - 1)
    spike_counts = np.zeros(len(bins) - 1)
    spiking_cell_count = 0
    for train in spike_trains:
        if len(train) > 0:
            spiking_cell_count += 1
        hist, bins = np.histogram(train, bins)
        cell_counts += (hist > 0)
        spike_counts += hist
    print('Total number of spiking cells', spiking_cell_count, 'out of', len(spike_trains))
    spike_rate = spike_counts * 1e3 / (ncell * (bins[1] - bins[0]))
    cell_frac = cell_counts * 1e3 / (ncell * (bins[1] - bins[0]))
    if rate_sym is None:
        ax.plot((bins[:-1] + bins[1:]) / 2.0, spike_rate, 
                label='spikes/ncells/binwidth(s)', alpha=alpha)
    else:
        ax.plot((bins[:-1] + bins[1:]) / 2.0, spike_rate, 
                rate_sym, label='spikes/ncells/binwidth(s)', alpha=alpha)
    if cell_sym is None:
        ax.plot((bins[:-1] + bins[1:]) / 2.0, cell_frac, 
            label='cells spiking/ncells/binwidth(s)', alpha=alpha)
    else:        
        ax.plot((bins[:-1] + bins[1:]) / 2.0, cell_frac, cell_sym,
            label='cells spiking/ncells/binwidth(s)', alpha=alpha)
    end = timer()
    print('Plotted PSTH for {} spike trains in {} s'.format(
        len(spike_trains),
        end - start))
    return ax, spike_rate, cell_frac



def plot_kc_spikes(ax, fd, ca_side='both', color='k', marker=','):
    """Raster plot KC spike times for KCs belonging to the specified side
    of calyx ('lca' or 'mca').

    This function does not care about spatial clusters.

    Returns: the line object, list of spike times and list of their
    y-positions.

    """
    if ca_side == 'both':
        nodes = fd[nda.kc_st_path].keys()
    else:
        nodes = nda.get_kc_spike_nodes_by_region(fd, ca_side)
    spike_x, spike_y = [], []
    fname = fd.filename
    try:
        spike_x, spike_y = nda.get_event_times(
            fd['/data/event/kc/kc_spiketime'],
            nodes=nodes)
    except KeyError:
        dirname = os.path.dirname(fname)
        fname = 'kc_spikes_' + os.path.basename(fd.filename)
        with h5.File(os.path.join(dirname, fname)) as kc_file:
            spike_x, spike_y = nda.get_event_times(kc_file, nodes=nodes)
    if len(spike_x) > 0:
        ret = ax.plot(np.concatenate(spike_x), np.concatenate(spike_y),
                      color=color, marker=marker, linestyle='none')
    else:
        ret = None
    return ret, spike_x, spike_y


def plot_kc_vm(ax, fd, region, count, color='k', alpha=0.5):
    """Plot Vm of `count` KCs from sepcified region."""
    match = nda.get_kc_vm_idx_by_region(fd, region)
    if len(match) == 0:
        return [], []
    selected = random.sample(match, min(count, len(match)))
    kc_vm_node = fd['/data/uniform/kc/KC_Vm']
    try:
        t = np.arange(kc_vm_node.shape[1]) * kc_vm_node.attrs['dt']
    except KeyError:
        t = np.arange(kc_vm_node.shape[1])
    for name, idx in selected:
        ax.plot(t, kc_vm_node[idx, :], label=name, color=color, alpha=alpha)
    return selected


def plot_ggn_vm(ax, fd, dataset, region=None, count=5, color='k', alpha=0.5):
    """Plot Vm of GGN from dataset"""
    sec_list = [sec.decode('utf-8')  for sec in dataset.dims[0]['source']]
    match = []
    if region is None:
        match = [(sec, ii) for ii, sec in enumerate(sec_list)]
    else:
        rsid = ng.name_sid[region]
        for ii, sec in enumerate(sec_list):
            sid = nda.ggn_sec_to_sid(sec)
            if sid == rsid:
                match.append((sec, ii))
    # print(match)
    if len(match) == 0:
        return [], []
    selected = random.sample(match, min(len(match), count))
    try:
        t = np.arange(dataset.shape[1]) * dataset.attrs['dt']
    except KeyError:
        t = np.arange(dataset.shape[1])
    for sec, ii in selected:
        ax.plot(t, dataset[ii, :], label=sec, color=color, alpha=alpha)
    return selected
    



def plot_kc_spike_count_hist(fname, bins=None, save=False, figdir='figures'):
    """Plot histogram of spike counts recorded as 1D datasets under group
    path in file fname"""
    with h5.File(fname, 'r') as fd:
        kc_st_grp = fd['/data/event/kc/kc_spiketime']
        lca_kcs = nda.get_kc_spike_nodes_by_region(fd, 'LCA')
        lca_spike_count = [len(kc_st_grp[kc]) for kc in lca_kcs] 
        mca_kcs = nda.get_kc_spike_nodes_by_region(fd, 'mca')
        mca_spike_count = [len(kc_st_grp[kc]) for kc in mca_kcs]        
        fig, axes = plt.subplots(nrows=2, ncols=1, sharey='all', sharex='all')
        if bins is None:
            bins = np.arange(max(lca_spike_count + mca_spike_count) + 1)
        hist, bins, patches = axes[0].hist(lca_spike_count, bins=bins)
        axes[0].arrow(max(lca_spike_count), max(hist)/2, 0, -max(hist)/2.0,
                      head_width=0.5, head_length=max(hist)/20.0,
                      length_includes_head=True)
        axes[0].set_title('LCA')
        hist, bins, patches = axes[1].hist(mca_spike_count, bins=bins)
        axes[1].arrow(max(mca_spike_count), max(hist)/2, 0, -max(hist)/2,
                      head_width=0.5, head_length=max(hist)/20.0,
                      length_includes_head=True)
        axes[1].set_title('MCA')
        fname = os.path.basename(fname)
        fig.suptitle(os.path.basename(fname))
        fname = fname.rpartition('.h5')[0]
        if save:
            fname = os.path.join(figdir, fname) + '_kc_spikecount_hist.png'
            fig.tight_layout()
            fig.savefig(fname)
            print('Saved spike count histogram of KCs in', fname)
    return fig, axes


        
# 
# mbnetplot.py ends here
