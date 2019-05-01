# figure_2e.py --- 
# Author: Subhasis Ray
# Created: Fri Sep  7 16:57:34 2018 (-0400)
# Last-Updated: Tue Apr 30 19:14:35 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:
from __future__ import print_function
import os
import random
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from matplotlib import gridspec
from collections import defaultdict
import neurograph as ng
import nrnutils as nu
from neuron import h

fontsize = 8
plt.rcParams['figure.figsize'] = (6/2.54, 5.5/2.54)
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

def get_ggn(fname):
    with h5.File(fname, 'r') as fd:
        ggn_model = fd['/celltemplate'].value.decode('utf-8')
        cellname = [line for line in ggn_model.split('\n')
                    if 'begintemplate' in line]
        rest = cellname[0].partition('begintemplate')[-1]
        cellname = rest.split()[0]
        if not hasattr(h, cellname):
            h(ggn_model)
        return eval('h.{}()'.format(cellname))

outfilename = '../simulated/Fig_2ef/Vm_inhpoisson_stim_series_20180907_171406.h5'
colormap = '15cb'
plots_per_region = 1

if __name__ == '__main__':
    ggn = get_ggn(outfilename)
    with h5.File(outfilename, 'r') as fd:
        graph = nu.nrngraph(ggn)
        leaves = [graph.node[n]['p'] for n in graph.nodes if graph.nodes[n]['orig'] is None]
        sid_leaf_map = defaultdict(list)
        for leaf in leaves:
            sid_leaf_map[graph.node[leaf]['s']].append(leaf)
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        ax_vm = fig.add_subplot(gs[0])   # Plot the Vm
        syninfo = [s for s in fd['syninfo']]
        sid_data_dict = defaultdict(list)
        for nodename in fd:
            if nodename.startswith('v_'):
                sid = fd[nodename].attrs['sid']
                # Take only leaf nodes
                if fd[nodename].attrs['section'] in sid_leaf_map[sid]:
                    sid_data_dict[sid].append(nodename)
        # Here I plot a random sampling of 5 or fewer datasets from each structure id
        cmap = ng.colormaps[colormap]
        syn_plotted = []
        legend_lines = []
        legend_labels = []
        sid_label_map = {1: 'soma', 3: 'basal', 5: 'LCA', 6: 'MCA', 7: 'LH', 8: r'$\alpha$ lobe'}
        for sid, nodelist in sid_data_dict.items():
            print('sid', sid)
            if sid == 1:  # skip soma
                continue
            nodes = random.sample(nodelist, min(plots_per_region, len(nodelist)))
            color = np.array(cmap[sid % len(cmap)]) / 255.0
            line = None
            for nodename in nodes:
                sec = fd[nodename].attrs['section']
                ls = '-'
                line = ax_vm.plot(fd['time'], fd[nodename], color=color, ls=ls, label=sec)[0]
            if line is not None:
                legend_lines.append(line)
                legend_labels.append(sid_label_map[sid])
        # Plot the input spike raster, put the ones delivered to the plotted sections on top and in red
        ax_stim = fig.add_subplot(gs[1], sharex=ax_vm)
        syn_color = 'yellow'
        other_color = 'blue'
        syn_stim = []
        other_stim = []
        for ii, stimnode in enumerate(fd['stimulus']):
            ds = fd['stimulus'][stimnode]
            ax_stim.plot(ds, np.ones_like(ds)*ii, color=other_color, ls='', marker=',', alpha=0.3)
        ax_stim.set_yticks((0, len(fd['stimulus'])))
        ax_vm.set_yticks([-50, -45, -40])
        ax_vm.xaxis.set_visible(False)
        for ax in [ax_vm, ax_stim]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.7, bottom=0.12, wspace=0.0, hspace=0.1)
        ax_vm.legend(legend_lines, legend_labels, ncol=2, loc=3,
                     bbox_to_anchor=(0.0, 1.02, 1.0, 0.30), borderaxespad=0, mode='expand')
        ax_stim.set_xlim(0, fd['time'].value[-1])
        figfile = '../figures/Figure_2e.svg'.format(os.path.basename(outfilename))
        fig.savefig(figfile, dpi=300, papertype='a4', frameon=False, transparent=True)
        print('Figure saved in', figfile)
        plt.show()



# 
# figure_2e.py ends here
