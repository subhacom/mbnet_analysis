# figure_2g.py --- 
# Author: Subhasis Ray
# Created: Mon Sep 10 14:11:17 2018 (-0400)
# Last-Updated: Tue Apr 30 19:21:16 2019 (-0400)
#           By: Subhasis Ray
# Version: $Id$

# Code:
"""Plot the results of parameter sweep of RM and RA of GGN with
staggered input.

Assumes all the files have same prefix and then RA{index}_RM{index}

"""
import numpy as np
from matplotlib import pyplot as plt
import h5py as h5
import os
from collections import defaultdict
import nrnutils as nu


plt.style.use('bmh')
fontsize = 8
plt.rcParams['figure.figsize'] = (6/2.54, 4/2.54)
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

fig = plt.figure(num=1, dpi=300)
ax = fig.add_subplot(111)
ddir = '../simulated/Fig_2g'
RM_RA_median_map = defaultdict(dict)
RM_RA_vdef_map = defaultdict(dict)
for fname in os.listdir(ddir):
    if not fname.endswith('.h5'):
        continue
    with h5.File(os.path.join(ddir, fname), 'r') as fd:
        sections = fd['section'].value
        vpeak = fd['v_deflection'].value
        em = fd.attrs['e_pas']
        vdef = vpeak - em
        region_vpeak_map = defaultdict(list)
        ref_def = fd['vclamp'].attrs['clamp_voltage'] - fd.attrs['e_pas']
        for sec, dv in zip(sections, vdef):
            region_vpeak_map[nu.sectype(sec)].append(dv)
        RM = 1e-3/fd.attrs['g_pas']
        RA = fd.attrs['RA']
        out_vdef = np.concatenate([region_vpeak_map[5],
                                   region_vpeak_map[6]]) # /ref_def
        RM_RA_median_map[RM][RA] = np.median(out_vdef)
        RM_RA_vdef_map[RM][RA] = out_vdef
RM_lines = []
for RM in RM_RA_vdef_map:    
    RA = sorted(RM_RA_vdef_map[RM].keys())
    medians = [RM_RA_median_map[RM][r] for r in RA]
    out_vdefs = [RM_RA_vdef_map[RM][r] for r in RA]
    vp = ax.violinplot(out_vdefs, RA, showmedians=False, showextrema=False, widths=3.0)
    RM_lines.append((RM, medians, vp['bodies'][0].get_facecolor()[0]))
    # ax.plot(RA, medians, ls='-', marker='.', label=r'{}'.format(int(RM)))
RM_lines = sorted(RM_lines, key=lambda x: x[1][-1])
ypos = np.linspace(RM_lines[0][1][-1], ax.get_ylim()[1], len(RM_lines))
for y, (RM, medians, color) in zip(ypos, RM_lines):
    lines = ax.plot(RA, medians, color=color, ls='-', marker=',')
    ax.text(RA[-1] + 5, y, '{}'.format(int(RM)), color=color, alpha=1.0, verticalalignment='top')
ax.legend(loc=(1, 0))    
ax.tick_params(axis='y', which='right', length=0)
ax.tick_params(axis='x', which='top', length=0)
ax.xaxis.set_ticks([50, 100, 150])
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.grid(False)
# ax.text(ax.get_xlim()[1], ax.get_ylim()[0], r'Membrane resistivity (k$Omega$ cm)', rotation=90, horizontalalignment='left', verticalalignment='bottom')
# ax.set_xlabel(r'Axial resistivity ($\Omega$ cm)')
# ax.set_yticks([10, 15, 20, 25])
[s.set_visible(False) for s in ax.spines.values()]
fig.frameon = False
fig.subplots_adjust(left=0.1, right=0.8, top=0.99, bottom=0.1, wspace=0.1, hspace=0.1)
fname =  '../figures/Figure_2g.svg'
fig.savefig(fname, dpi=300, transparent=True)
print('Saved figure in {}'.format(fname))
plt.show()    
    


# 
# figure_2g.py ends here
