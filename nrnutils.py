# nrnutils.py --- 
# 
# Filename: nrnutils.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Tue Jul 26 16:27:51 2016 (-0400)
# Version: 
# Package-Requires: ()
# Last-Updated: Tue Apr 30 16:53:59 2019 (-0400)
#           By: Subhasis Ray
#     Update #: 535
# URL: 
# Doc URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Code:

from __future__ import print_function

import sys
import numpy as np
import networkx as nx
import neurograph as ng

from neuron import h

print('#C')
sys.stdout.flush()

# plt.rcParams['axes.facecolor'] = 'black'
# plt.style.use('dark_background')
h.load_file("stdrun.hoc")

sys.stdout.flush()


def sectype(cid):
    """Get the SWC structure type (sid) from section name `cid`"""
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


def nrngraph(cell):
    """Convert a NEURON cell model into networkx graph. The nodes are the
    section names in the cell.

    Each node has the parent section name in attribute `p`. It is -1
    if there is no parent.  x, y, z represent the coordinates of the 0
    end.  orig is SectionRef to the original NEURON section.

    For display purposes, we add the 1 end of the leaf sections as
    dummy nodes. They can be identified by the attribute orig=None.

    """
    g = nx.DiGraph()
    cell.soma.push()   # Assume there is a designated soma
    # This appends all sections accessible from currently
    # accessed section in root to leaves
    ordered_tree = h.SectionList()
    ordered_tree.wholetree()
    h.pop_section()
    for comp in ordered_tree:
        comp.push()
        cid = comp.name()
        stype = sectype(cid)
        g.add_node(cid, x=h.x3d(0),
                   y=h.y3d(0), z=h.z3d(0),
                   r=h.diam3d(0)/2.0, s=stype, orig=comp)
        ref = h.SectionRef(sec=comp)
        if ref.has_parent():
            parent = ref.parent
            g.add_edge(parent.name(), cid, length=parent.L)
            g.node[cid]['p'] = parent.name()
        else:
            g.node[cid]['p'] = -1
        # Insert the 1-end of a section as a dummy node
        if ref.nchild() == 0:
            leaf_node = '{}_1'.format(cid)
            g.add_node(leaf_node, x=h.x3d(1),
                       y=h.y3d(1), z=h.z3d(1),
                       r=h.diam3d(1)/2.0, s=stype, p=comp.name(), orig=None)
            g.add_edge(cid, leaf_node, length=comp.L)
        h.pop_section()
    return g


def get_section_node_map(g):
    ret = {}
    for n, d in g.nodes_iter(data=True):
        if d['orig'] is None:
            continue
        ret[d['orig'].name()] = n
    return ret


def get_alen_pos3d(sec):
    """Get the arclength and 3D position of the poinst in section sec.
    Inspired by
    http://www.neuron.yale.edu/ftp/ted/neuron/extracellular_stim_and_rec.zip:
    interpxyz.hoc

    Returns: length, pos where length is the list of arc3d lengths and
    pos the list of 3D positions (x, y, z), of the 3D points in
    section `sec`.

    """
    npts = int(h.n3d(sec=sec))
    pos = []
    length = []
    for ii in range(npts):
        pos.append((h.x3d(ii, sec=sec),
                    h.y3d(ii, sec=sec),
                    h.z3d(ii, sec=sec)))
        length.append(h.arc3d(ii, sec=sec))
    return length, pos


def get_seg_3dpos(sec, xlist):
    """Obtain the nearest 3D point position on or before the segments
    specified by 1D position in xlist."""
    length, pos = get_alen_pos3d(sec)
    length = np.array(length) / length[-1]  # normalize the length
    ret = []
    for xx in xlist:
        ii = np.searchsorted(length, xx)
        ret.append(pos[ii])
    return ret

                   
def select_good_nodes_by_sid(g, sid_list, counts, replace=False):
    """For each sid in `sid_list` select `count` random nodes with an
    underlying `section` from `g` - a neurongraph.

    Returns a list of selected nodes.

    @seealso: neurograp.select_random_nodes_by_sid

    """
    good_nodes = [n for n, data in g.nodes(data=True) if data['orig'] is not None]
    type_node_map = ng.get_stype_node_map(g.subgraph(good_nodes))
    synnodes = []
    for branch_id, count in zip(sid_list, counts):
        size = len(type_node_map[branch_id])
        if (count > size) and (replace == False):
            print('Changing number of nodes to maximum {} available in branch, since replace={}'.format(size, replace))
            count = size
        synnodes += list(np.random.choice(type_node_map[branch_id],
                                          size=count, replace=replace))
    return synnodes


def select_random_segments_by_sid(g, sid, count, by_length=True, replace=True):
    """Select segments from sections with specified sid.  If by_length is
    True, select with probability proportional to the length of the
    segment.

    """
    good_secs = [data['orig'] for n, data in g.nodes(data=True)
                 if (g.node[n]['orig'] is not None)
                 and (g.node[n]['s'] == sid)]
    seg_list = []
    seg_lengths = []
    for sec in good_secs:
        for seg in sec:
            seg_list.append(seg)
            seg_lengths.append(sec.L/sec.nseg)
    seg_lengths = np.array(seg_lengths)
    probability = None
    if by_length:
        probability = seg_lengths / np.sum(seg_lengths)
    segs = np.random.choice(seg_list, size=count, p=probability, replace=replace)
    # Don't keep the segments around beyond this function, x and sec
    # will suffice to retrieve them
    return [( seg.sec, seg.x) for seg in segs]


def select_random_terminal_segments_by_sid(g, sid, count, by_length=True, replace=True):
    """Select segments from sections with specified sid.  If by_length is
    True, select with probability proportional to the length of the
    segment.

    """
    good_secs = []
    for n in g.nodes():
        sec = g.node[n]['orig']        
        if sec is not None and (g.node[n]['s'] == sid):
            ref = h.SectionRef(sec=sec)
            if len(ref.child) == 0:
                good_secs.append(sec)
    seg_list = []
    seg_lengths = []
    for sec in good_secs:
        for seg in sec:
            seg_list.append(seg)
            seg_lengths.append(sec.L/sec.nseg)
    seg_lengths = np.array(seg_lengths)
    probability = None
    if by_length:
        probability = seg_lengths / np.sum(seg_lengths)
    segs = np.random.choice(seg_list, size=count, p=probability, replace=replace)
    # Don't keep the segments around beyond this function, x and sec
    # will suffice to retrieve them
    return [( seg.sec, seg.x) for seg in segs]  

# 
# nrnutils.py ends here
