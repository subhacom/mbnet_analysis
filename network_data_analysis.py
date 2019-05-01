# network_data_analysis.py --- 
# 
# Filename: network_data_analysis.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Mar  7 17:35:29 2018 (-0500)
# Version: 
# Package-Requires: ()
# Last-Updated: Tue Apr 30 19:35:17 2019 (-0400)
#           By: Subhasis Ray
#     Update #: 500
# URL: 
# Doc URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Code:
from __future__ import print_function
import numpy as np
import yaml
import pint
import pandas as pd
from timeit import default_timer as timer
import neurograph as ng


_ur = pint.UnitRegistry()
Q_ = _ur.Quantity

ggn_kc_syn_path = '/data/static/ggn_kc_synapse/ggn_kc_synapse'
pn_kc_syn_path =  '/data/static/pn_kc_synapse/pn_kc_synapse'
kc_st_path = '/data/event/kc/kc_spiketime'
pn_st_path = '/data/event/pn/pn_spiketime'
mca_cluster_path = '/data/static/mca_cluster_labels/mca_cluster_labels'
mca_kc_cluster_path = '/data/static/mca_kc_cluster_labels/mca_kc_cluster_labels'
lca_cluster_path = '/data/static/lca_cluster_labels/lca_cluster_labels'
lca_kc_cluster_path = '/data/static/lca_kc_cluster_labels/lca_kc_cluster_labels'


def ggn_sec_to_sid(name):
    sec_name = name.rpartition('.')[-1].partition('[')[0]
    if sec_name.startswith('soma'):
        return 1
    elif sec_name.startswith('dend_'):
        return int(sec_name.partition('_')[-1])
    elif sec_name == 'dend':
        return 3


def get_ggn_kc_syn_info(fd):
    """Get the GGN->KC syninfo with KC as the index. There is one synapse for each KC, so this makes sense.
    Also, the KCs 1-1 correspond to the clustering data."""    
    ggn_kc_syn =  fd['/data/static/ggn_kc_synapse/ggn_kc_synapse'].value.flatten()
    presec = np.char.decode(ggn_kc_syn['pre'].astype('S'))
    postsec = np.char.decode(ggn_kc_syn['post'].astype('S'))
    syndf = pd.DataFrame(data={'pre': presec, 'post': postsec, 'pos': ggn_kc_syn['prepos'][0]})
    syndf = syndf.set_index('post', drop=False, verify_integrity=True)
    return syndf


def get_kc_st(fd):
    """Get a list of KCs and their spike times"""
    return [(node.attrs['source'].decode('utf-8'), node.value)  \
            for node in fd[kc_st_path].values()]        


def get_event_times(group, nodes=None):
    """Get the spike times under group and the index of the spike train for raster plots.

    group: the HDF5 group under which the spike times are stored.

    nodes: list of str specifying the dataset names under `group`.

    Returns: (spike_x, spike_y) where spike_x is a list of arrays
    containing spike times and spike_y is a list of arrays of the same
    size containing the index of that spike train.

    """
    start = timer()
    spike_x = []
    spike_y = []
    if nodes is None:
        nodes = group
    for ii, node in enumerate(nodes):
        st = group[node][:]
        spike_x.append(st)
        sy = np.zeros(st.shape)
        sy[:] = ii
        spike_y.append(sy)
    end = timer()
    print('get_event_times: {}s'.format(end - start))
    return spike_x, spike_y


def get_kc_event_node_map(fd):
    """Return a dict of kc section<-> spike train data node name"""
    grp = fd[kc_st_path]
    ret = {}
    for node in grp.values():
        ret[node.attrs['source']] = node.name
    return ret

def get_kcs_by_region(fd, region):
    """Returns a set containing the section names of kcs in the region
    ('lca' or 'mca' or 'all').

    fd (h5py.File) is the open file handle.

    """
    ggn_kc_syn = get_ggn_kc_syn_info(fd)
    # The dendrites are named dend_{n}[iii] where n is the numerical
    # type in the SWC file of the GGN morphology
    if region == 'all':
        grp = fd[kc_st_path]
        matching_kcs = [grp[node].attrs['source'] for node in grp]
    else:
        dend_pattern = 'dend_{}'.format(ng.custom_types[region.upper()])
        matching_kcs = ggn_kc_syn.post[ggn_kc_syn.pre.str.find(dend_pattern) >= 0]
    return set(matching_kcs)


def get_kc_spike_nodes_by_region(fd, region):
    """Select dataset names for KC spiketimes where the KC belongs to the
    specified region ('lca' or 'mca' or 'all') of the calyx"""
    start = timer()
    try:
        kc_st = pd.DataFrame(data=fd['/map/event/kc/kc_spiketime'].value)
        if region == 'all':
            kc_spike_nodes = [fd[node].name for index, node in kc_st['data'].iterrows()]
        else:
            kcs = pd.DataFrame(
                data=fd['/data/static/{}_cluster_labels/{}_cluster_labels'.format(
                    region, region)].value['sec'])
            kc_st = pd.merge(kcs, kc_st, left_on='sec', right_on='source')
            kc_spike_nodes = [fd[node].name for index, node in kc_st['data'].iterrows()]            
    except KeyError:
        # Find KCs that are postsynaptic to a GGN dendrite corresponding to region
        matching_kcs = get_kcs_by_region(fd, region)
        ## Now retrieve the map between KC name and spike train dataset
        kc_grp = fd['/data/event/kc/kc_spiketime']
        kc_spike_nodes = [node for node in kc_grp
                          if kc_grp[node].attrs['source'].decode()
                          in matching_kcs]
    end = timer()
    print('get_kc_spike_nodes_by_region: {} s'.format(end - start))
    return kc_spike_nodes


def get_kc_vm_idx_by_region(fd, region):
    """Obtain the row indices in 2D Vm array for KCs in `region`. Returns
    a list containing (KC name, row index)"""
    kc_vm_node = fd['/data/uniform/kc/KC_Vm']
    kcs = get_kcs_by_region(fd, region)    
    kc_vm_name_index = {name.decode(): ii for ii, name in enumerate(kc_vm_node.dims[0]['source'])}
    match = kcs.intersection(kc_vm_name_index.keys())
    print(region, list(kcs)[0], list(kc_vm_name_index.keys())[0], len(match))
    return [(name, kc_vm_name_index[name]) for name in match]


def load_config(fd):
    """Load and return the config from open file fd as an YAML structure"""
    config = None
    try:
        config = yaml.load(fd.attrs['config'].decode())
    except KeyError:
        config = yaml.load(fd['/model/filecontents/mb/network/config.yaml'].value[0])
    return config


def get_simtime(fd):
    config = load_config(fd)
    simtime = Q_(config['stimulus']['onset']).to('ms').m +  \
        Q_(config['stimulus']['duration']).to('ms').m + \
        Q_(config['stimulus']['tail']).to('ms').m
    return simtime

def get_stimtime(fd):
    """Returns the timing parameters (in ms) for stimulus,

    onset: time when stimulus started,
    duration: interval for which stimulus was on,
    offdur: duration of OFF response"""
    config = load_config(fd)
    return {'onset': Q_(config['stimulus']['onset']).to('ms').m,
            'duration': Q_(config['stimulus']['duration']).to('ms').m,
            'offdur': Q_(config['pn']['offdur']).to('ms').m}
    

def get_spiking_kcs(fd):
    return [node.attrs['source'].decode('utf-8') for node in fd[kc_st_path].values() if len(node) > 0]
    

# 
# network_data_analysis.py ends here
