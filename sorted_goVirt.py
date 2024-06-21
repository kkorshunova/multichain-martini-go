#!/usr/bin/env python

import argparse
import numpy as np
import math
import re
import pandas as pd  # needed for get_bb_pair_sigma_epsilon()

# global variables:
seqDist = 4  # minimal distance in the sequence to add a elastic bond (ElNedyn=3 [Perriole2009]; Go=4 [Poma2017])
# (this has to result in the correct atom number when added to "k_at" compared to the .itp file)
c6c12 = 0  # if set to 1, the C6 and C12 term are expected in the .itp file; if set to 0, sigma and go_eps are used
mod_enabled = True
debug_mode = True

# names of the output included itp files:
if mod_enabled:
    fnames = ['atomtypes_go_mono.itp',
              'nonbond_params_go_mono.itp',
              'atoms_go_mono.itp',
              'virtual_sitesn_go_mono.itp',
              'exclusions_go_mono.itp',
              'viz_go_mono.itp',
              'go_mono.itp',
              'go_mono.top']
else:
    fnames = ['atomtypes_go.itp',
              'nonbond_params_go.itp',
              'atoms_go.itp',
              'virtual_sitesn_go.itp',
              'exclusions_go.itp',
              'viz_go.itp',
              'go.itp',
              'go.top']


##################### FUNCTIONS #####################
def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='File containing the coarse-grained structure of the protein in pdb format.')
    parser.add_argument('-i', help='File containing the topology of coarse-grained protein in itp format.')
    parser.add_argument('-f', help='File containing the contact analysis of the (atomistic) protein structure.')
    parser.add_argument('--nb', help='File containing martini_go.ff in itp format.')
    parser.add_argument('--moltype', default='mol',
                        help='String used as prefix in atomtypes of virtual sites, as well as in output file names '
                             '(default: mol)')
    parser.add_argument('--go_eps_inter', type=float, default=9.414,
                        help='Dissociation energy [kJ/mol] of the Lennard-Jones potential used in the Go-like model '
                             '(default: 9.414).')
    parser.add_argument('--go_eps_intra', type=float, default=9.414,
                        help='Dissociation energy [kJ/mol] of the Lennard-Jones potential used in the Go-like model '
                             '(default: 9.414).')
    parser.add_argument('--cutoff_short', type=float, default=0.3,
                        help='Lower cutoff distance [nm]: contacts with a shorter distance than cutoff_short are not '
                             'included in the Go-like interactions (default: 0.3).')
    parser.add_argument('--cutoff_long', type=float, default=1.1,
                        help='Upper cutoff distance [nm]: contacts with a longer distance than cutoff_long are not '
                             'included in the Go-like interactions (default: 1.1).')
    parser.add_argument('--missres', type=int, default=0,
                        help='Number of missing residues at the beginning of the atomistic pdb structure which is '
                             'needed if the numbering of the coarse-grained structure starts at 1 (default: 0).')
    parser.add_argument('--chain_sort', type=int, default=0,
                        help='Chain sorting method: 0 = distance-based (default), 1 = pdb chain-ID based, '
                             '2 = user input based (provide txt file using --chain_file flag)')
    parser.add_argument('--bb_cutoff', type=int, default=10,
                        help='Max distance (in A) allowed between next-neighbor BBs (default: 10 A).')
    parser.add_argument('--chain_file', help='File containing chain IDs (one per line; same order as input CG PDB)')
    parser.add_argument('--chain_id', default='A',
                        help='Select the output chain for which the contacts will be analyzed. (ChainID either from the'
                             'input pdb file (--chain_sort 1) or the user provided text file (--chain_sort 2)'
                             '(default: A)')
    args = parser.parse_args()
    return args


def read_data(cg_pdb, file_contacts):
    """
    1. reads the input CG pdb file (mind the fixed pdb file format!)
    2. parses data from the .map file (section "Residue residue contacts");

    Returns:
        pdb_data (list):  atomnr, atomname, resname, resnr, x,y,z, 0
        pdb_chain_ids (list): chain id
        indBB (array): BBs' atomnr, x,y,z
        map_OVrCSU (list): resid, resid, CA-CA distance in Å
    """
    pdb_data = [ ]
    indBB = [ ]  # separate from pdb_data because indBB needs to be a numpy array
    pdb_chain_ids = [ ]  # character-based chain IDs from the pdb
    with open(cg_pdb, 'r') as file:
        # create a 2d array with all relevant data: pdb_data columns 1-3,5-8
        #    here: omitted chain_id (column 4)
        for line in file:
            # later: pdb records HELIX, SHEET, SSBOND have different formats - not found in martinize2 output?
            if line[0:4] == 'ATOM' and line[12:16].strip() != 'CA':
                # later: more flexible ways to filter out unnecessary VSs written by martinize?
                # save only relevant columns in pdb_data + add column for chain id:
                pdb_data.append(
                    [int(line[6:11].strip()),  # atomnr
                     line[12:16].strip(),  # atomname
                     line[17:20],  # resname
                     int(line[22:26].strip()),  # resnr
                     float(line[30:38].strip()),  # x
                     float(line[38:46].strip()),  # y
                     float(line[46:54].strip()),  # z
                     0])  # chain_id placeholder
                #  e.g.: [1, 'BB', 'GLY', 1, -26.214, 5.188, -11.96, 0]
                pdb_chain_ids.append(line[21])  # chain IDs for all atoms
                if line[12:16].strip() == 'BB':
                    indBB.append([int(line[6:11].strip()),
                                  float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
                    # indBB:           atomnr           x              y              z
            else:
                continue  # skips irrelevant lines (e.g. CONECT if it's present in file)
    indBB = np.array(indBB)

    # read the map file
    map_OVrCSU = [] # instead of shell calls (og ver)
    with open(file_contacts, 'r') as f:
        for line in f:
            if re.search(r'1 [01] [01] [01]', line):
                line = line.split()
                if len(line) < 19:  # check for header lines (instead of 'header_lines = 0')
                    continue
                else:
                    map_OVrCSU.append(
                        [float(line[5]), float(line[9]), float(line[10])])  # instead of 'cols = [5, 9, 10]'
    # read the file again so that OV and rCSU blocks are separated, not mixed in map_OVrCSU[ ]
    with open(file_contacts, 'r') as f:
        for line in f:
            if re.search(r'0 [01] [01] 1', line):
                line = line.split()
                if len(line) < 19:  # check for header lines (instead of 'header_lines = 0')
                    continue
                else:
                    map_OVrCSU.append(
                        [float(line[5]), float(line[9]), float(line[10])])  # instead of 'cols = [5, 9, 10]'

    return indBB, map_OVrCSU, pdb_data, pdb_chain_ids


def get_go(indBB, map_OVrCSU, cutoff_short, cutoff_long, go_eps_intra, seqDist, missRes):
    """
    1. replaces AT CA-CA distances in map_OVrCSU with CG BB-BB distances using indBB array
    2. filters pairs in map_OVrCSU based on min/max cutoff distances, creates sym_pars list

    Returns:
        sym_pairs (list): atomnr_i BB, atomnr_j BB, Vii, Wii, resnr_i BB, resnr_j BB, dist, sigma (from map_OVrCSU)
        (usage: atomnr: exclusions, resnr, dist, sigma: go_table)
    """
    # calculate the distances based on the coordinates of the CG BB bead
    for k in range(0, len(map_OVrCSU)):
        dist_vec = indBB[ int(map_OVrCSU[k][1])-missRes-1 ,1:4] - indBB[ int(map_OVrCSU[k][0])-missRes-1 ,1:4]
        map_OVrCSU[k][2] = np.linalg.norm(dist_vec) /10     # [Ang] to [nm]

    # from the original goVirt script: create pairs list:
    pairs = []
    for k in range(0, len(map_OVrCSU)):
        if (map_OVrCSU[k][2] > cutoff_short) and (map_OVrCSU[k][2] < cutoff_long) \
                and ( abs(map_OVrCSU[k][1]-map_OVrCSU[k][0]) >= seqDist ):
            # parameters for LJ potential
            sigma = map_OVrCSU[k][2] / 1.12246204830        # calc sigma for the LJ potential in [nm]
            Vii = 4.0 * pow(sigma,6) * go_eps_intra
            Wii = 4.0 * pow(sigma,12) * go_eps_intra
            pairs.append([ int(indBB[ int(map_OVrCSU[k][0])-missRes-1 ,0]),  # atomnr BB_i
                           int(indBB[ int(map_OVrCSU[k][1])-missRes-1 ,0]),  # atomnr BB_j
                           Vii, Wii,                                         # todo: sigma, eps from map data (can be replaced with custom unsorted data)
                           int(map_OVrCSU[k][0]),                            # resnr BB_i
                           int(map_OVrCSU[k][1]),                            # resnr BB_j
                           map_OVrCSU[k][2],                                 # distance
                           sigma ])                                          # sigma
            Vii = []
            Wii = []

    sym_pairs = []
    # count contacts only once; exclude asymmetric rCSU contacts (cf. doi 10.1063/1.4929599)
    for k in range(0, len(pairs)):
        if pairs[k][0] < pairs[k][1]:
            for l in range(k+1, len(pairs)):
                if (pairs[l][0] == pairs[k][1]) and (pairs[l][1] == pairs[k][0]):
                    sym_pairs.append(pairs[k])
    if debug_mode:
        print("Total number of Go-bonds extracted from the map file: " + str(len(sym_pairs)))

    return sym_pairs


########## INTRA-INTER SORTING PROCEDURES ##########
def assign_chain_ids(pdb_data, bb_cutoff, pdb_chain_ids, chain_sort_method, chain_file, selected_chain):
    """
    1. updates pdb_data (last column) with chain IDs (3 options: distance-based, from input pdb, from user input file)
    2. defines length (in atomnr and resnr) of a single chain of the homopolymer and stores it in single_chain_mods

    Returns:
        pdb_data (list): created in read_data(), updated with chain-IDs here
        single_chain_mods (list): atomnr, resnr
    """
    # [DEPRECIATED] Distance-based method: only works for whole chains (no fragments), with all atoms in order
    if chain_sort_method == 0:
        new_chain_begins = []  # list of atom indices which start a new chain (starting from 2nd chain)
        system_BB_only = []
        for line in pdb_data:
            if line[1] == 'BB':  # select only BB lines:
                system_BB_only.append(line)
        # set the cutoff distance
        max_dist = pow(bb_cutoff, 2)  # (squared, A) distance between two consecutive residues in backbone
        # compute squared distances between sequential BBs
        # input: system_BB_only (2d list), output: new_chain_begins (1d list)
        # calculate squared distances and create an index list of chain beginnings:
        for index, line in enumerate(system_BB_only):
            if index+1 < len(system_BB_only):  # if-clause prevents going out of range for the last BB pair
                # calculate square of distance between all sequential BBs
                dist = math.pow((system_BB_only[index][4] - system_BB_only[index+1][4]), 2) \
                       + math.pow((system_BB_only[index][5] - system_BB_only[index+1][5]), 2) \
                       + math.pow((system_BB_only[index][6] - system_BB_only[index+1][6]), 2)
                if dist > max_dist:
                    new_chain_begins.append(system_BB_only[index+1][0])
        # assign the IDs based on the indices of chain "heads"
        chain_flag = 0  # this variable will change as script progresses down the list of residues
        current_switch = new_chain_begins.pop(0)
        for line in pdb_data:
            if line[0] == current_switch:
                chain_flag += 1
                if new_chain_begins:  # avoid popping an empty list: if only 1 element in list, this step is omitted
                    current_switch = new_chain_begins.pop(0)
            line[-1] = chain_flag

    # PDB-ID based method
    elif chain_sort_method == 1:  # chain-ID based approach
        # insert pdb_chain_ids into the last column:
        for ndx in range(len(pdb_data)):
            pdb_data[ndx][-1] = pdb_chain_ids[ndx]  # these are now strings

    # User input based method
    elif chain_sort_method == 2:
        # insert user input ids into the last column:
        # step 1: basic check if the file has the correct amount of entries
        with open(chain_file, 'r') as f:
            for counter, line in enumerate(f):
                pass
        if counter + 1 != len(pdb_data):
            exit("Error: Number of lines in {} file ({}) is not consistent with the number of "
                 "ATOM records in input pdb ({})".format(chain_file, counter + 1, len(pdb_data)))
        # step 2: open the file again and store the entries in a list:
        userinput_ids = [ ]
        with open(chain_file, 'r') as f:
            for line in f:
                userinput_ids.append(line.strip())
        # step 3: insert chain ids into the pdb_data
        for ndx in range(len(pdb_data)):
            pdb_data[ndx][-1] = userinput_ids[ndx]

    # count the length of the selected chain in atoms and residues:
    current_chain_atomnums = [ ]
    current_chain_resnums = [ ]
    for line in pdb_data:
        if line[-1] == selected_chain: # gets all residue numbers for selected chain ID
            current_chain_atomnums.append(line[0])
            current_chain_resnums.append(line[3])
    mod_chain_at_nums = [min(current_chain_atomnums), max(current_chain_atomnums)]
    mod_chain_res_nums = [min(current_chain_resnums), max(current_chain_resnums)]

    # put single chain length (in atoms and residues) in one array for later use:
    single_chain_mods = [(mod_chain_at_nums[1] + 1) - mod_chain_at_nums[0],
                         (mod_chain_res_nums[1] + 1) - mod_chain_res_nums[0]]
    if debug_mode:
        print("Length of chain " + selected_chain + " in atoms = " + str(single_chain_mods[0]))
        print("Length of chain " + selected_chain + " in residues = " + str(single_chain_mods[1]))

    return pdb_data, single_chain_mods, mod_chain_at_nums, mod_chain_res_nums


def sym_pair_sort(sym_pairs, out_pdb, single_chain_mods):
    """
    1. takes sym_pairs list and separates it into INTRA and INTER type pairs based on the chain IDs provided by out_pdb
    (pdb_data) list
    2. shortens/filters the pairs lists (INTRA and INTER) by choosing a single chain and performing the modulo
    operation for INTER pairs (so that all atomnr and resnr are < chain length)
    2.2. if after modulo operation, two or more pairs end up having the same indices, only the 1st is taken into account

    :param sym_pairs: atomnr_i BB, atomnr_j BB, Vii, Wii, resnr_i BB, resnr_j BB, dist, sigma (from map_OVrCSU)
    :param out_pdb: (pdb_data) atomnr, atomname, resname, resnr, x,y,z, chainID
    :param single_chain_mods: atomnr, resnr
    Returns:
        sym_pairs_intra/inter (list): (upd) atomnr_i, (upd) atomnr_j, Vii, Wii, (upd) resnr_i, (upd) resnr_j, dist, sigma
        resnr_intra/inter (list): unique sorted residue numbers of residues involved in INTRA and INTER pairs
    """
    sym_pairs_intra = [ ]
    sym_pairs_inter = [ ]
    for index, pair in enumerate(sym_pairs):
        index_i = sym_pairs[index][0]
        index_j = sym_pairs[index][1]
        #  if chain ID (last column) of atom i is the same as chain ID of atom j --> INTRA, otherwise INTER
        if out_pdb[index_i][-1] == out_pdb[index_j][-1]:
            sym_pairs_intra.append(pair)
        else:
            sym_pairs_inter.append(pair)
    if mod_enabled:  # edit sym_pairs_inter and intra by shortening them, otherwise skip
        # INTRA: take only pairs from the 1st chain, i.e. atom_ids < chain length:
        mono_intra = []
        for pair in sym_pairs_intra:
            #if sym_pairs_intra[i][0] >= mod_chain_at_nums[0] and sym_pairs_intra[i][0] <= mod_chain_at_nums[1]:
            if mod_chain_at_nums[0] <= pair[0] <= mod_chain_at_nums[1]:
                # (2nd half of the pair will be in the same chain by def)
                # reset atom and residue numbers to starting with 1 using mod operation:
                mono_intra_0 = pair[0] % single_chain_mods[0]
                if mono_intra_0 == 0:
                    mono_intra_0 = single_chain_mods[0]
                mono_intra_1 = pair[1] % single_chain_mods[0]
                if mono_intra_1 == 0:
                    mono_intra_1 = single_chain_mods[0]
                mono_intra_4 = pair[4] % single_chain_mods[1]
                if mono_intra_4 == 0:
                    mono_intra_4 = single_chain_mods[1]
                mono_intra_5 = pair[5] % single_chain_mods[1]
                if mono_intra_5 == 0:
                    mono_intra_5 = single_chain_mods[1]
                mono_intra.append([mono_intra_0, mono_intra_1, pair[2], pair[3], mono_intra_4, mono_intra_5,
                                   pair[6], pair[7]])
        sym_pairs_intra = mono_intra
        # INTER: filter pairs involving chain 1, then recalculate indices using mod to fit in one chain range
        mono_inter = []
        for pair in sym_pairs_inter:
            # if both atom indices of the current pair are within the chosen chain:
            if (mod_chain_at_nums[0] <= pair[0] <= mod_chain_at_nums[1]) or \
                    (mod_chain_at_nums[0] <= pair[1] <= mod_chain_at_nums[1]):
                # reset atom and residue numbers to starting with 1 using mod operation:
                mono_inter_0 = pair[0] % single_chain_mods[0]
                if mono_inter_0 == 0:
                    mono_inter_0 = single_chain_mods[0]
                mono_inter_1 = pair[1] % single_chain_mods[0]
                if mono_inter_1 == 0:
                    mono_inter_1 = single_chain_mods[0]
                mono_inter_4 = pair[4] % single_chain_mods[1]
                if mono_inter_4 == 0:
                    mono_inter_4 = single_chain_mods[1]
                mono_inter_5 = pair[5] % single_chain_mods[1]
                if mono_inter_5 == 0:
                    mono_inter_5 = single_chain_mods[1]
                mono_inter.append([mono_inter_0, mono_inter_1,
                                   pair[2], pair[3],
                                   mono_inter_4, mono_inter_5,
                                   pair[6], pair[7]])
        # todo: reconsider how to get rid of repeats in mono_inter?
        # Currently: simply takes only the first instance of a pair, i.e. C3-C22
        # (second C3-C22 or a C22-C3 pair will be ignored)
        for i in range(0, len(mono_inter)):
            for j in range(i+1, len(mono_inter)):
                if (((mono_inter[i][4] == mono_inter[j][4]) and (mono_inter[i][5] == mono_inter[j][5]))
                        or ((mono_inter[i][5] == mono_inter[j][4]) and (mono_inter[i][4] == mono_inter[j][5]))):
                    mono_inter[j][0] = "SKIP"   # tried using list.pop(j) but it messes up the indexing in the loop
        mono_inter_sorted = []
        for line in mono_inter:
            if line[0] == "SKIP":
                continue
            else:
                mono_inter_sorted.append(line)
        sym_pairs_inter = mono_inter_sorted
        if debug_mode:
            print("Number of mono INTRA pairs: " + str(len(sym_pairs_intra)))
            print("Number of mono INTER pairs: " + str(len(sym_pairs_inter)))

    if debug_mode and not mod_enabled:
        print("Number of INTRA pairs: " + str(len(sym_pairs_intra)))
        print("Number of INTER pairs: " + str(len(sym_pairs_inter)))

    # write the residue numbers for intra sites A,B and inter sites C,D
    # (this can't be done earlier since the intra-inter sorting happens in this function)
    resnr_intra = [ ]
    resnr_inter = [ ]
    for line in sym_pairs_intra:
        resnr_intra.append(line[4])
        resnr_intra.append(line[5])
    resnr_intra = list(set(resnr_intra))
    resnr_intra.sort()
    for line in sym_pairs_inter:
        resnr_inter.append(line[4])
        resnr_inter.append(line[5])
    resnr_inter = list(set(resnr_inter))
    resnr_inter.sort()
    return sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter


def get_eps_array(sym_pairs_intra, sym_pairs_inter, go_eps_intra, go_eps_inter):
    """
    temporary function, converts scalar eps values into arrays
    currently: same as scalar go_eps_intra/inter; later: read from file or anywhere else
    requires knowledge of the amount of inter and intra pairs (=lengths of sym_pairs_intra/inter)

    Returns:
        eps_intra_custom, eps_inter_custom (lists)
    """
    eps_intra_custom = []
    eps_inter_custom = []
    for _ in sym_pairs_intra:
        eps_intra_custom.append(go_eps_intra)
    for _ in sym_pairs_inter:
        eps_inter_custom.append(go_eps_inter)
    #print(len(eps_intra_custom), len(eps_inter_custom))
    return eps_intra_custom, eps_inter_custom


def update_pdb(file_pref, out_pdb, resnr_intra, resnr_inter, selected_chain):
    """
    1. appends virtual site entries for INTRA and INTER go sites to the copy of (now single chain) out_pdb
    2. creates a list for the "virtual sites" section in the output itp
    3. creates lists for "exclusions" section of the output itp (separate for VS B,C,D)

    Returns:
        vwb_excl, vwc_excl, vwd_excl (lists): resnr, VS (B,C,D) atomnr
        virtual_sites (list): VS (B,C,D) atomnr, BB atomnr
        upd_out_pdb (list): atomnr, atomname, resname, resnr, x, y, z, chain_id
    """
    # atomnr: running index, starts as last index in out_pdb OR last index of a single chain (mono)
    # VS atomnr and resnr will start from atomnr+1
    if mod_enabled:
        atomnr = single_chain_mods[0]
    else:
        atomnr = out_pdb[-1][0]

    upd_out_pdb = out_pdb.copy()

    # for exclusions: additional lists for later:
    vwb_excl = [] # resnr, VS atomnr
    vwc_excl = []
    vwd_excl = []
    virtual_sites = [] # VS atomnr, its respective BB atomnr
    # given the list of resnr_intra, write new entries for virtual sites A:

    if mod_enabled:  # writes pdb with single chain out of N identical chains in the input complex
        running_at_ndx = 1
        reset_res_ndx = 1
        reset_res_ndx_chain = [ ] # list of residue indices starting from 1. Will be used in writing section [ atoms ]
        mono_pdb = []
        # write the Martini chain section in pdb
        for ndx, line in enumerate(out_pdb):
            if line[-1] == selected_chain:
                reset_res_ndx_chain.append(reset_res_ndx)
                mono_pdb.append([running_at_ndx, line[1], line[2], reset_res_ndx, line[4], line[5], line[6], line[7]])
                # if the residue number changes in the next line, increment the residue number:
                if ndx+1 < len(out_pdb): # if-clause prevents going out of range for the last pdb line
                    if out_pdb[ndx+1][3] != out_pdb[ndx][3]:
                        reset_res_ndx += 1
                # atom number is incremented every line regardless
                running_at_ndx += 1
        upd_out_pdb = mono_pdb
        # write the Go chain section in pdb, as well as exclusion lists for itps later
        # this section has to use the mono_pbd list and not out_pdb so that all the atom and residue numbering
        # is as in the selected chain (after the modulo operation), as well as the coordinates!!
        for k in resnr_intra:
            atomnr += 1
            for line in mono_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWA', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
        # repeat for sites B:
        for k in resnr_intra:
            atomnr += 1
            for line in mono_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWB', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwb_excl.append([k, atomnr])  # dict: key=resnr : val=atomnr
        # sites C and D:
        for k in resnr_inter:
            atomnr += 1
            for line in mono_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWC', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwc_excl.append([k, atomnr])  # dict: key=resnr : val=atomnr
        for k in resnr_inter:
            atomnr += 1
            for line in mono_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWD', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwd_excl.append([k, atomnr])  # dict: key=resnr : val=atomnr
    else:
        for k in resnr_intra:
            atomnr += 1
            for line in out_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWA', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
        # repeat for sites B:
        for k in resnr_intra:
            atomnr += 1
            for line in out_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWB', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwb_excl.append([k, atomnr]) # dict: key=resnr : val=atomnr
        # sites C and D:
        for k in resnr_inter:
            atomnr += 1
            for line in out_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWC', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwc_excl.append([k, atomnr])  # dict: key=resnr : val=atomnr
        for k in resnr_inter:
            atomnr += 1
            for line in out_pdb:
                if line[1] == 'BB' and line[3] == k:
                    resname = line[2]
                    x = line[4]
                    y = line[5]
                    z = line[6]
                    ch_id = line[7]
                    upd_out_pdb.append([atomnr, 'VWD', resname, k, x, y, z, ch_id])
                    virtual_sites.append([atomnr, line[0]])
                    vwd_excl.append([k, atomnr])  # dict: key=resnr : val=atomnr
        reset_res_ndx_chain = 0 # in non-mono case, this variable won't be used

    # write an updated pdb file:
    if mod_enabled:
        pdb_name = '_cg_go_mono.pdb'
    else:
        pdb_name = '_cg_go.pdb'
    with open(file_pref + '_' + selected_chain + pdb_name, 'w') as f:
        for line in upd_out_pdb:
            s2print = "ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (line[0], line[1], line[2], line[3],
                                                                        line[4], line[5], line[6])
            f.write(s2print)
        f.write('END   ')

    return vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb, reset_res_ndx_chain


def get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra, sym_pairs_inter):
    """
    creates exclusion lists for VS B,C,D (atomnr - atomnr) based on the lists of intra and inter residue number pairs
    (maps indices of VS to these residue pairs)

    Returns:
        excl_b, excl_c, excl_d (lists): atomnr VS, atomnr VS
        intra_pairs, inter_pairs (lists): resnr VS, resnr VS
    """
    # turn arrays into dicts:
    # create a dictionary: key: resnr; value: atomnr
    vwb_excl = np.array(vwb_excl)
    vwb_excl_dict = dict(zip(vwb_excl[:, 0], vwb_excl[:, 1]))  # {1: 165, 2: 166, 3: 167, 4: 168 ...}
    vwc_excl = np.array(vwc_excl)
    vwc_excl_dict = dict(zip(vwc_excl[:, 0], vwc_excl[:, 1]))  # {1: 207, 2: 208, 3: 209, 4: 210 ...}
    vwd_excl = np.array(vwd_excl)
    vwd_excl_dict = dict(zip(vwd_excl[:, 0], vwd_excl[:, 1]))  # {1: 244, 2: 245, 3: 246, 4: 247 ...}

    # look up the intra pairs (by resnr) in the sym_pairs_intra[k][4] - [5] list and write exclusion lists:
    excl_b = []
    excl_c = []
    excl_d = []
    intra_pairs = []
    inter_pairs = []
    for k in sym_pairs_intra:
        intra_pairs.append([k[4], k[5]])
    for k in sym_pairs_inter:
        inter_pairs.append([k[4], k[5]])
    for pair in intra_pairs:
        replaced_excl_b = [x if x not in vwb_excl_dict else vwb_excl_dict[x] for x in pair]
        excl_b.append(replaced_excl_b)
    for pair in inter_pairs:
        replaced_excl_c = [x if x not in vwc_excl_dict else vwc_excl_dict[x] for x in pair]
        excl_c.append(replaced_excl_c)
        replaced_excl_d = [x if x not in vwd_excl_dict else vwd_excl_dict[x] for x in pair]
        excl_d.append(replaced_excl_d)
    # can this be shorter?
    #for line in excl_b:
    #    print(line)
    return excl_b, excl_c, excl_d, intra_pairs, inter_pairs


def get_bb_pair_sigma_epsilon(itp_filename, martini_file, sym_pairs_inter):
    """
    Extracts the sigma-epsilon value pairs from the martini .itp
    (needed to create VS D which counteract BB particle interactions, hence eps(VSD_i) = -eps(BB_i)

    Returns:
        sigma_d, eps_d (lists): used in (file_pref)_nonbond_params_go.itp for VS D
    """

    # 1. extract "atom index - atomtype" information from the [ atoms ] section of molecule.itp
    atoms_section = []
    with open(itp_filename, 'r') as file:
        match = False  # logical switch allows to read only lines between (excluding) the two matched lines
        for line in file:
            if re.search(r'\[ atoms ]', line):
                match = True
                continue
            elif re.search(r'\[ position_restraints ]', line):  #todo: or line == '\n' - if sections are in different order
                match = False
                continue
            elif match:
                line = line.split()
                if len(line) > 1 and line[4] == 'BB':  # line length check gets rid of a new line at the end of section
                    atoms_section.append([int(line[0]), line[1], int(line[2]), line[3]])
                    #            atomindex    atomtype      resnr       resname
                    #              4              SP2         3            VAL
    # create a dictionary: key: atomindex; value: atomtype
    atoms_section = np.array(atoms_section)
    atomtype_dict = dict(zip(atoms_section[:, 0].astype('int'), atoms_section[:, 1]))  # mind the typecasting!
    # {1: 'Q5', 2: 'P2', 4: 'SP2', ...}

    # 2. dataframe for sym_pairs_inter (mind the typecasting! must match datatype of the dict keys (int)):
    # todo: add resnumbers from sym_pairs_inter?
    ndxpairs_df = pd.DataFrame(np.array(sym_pairs_inter)[:, 0:2].astype('int'), columns=['atomndx_i', 'atomndx_j'])
    # map atomtype from the dictionary to atom indices; add the resulting values as new columns of the dataframe:
    ndxpairs_df['atomtype_i'] = ndxpairs_df['atomndx_i'].map(atomtype_dict)
    ndxpairs_df['atomtype_j'] = ndxpairs_df['atomndx_j'].map(atomtype_dict)
    #     atomndx_i  atomndx_j atomtype_i atomtype_j
    # 0           1        121         Q5         Q5

    # extra column containing pairs of atomtypes (to match with the unique key of the dictionary later)
    ndxpairs_df['pair_key'] = ndxpairs_df['atomtype_i'].astype(str) + ' ' + ndxpairs_df['atomtype_j'].astype(str)

    # 3. filter for the "ff database" (martini [ nonbonded_params ]) = list of unique atomtypes found in pairs:
    unique_atomtypes = pd.concat([ndxpairs_df['atomtype_i'], ndxpairs_df['atomtype_j']]).unique()
    # ['Q5' 'P2' 'SP2' 'SP2a' 'SP1']

    # 4. fetch the sigma-epsilon values in the database (martini_v3.0.0.itp - [ nonbond_params])
    # open martini_v3 file and save all entries to a new dataframe:
    martini_nonbond = []
    with open(martini_file, 'r') as f:
        match = False
        for line in f:
            if re.search(r'\[ nonbond_params ]', line):
                match = True
                continue
            elif len(line.split()) < 5 or line == '\n':  # end of [ nonbond_params ]: empty line or other N(columns)
                match = False
                continue
            elif match:  # for lines inside the [ nonbond_params ] section: matching and writing?
                # todo: is it possible to apply np.where directly here?
                line = line.split()
                martini_nonbond.append([line[0], line[1], line[3], line[4]])  # contains the entire nonbond section of martini

    nonbonded_df = pd.DataFrame(martini_nonbond, columns=['atomtype_i', 'atomtype_j', 'sigma', 'eps'])
    # filter out the atomtypes not found in unique_atomtypes[ ] of the system (see step 3):
    nonbonded_df = nonbonded_df[nonbonded_df['atomtype_i'].isin(unique_atomtypes) & nonbonded_df['atomtype_j'].isin(unique_atomtypes)]
    #  atomtype_i atomtype_j         sigma           eps
    # 9438           P2         P2  4.700000e-01  4.060000e+00
    # 9698           P2        SP2  4.300000e-01  3.770000e+00

    # flip the non-symmetric pairs and append them at the end (to take into account all options):
    nonbonded_df['asym'] = np.where(nonbonded_df['atomtype_i'] == nonbonded_df['atomtype_j'], 1, 0)
    temp_df = nonbonded_df[nonbonded_df['asym'] == 0]
    temp_df = temp_df[['atomtype_j', 'atomtype_i', 'sigma', 'eps']] # flipped asymmetric pairs
    nonbonded_df = nonbonded_df.drop('asym', axis=1) # get rid of the temp filter axis
    nonbonded_df = pd.concat([ nonbonded_df, temp_df.rename(columns={'atomtype_j':'atomtype_i',
                                                                     'atomtype_i':'atomtype_j'}) ], ignore_index=True)
    # add a pair column and put it first:
    nonbonded_df['pair_key'] = nonbonded_df['atomtype_i'].astype(str) + ' ' + nonbonded_df['atomtype_j'].astype(str)
    nonbonded_df = nonbonded_df[['pair_key', 'sigma', 'eps']]
    #  pair_key         sigma           eps
    # 0    P6 P6  4.700000e-01  4.990000e+00
    # 1    P6 P5  4.700000e-01  4.730000e+00

    # transpose and turn into a dictionary with pairs (all combinations) as unique keys:
    sig_eps_dict = nonbonded_df.set_index('pair_key').T.to_dict('list')
    # {'P2 P2': ['4.700000e-01', '4.060000e+00'], 'P2 SP2': ['4.300000e-01', '3.770000e+00'], ...}

    # map the dictionary entries onto the dataframe of index pairs and append the matching sigma/eps pairs as new col:
    ndxpairs_df['new'] = ndxpairs_df['pair_key'].map(sig_eps_dict)
    # split the 'new' column into sigma and epsilon:
    sig_eps = pd.DataFrame(ndxpairs_df['new'].to_list(), columns=['sigma', 'eps'])
    sigma_d = sig_eps['sigma'].to_list()  # final list of all sigmas
    eps_d = sig_eps['eps'].to_list()  # final list of all eps values (hopefully, in order...)

    return sigma_d, eps_d


########## FILE WRITING PROCEDURES ##########
def write_include_files(file_pref, missRes, go_eps_intra, go_eps_inter, c6c12,     #todo: either remove go_eps_intra, go_eps_inter or eps_intra_custom, eps_inter_custom
                sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs, inter_pairs,
                        virtual_sites, upd_out_pdb, fnames, sigma_d, eps_d, eps_intra_custom, eps_inter_custom, selected_chain):
    """
    writes all included .itp files
    """
    # main.top -> martini_v3.0.0_go.itp [ atomtypes ]-> atomtypes_go.itp -> (file_pref)_atomtypes_go.itp
    # here: sets of (file_pref)_[A-D] VSites
    with open(file_pref + '_' + selected_chain + '_' + fnames[0], 'w') as f:
        f.write('; protein BB virtual particles \n')
        f.write('; INTRA particles\n')
        for k in resnr_intra:
            s2print = '%s_A%s 0.0 0.000 A 0.0 0.0 \n' % (
            file_pref, str(k + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        for k in resnr_intra:
            s2print = '%s_B%s 0.0 0.000 A 0.0 0.0 \n' % (
                file_pref, str(k + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        f.write('; INTER particles\n')
        for j in resnr_inter:
            s2print = '%s_C%s 0.0 0.000 A 0.0 0.0 \n' % (
                file_pref, str(j + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        for j in resnr_inter:
            s2print = '%s_D%s 0.0 0.000 A 0.0 0.0 \n' % (
                file_pref, str(j + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
    # add the name of the created .itp file into the wrapper atomtypes_go.itp
    with open('atomtypes_go.itp', 'w') as f:  # this filename doesn't change, unless it changes in martini_v3.0.0_go.itp
        s2print = '#include "%s_%s_%s"\n' % (file_pref, selected_chain, fnames[0])
        f.write(s2print)

    # main.top -> martini_v3.0.0_go.itp [ nonbond_params ]-> nonbond_params_go.itp -> (file_pref)_nonbond_params_go.itp
    with open(file_pref + '_' + selected_chain + '_' + fnames[1], 'w') as f:
        f.write('; OV + symmetric rCSU contacts \n')
        if (c6c12 == 1):  # this setting uses sigma/eps computed using Vii, Wii
            f.write('; not implemented yet\n')
        else:  # default setting, uses sigma + go_eps_*
            f.write('; INTRA section: A-B pairs (+/- go_eps_intra) \n')
            for k in range(0, len(sym_pairs_intra)):
                # VWA-VWA pair: BB go_eps_intra
                s2print = ' %s_A%s  %s_A%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n' % (file_pref,
                                                                                      str(int(sym_pairs_intra[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_intra[k][5])),
                                                                                      sym_pairs_intra[k][7],
                                                                                      eps_intra_custom[k],             # go_eps_intra
                                                                                      str(int(sym_pairs_intra[k][0])),
                                                                                      str(int(sym_pairs_intra[k][1])),
                                                                                      sym_pairs_intra[k][6])
                f.write(s2print)
                # VWB-VWB pair: BB -go_eps_intra
                s2print = ' %s_B%s  %s_B%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n' % (file_pref,
                                                                                      str(int(sym_pairs_intra[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_intra[k][5])),
                                                                                      sym_pairs_intra[k][7],
                                                                                      -eps_intra_custom[k] + 0.00001,  # avoid exact val, -go_eps_intra
                                                                                      str(int(sym_pairs_intra[k][0])),
                                                                                      str(int(sym_pairs_intra[k][1])),
                                                                                      sym_pairs_intra[k][6])
                f.write(s2print)
            f.write('; INTER section: C (go_eps_inter) \n')
            for k in range(0, len(sym_pairs_inter)):
                s2print = ' %s_C%s  %s_C%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n' % (file_pref,
                                                                                      str(int(sym_pairs_inter[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_inter[k][5])),
                                                                                      sym_pairs_inter[k][7],
                                                                                      eps_inter_custom[k],               # go_eps_inter
                                                                                      str(int(sym_pairs_inter[k][0])),
                                                                                      str(int(sym_pairs_inter[k][1])),
                                                                                      sym_pairs_inter[k][6])
                f.write(s2print)
            f.write('; INTER section: D (-go_eps_BB) \n')
            for k in range(0, len(sym_pairs_inter)):
                # replaced sym_pairs_inter[k][7] and go_eps_inter with BB sigma/eps values
                s2print = ' %s_D%s  %s_D%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n' % (file_pref,
                                                                                      str(int(sym_pairs_inter[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_inter[k][5])),
                                                                                      float(sigma_d[k]),
                                                                                      -float(eps_d[k]) + 0.00001,  # avoid exact val
                                                                                      str(int(sym_pairs_inter[k][0])),
                                                                                      str(int(sym_pairs_inter[k][1])),
                                                                                      sym_pairs_inter[k][6])
                f.write(s2print)
    # add the name of the created .itp file into the wrapper go-table_VirtGoSites.itp
    with open('nonbond_params_go.itp','w') as f:  # this filename doesn't change, unless it changes in martini_v3.0.0_go.itp
        s2print = '#include "%s_%s_%s"\n' % (file_pref, selected_chain, fnames[1])
        f.write(s2print)

    # main.top -> (file_pref)_go.itp [ atoms ] -> (file_pref)_atoms_go.itp
    with open(file_pref + '_' + selected_chain + '_' + fnames[2], 'w') as f:
        f.write('; virtual sites\n')
        for entry in upd_out_pdb:
            if entry[1] == 'VWA':
                suffix = 'A'
            elif entry[1] == 'VWB':
                suffix = 'B'
            elif entry[1] == 'VWC':
                suffix = 'C'
            elif entry[1] == 'VWD':
                suffix = 'D'
            else:
                continue
            s2print = '%3d %s_%s%-3d %6d %3s %-3s %-5d 0.0\n' % (entry[0], file_pref, suffix, entry[3], entry[3],
                                                                 entry[2], entry[1], entry[0])
            # atomnr atomtype=(filepref_[A-D]resnr) resnr resname atomname chrggrp=atomnr q=0.0
            f.write(s2print)

    # main.top -> molecule.itp [ virtual_sitesn ] -> (file_pref)_virtual_sitesn_go.itp
    with open(file_pref + '_' + selected_chain + '_' + fnames[3], 'w') as f:
        f.write('; VS index - funct - constructing atom index/indices\n')
        for pair in virtual_sites:
            s2print = '%4d 1 %3d\n' % (pair[0], pair[1])
            f.write(s2print)

    # main.top -> molecule.itp [ exclusions ] -> (file_pref)_exclusions_go.itp
    # exclusion pairs sorted by intra-inter:
    with open(file_pref + '_' + selected_chain + '_' + fnames[4], 'w') as f:
        f.write('; [ exclusions for intra BB sites ]\n')
        f.write('; atomnr atomnr  -  resnr resnr\n')
        for line in sym_pairs_intra:
            s2print = ' %d  %d  \t ;  %d  %d \n' % (line[0], line[1], line[4], line[5])
            f.write(s2print)
        f.write('; [ exclusions for intra VWB sites ]\n')
        for ind in range(len(excl_b)):
            s2print = ' %d  %d  \t ;  %d  %d \n' % (excl_b[ind][0], excl_b[ind][1],
                                                    intra_pairs[ind][0], intra_pairs[ind][1])
            f.write(s2print)
        f.write('; [ exclusions for inter VWC sites ]\n')
        for ind in range(len(excl_c)):
            s2print = " %d  %d  \t ;  %d  %d \n" % (excl_c[ind][0], excl_c[ind][1],
                                                    inter_pairs[ind][0], inter_pairs[ind][1])
            f.write(s2print)
        f.write('; [ exclusions for inter VWD sites ]\n')
        for ind in range(len(excl_d)):
            s2print = ' %d  %d  \t ;  %d  %d \n' % (excl_d[ind][0], excl_d[ind][1],
                                                    inter_pairs[ind][0], inter_pairs[ind][1])
            f.write(s2print)

    # visualize the go-bonds: (file_pref)_viz_go.itp
    with open(file_pref + '_' + selected_chain + '_' + fnames[5], 'w') as f:
        f.write('; Intra Go bonds as harmonic bonds (VWB)\n')
        for ind in range(len(sym_pairs_intra)):
            s2print = ' %d  %d  1  %.3f  1250\n' % (excl_b[ind][0], excl_b[ind][1], sym_pairs_intra[ind][6])
            f.write(s2print)
        f.write('; Inter Go bonds as harmonic bonds (VWC)\n')
        for ind in range(len(sym_pairs_inter)):
            s2print = ' %d  %d  1  %.3f  1250\n' % (excl_c[ind][0], excl_c[ind][1], sym_pairs_inter[ind][6])
            f.write(s2print)

def write_main_top(file_pref, molecule_itp, fnames, selected_chain, mod_chain_at_nums):
    """
    writes the main .itp file and .top file
    """
    # read and store sections of input itp file:
    # dictionary of sections, with a value = columns to modify via modulo operation
    itp_section_list_dict = {'atoms': (0, 2, 5), 'position_restraints': 0,
                             'bonds': (0, 1), 'constraints': (0, 1),
                             'angles': (0, 1, 2), 'dihedrals': (0, 1, 2, 3),
                             'exclusions': 0, 'virtual_sitesn': 0}

    all_itp_sections = [] # will contain a list of lists with headless section contents
    for section in itp_section_list_dict: # section = dict key only
        # print(itp_section_list_dict[section]) # get value by key
        with open(molecule_itp, 'r') as file:
            temp_section = []  # 2d list to store the current section in
            match = False
            for line in file:
                # if line contains key value, save from next line:
                if line.startswith('[') and section in line:
                    match = True
                    continue
                # if line contains '[' again, stop before that, don't write that line
                elif line.startswith('[') and section not in line:
                    match = False
                    continue
                # ignore empty lines:
                elif line.startswith('\n'):
                    continue
                elif match:
                    if line.startswith(('#', ';')):
                        temp_section.append(line)
                    else:  # process the actual data
                        line = line.split(';') # get rid of all in-line comments for consistent column numbers
                        line = line[0].split()
                        if section == 'atoms':
                            # atoms section: check if atomic index is in the atomistic range
                            if mod_chain_at_nums[0] <= int(line[0]) <= mod_chain_at_nums[1]:
                                line[0] = int(line[0]) % single_chain_mods[0]  # atomnr
                                line[5] = line[0]  # "charge group" in col. 6 (CG: per bead)
                                if line[0] == 0:
                                    line[0] = single_chain_mods[0]
                                    line[5] = line[0]
                                line[2] = int(line[2]) % single_chain_mods[1]  # resnr
                                if line[2] == 0:
                                    line[2] = single_chain_mods[1]
                                temp_section.append(line)
                        elif section == 'position_restraints':
                            # posres section: check if atom is in the selected chain
                            if mod_chain_at_nums[0] <= int(line[0]) <= mod_chain_at_nums[1]:
                                line[0] = int(line[0]) % single_chain_mods[0]  # atomnr
                                if line[0] == 0:
                                    line[0] = single_chain_mods[0]
                                temp_section.append(line)
                        elif section == 'exclusions':
                            # all indices have to be checked and converted
                            if all(mod_chain_at_nums[0] <= int(i) <= mod_chain_at_nums[1] for i in line):
                                for i in range(len(line)):
                                    line[i] = int(line[i]) % single_chain_mods[0]  # i-th atomnr
                                    if line[i] == 0:
                                        line[i] = single_chain_mods[0]
                                temp_section.append(line)
                        elif section == 'virtual_sitesn':
                            # separate indices (1st and 3rd-nth columns) and func type (always 2nd column)
                            func_type = line.pop(1)
                            # check and convert all indices
                            if all(mod_chain_at_nums[0] <= int(i) <= mod_chain_at_nums[1] for i in line):
                                for i in range(len(line)):
                                    line[i] = int(line[i]) % single_chain_mods[0]  # i-th atomnr
                                    if line[i] == 0:
                                        line[i] = single_chain_mods[0]
                                # join indices (line list) and func_type back together in correct order:
                                line.insert(1, func_type)
                                # append as usual
                                temp_section.append(line)
                        # other sections: bonds, constraints, angles, dihedrals
                        else:
                            # list of columns containing values to modulo:
                            line_atoms = [line[i] for i in itp_section_list_dict[section]]
                            # check if all selected atoms simultaneously fall into the range (if and and and)
                            if all(mod_chain_at_nums[0] <= int(i) <= mod_chain_at_nums[1] for i in line_atoms):
                                # apply modulo operation to all atoms selected by dictionary:
                                for i in itp_section_list_dict[section]:
                                    line[i] = int(line[i]) % single_chain_mods[0]  # i-th atomnr
                                    if line[i] == 0:
                                        line[i] = single_chain_mods[0]
                                temp_section.append(line)
        all_itp_sections.append(temp_section)

    # write modified itp file
    with open(file_pref + '_' + selected_chain + '_' + fnames[6], 'w') as file:
        # [ moleculetype ]
        file.write('[ moleculetype ]\n' + file_pref + '    1\n\n')

        # [ atoms ]
        file.write('[ atoms ]\n')
        for line in all_itp_sections[0]:
            s2print = '%5d %-4s %4d %3s %-3s %-5d %4s\n' % (line[0], line[1], line[2], line[3],
                                                            line[4], line[5], line[6])
            file.write(s2print)
        file.write('#include "' + file_pref + '_' + selected_chain + '_' + fnames[2] + '"\n\n')

        # [ position_restraints ]
        file.write('[ position_restraints ]\n')
        for line in all_itp_sections[1]:
            if len(line) == 5:
                s2print = '%4d %2s  %s  %s  %s\n' % (line[0], line[1], line[2], line[3], line[4])
                file.write(s2print)
            else:
                file.write(line)
        file.write('\n')

        # [ bonds ]
        file.write('[ bonds ]\n')
        for line in all_itp_sections[2]:
            if len(line) == 5:
                s2print = '%4d %4d %2s  %s %s\n' % (line[0], line[1], line[2], line[3], line[4])
                file.write(s2print)
            else:
                file.write(line)
        file.write('\n')

        # [ constraints ]
        file.write('[ constraints ]\n')
        for line in all_itp_sections[3]:
            if len(line) == 4:
                s2print = '%4d %4d  %s  %s\n' % (line[0], line[1], line[2], line[3])
                file.write(s2print)
            else:
                file.write(line)
        file.write('\n')

        # [ angles ]
        file.write('[ angles ]\n')
        for line in all_itp_sections[4]:
            if len(line) == 6:
                s2print = '%4d %4d %4d %2s %3s %s\n' % (line[0], line[1], line[2], line[3], line[4], line[5])
                file.write(s2print)
            else:
                file.write(line)
        file.write('\n')

        # [ dihedrals ]
        # both regular (func type 1) and improper (func type 2) dihedrals are in the same section
        file.write('[ dihedrals ]\n')
        for line in all_itp_sections[5]:
            # proper dihedrals
            if len(line) == 8:
                s2print = '%4d %4d %4d %4d %s %6s %3s %s\n' % (line[0], line[1], line[2], line[3], line[4],
                                                               line[5], line[6], line[7])
                file.write(s2print)
            # improper dihedrals:
            elif len(line) == 7:
                s2print = '%4d %4d %4d %4d %s %6s %3s\n' % (line[0], line[1], line[2], line[3], line[4],
                                                               line[5], line[6])
                file.write(s2print)
            else:
                file.write(line)
        file.write('\n')

        # [ exclusions ]
        file.write('[ exclusions ]\n')
        for line in all_itp_sections[6]:
            # all indices, each line of variable length: convert list of integers to a single string
            s2print = '  '.join(str(ind) for ind in line) + '\n'
            file.write(s2print)
        file.write('#include "' + file_pref + '_' + selected_chain + '_' + fnames[4] + '"\n\n')

        # last section:
        #file.write('[ virtual_sitesn ]\n' + '#include "' + file_pref + '_' + fnames[3] + '"\n')
        file.write('[ virtual_sitesn ]\n')
        for line in all_itp_sections[7]:
            # all indices, convert list of integers to a single string
            s2print = '  '.join(str(ind) for ind in line) + '\n'
            file.write(s2print)
        file.write('#include "' + file_pref + '_' + selected_chain + '_' + fnames[3] + '"\n\n')


    # top file from scratch:
    with open(file_pref + '_' + selected_chain + '_' + fnames[7], 'w') as f:
        f.write('#define GO_VIRT\n')
        f.write('#include "martini_v3.0.0_go.itp"\n')
        f.write('#include "' + file_pref + '_' + selected_chain + '_' + fnames[6] + '"\n\n')
        f.write('[ system ]\n'+ file_pref + ' complex with Go bonds\n\n')
        f.write('[ molecules ]\n' + file_pref + '     1 \n') # check if 1 can be a variable


##################### MAIN #####################
# parse input arguments, initialize some vars:
args = user_input()
# read contact map data and store it in lists:
indBB, map_OVrCSU, system_pdb_data, pdb_chain_ids = read_data(args.s, args.f)
# write symmetric unsorted Go pairs
sym_pairs = get_go(indBB, map_OVrCSU, args.cutoff_short, args.cutoff_long, args.go_eps_intra, seqDist, args.missres)

# sort Go pairs into intra and inter sub-lists:
out_pdb, single_chain_mods, mod_chain_at_nums, mod_chain_res_nums = assign_chain_ids(system_pdb_data, args.bb_cutoff,
                                                                                     pdb_chain_ids, args.chain_sort,
                                                                                     args.chain_file, args.chain_id)
# group sym_pairs into intra and inter based on their chain IDs
sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter = sym_pair_sort(sym_pairs, out_pdb, single_chain_mods)
# temporary feature: array of separate inter and intra epsilon values:
eps_intra_custom, eps_inter_custom = get_eps_array(sym_pairs_intra, sym_pairs_inter, args.go_eps_intra, args.go_eps_inter)
# retrieve sigma-epsilon values for each BB involved in intra-Go bonds (for D virtual sites)
sigma_d, eps_d = get_bb_pair_sigma_epsilon(args.i, args.nb, sym_pairs_inter)


# write the updated pdb file (VS A-D):
vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb, reset_res_ndx_chain = update_pdb(args.moltype, out_pdb,
                                                                                           resnr_intra, resnr_inter,
                                                                                           args.chain_id)
excl_b, excl_c, excl_d, intra_pairs, inter_pairs = get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra,
                                                                  sym_pairs_inter)
# write the updated itp/top files:
write_include_files(args.moltype, args.missres, args.go_eps_intra,
                    args.go_eps_inter, c6c12, sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs,
                    inter_pairs, virtual_sites, upd_out_pdb, fnames, sigma_d, eps_d, eps_intra_custom, eps_inter_custom,
                    args.chain_id)
write_main_top(args.moltype, args.i, fnames, args.chain_id, mod_chain_at_nums)