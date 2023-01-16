#!/usr/bin/env python

import argparse
import numpy as np
import math
import re
import pandas as pd  # needed for get_bb_pair_sigma_epsilon()
import itertools  # needed for write_main_top()

# global variables:
seqDist = 4  # minimal distance in the sequence to add a elastic bond (ElNedyn=3 [Perriole2009]; Go=4 [Poma2017])
# (this has to result in the correct atom number when added to "k_at" compared to the .itp file)
c6c12 = 0  # if set to 1, the C6 and C12 term are expected in the .itp file; if set to 0, sigma and go_eps are used
mod_enabled = True  # %TODO this is a test feature!!
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


def itp_sections(x):  # helper function used to slice input itp into sections (called in write_main_top())
    if x.startswith('[ '):
        itp_sections.count += 1
    return itp_sections.count


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
    args = parser.parse_args()
    return args


# read_data() parses data from the .map file (output of the rCSU server)
# returns: indBB (list of xyz coords of BB), map_OVrCSU (list: resID resID distance...), pdb_data, pdb_chain_ids
def read_data(cg_pdb, file_contacts):
    # read the pdb file: mind the fixed file format!
    pdb_data = [ ]
    indBB = [ ]  # separate from pdb_data[] because indBB needs to be a numpy array
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

    # read the map file.
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


# get_go() calculates and filters the Go pairs according to requirements (AT server map data -> CG structure)
# returns: sym_pairs = [0:indBB, 1:indBB, 2:sigma, 3:eps, 4:resnr, 5:resnr, 6:distance, 7:sigma_const]
#                     0, 1 - exclusions (molecule.itp)   2,3 - ignored   4-7 - go_table (martini.itp)
# Vii and Wii:  not used in current ver., leftover from the original create_goVirt.py
def get_go(indBB, map_OVrCSU, cutoff_short, cutoff_long, go_eps_intra, seqDist, missRes):
    # calculate the distances based on the coordinates of the CG BB bead
    for k in range(0, len(map_OVrCSU)):
        dist_vec = indBB[ int(map_OVrCSU[k][1])-missRes-1 ,1:4] - indBB[ int(map_OVrCSU[k][0])-missRes-1 ,1:4]
        map_OVrCSU[k][2] = np.linalg.norm(dist_vec) /10     # [Ang] to [nm]

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
                           Vii, Wii,                                         # sigma, eps from map data (not used)
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
        # for line in sym_pairs:
        #     print(line)
        print("Total number of Go-bonds extracted from the map file: " + str(len(sym_pairs)))

    return sym_pairs


########## INTRA-INTER SORTING PROCEDURES ##########
def assign_chain_ids(pdb_data, bb_cutoff, pdb_chain_ids, chain_sort_method, chain_file):
    # Distance-based method: only works for whole chains (no fragments), with all atoms in order
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

    # count the length of the first chain in atoms and residues:
    first_id = pdb_data[0][-1]
    chain_ids = []  # one column of a 2d array (pdb_data) is more accessible on its own
    for line in pdb_data:
        chain_ids.append(line[-1])
    chain_length = chain_ids.count(first_id)  # counted the occurrences of the fist id only
    # put single chain length (in atoms and residues) in one array for later use:
    single_chain_mods = [chain_length, pdb_data[chain_length-1][3]]
    if debug_mode:
        print("Length of 1st chain in atoms = " + str(single_chain_mods[0]))
        print("Length of 1st chain in residues = " + str(single_chain_mods[1]))

    return pdb_data, single_chain_mods


# sym_pair_sort() separates sym_pairs into sym_pairs_intra and sym_pairs_inter based on the output of out_pdb
# function input parameters: sym_pairs, out_pdb
# function output: sym_pairs_intra, sym_pairs_inter
# Q: separate intras further chain-wise - needed or not?
# different epsilon entries: taken from the script input (--go_eps_intra, --go_eps_inter)
def sym_pair_sort(sym_pairs, out_pdb, single_chain_mods):
    sym_pairs_intra = [ ]
    sym_pairs_inter = [ ]
    for index, pair in enumerate(sym_pairs):
        index_i = sym_pairs[index][0]
        index_j = sym_pairs[index][1]
        if out_pdb[index_i][-1] == out_pdb[index_j][-1]:
            sym_pairs_intra.append(pair)
        else:
            sym_pairs_inter.append(pair)
    if mod_enabled:  # edit sym_pairs_inter and intra by shortening them, otherwise skip
        # INTRA: take only pairs from the 1st chain, i.e. atom_ids < chain length:
        mono_intra = []
        for i in range(len(sym_pairs_intra)):
            if sym_pairs_intra[i][0] <= single_chain_mods[0]:  # 2nd half of the pair will be in the same chain by def
                mono_intra.append(sym_pairs_intra[i])
        sym_pairs_intra = mono_intra
        # INTER: filter pairs involving chain 1, then recalculate indices using mod to fit in one chain range
        mono_inter = []
        unique_sigmas = []  # temp array to store a unique sigma value (impromptu dictionary key)
        for pair in sym_pairs_inter:
            if (pair[0] <= single_chain_mods[0]) or (pair[1] <= single_chain_mods[0]):
                if pair[3] not in unique_sigmas:
                    unique_sigmas.append(pair[3])
                    mono_inter.append([pair[0] % single_chain_mods[0], pair[1] % single_chain_mods[0],
                                       pair[2], pair[3],
                                       pair[4] % single_chain_mods[1], pair[5] % single_chain_mods[1],
                                       pair[6], pair[7]])
        sym_pairs_inter = mono_inter
        if debug_mode:
            print("Number of mono INTRA pairs: " + str(len(sym_pairs_intra)))
            print("Number of mono INTER pairs: " + str(len(sym_pairs_inter)))

    if debug_mode and not mod_enabled:
        # print('INTRA pairs')
        # for line in sym_pairs_intra:
        #     print(line)
        # print('INTER pairs')
        # for line in sym_pairs_inter:
        #     print(line)
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


# new pdb file with all relevant particles: CG structure, VS[A-D]
# input: system_pdb_data
# output: updated pdb file (implicitly), exclusions, virtual sites mapping
def update_pdb(file_pref, out_pdb, resnr_intra, resnr_inter):
    # starting atomnr: last index in out_pdb+1 OR last index of a signle chain (mono)
    if mod_enabled:
        atomnr = single_chain_mods[0]
    else:
        atomnr = out_pdb[-1][0]

    upd_out_pdb = out_pdb.copy()

    if mod_enabled:  # pdb with just the 1st of the N identical chains
        mono_pdb = []
        chain_id = out_pdb[0][-1]  # first chain-ID
        for line in out_pdb:
            if line[-1] == chain_id:
                mono_pdb.append(line)
        upd_out_pdb = mono_pdb

    # for exclusions: additional structs for later:
    vwb_excl = []
    vwc_excl = []
    vwd_excl = []
    virtual_sites = []
    # given the list of resnr_intra, write new entries for virtual sites A:
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

    # write an updated pdb file:
    if mod_enabled:
        pdb_name = '_cg_go_mono.pdb'
    else:
        pdb_name = '_cg_go.pdb'
    with open(file_pref + pdb_name, 'w') as f:
        for line in upd_out_pdb:
            s2print = "ATOM  %5d %-4s %3s  %4d    %8.3f%8.3f%8.3f  1.00  0.00\n" % (line[0], line[1], line[2], line[3],
                                                                        line[4], line[5], line[6])
            f.write(s2print)
        f.write('END   ')

    return vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb


def get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra, sym_pairs_inter):
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
def write_include_files(file_pref, missRes, go_eps_intra, go_eps_inter, c6c12,
                sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs, inter_pairs,
                        virtual_sites, upd_out_pdb, fnames, sigma_d, eps_d):
    # main.top -> martini_v3.0.0_go.itp [ atomtypes ]-> atomtypes_go.itp -> (file_pref)_atomtypes_go.itp
    # here: sets of (file_pref)_[A-D] VSites
    with open(file_pref + '_' + fnames[0], 'w') as f:
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
        s2print = '#include "%s_%s"\n' % (file_pref, fnames[0])
        f.write(s2print)

    # main.top -> martini_v3.0.0_go.itp [ nonbond_params ]-> nonbond_params_go.itp -> (file_pref)_nonbond_params_go.itp
    with open(file_pref + '_' + fnames[1], 'w') as f:
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
                                                                                      go_eps_intra,
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
                                                                                      -go_eps_intra + 0.00001,  # avoid exact val
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
                                                                                      go_eps_inter,
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
        s2print = '#include "%s_%s"\n' % (file_pref, fnames[1])
        f.write(s2print)

    # main.top -> (file_pref)_go.itp [ atoms ] -> (file_pref)_atoms_go.itp
    with open(file_pref + '_' + fnames[2], 'w') as f:
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
    with open(file_pref + '_' + fnames[3], 'w') as f:
        f.write('; VS index - funct - constructing atom index/indices\n')
        for pair in virtual_sites:
            s2print = '%4d 1 %3d\n' % (pair[0], pair[1])
            f.write(s2print)

    # main.top -> molecule.itp [ exclusions ] -> (file_pref)_exclusions_go.itp
    # exclusion pairs sorted by intra-inter:
    with open(file_pref + '_' + fnames[4], 'w') as f:
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
    with open(file_pref + '_' + fnames[5], 'w') as f:
        f.write('; Intra Go bonds as harmonic bonds (VWB)\n')
        for ind in range(len(sym_pairs_intra)):
            s2print = ' %d  %d  1  %.3f  1250\n' % (excl_b[ind][0], excl_b[ind][1], sym_pairs_intra[ind][6])
            f.write(s2print)
        f.write('; Inter Go bonds as harmonic bonds (VWC)\n')
        for ind in range(len(sym_pairs_inter)):
            s2print = ' %d  %d  1  %.3f  1250\n' % (excl_c[ind][0], excl_c[ind][1], sym_pairs_inter[ind][6])
            f.write(s2print)


# modifies the .top and .itp written by martinize2 (w/o -go-vs flag), inserts "#include" lines
def write_main_top_files(file_pref, molecule_itp, fnames):
    itp_file = [ ]
    itp_sections.count = 0
    itp_go_sections = ['[ virtual_sitesn ]\n', '[ exclusions ]\n']  # sections requiring modification by Go
    # [ atoms ] section can't be missing; check only for the other two
    input_itp_headers = [ ]
    with open(molecule_itp, 'r') as f:
        for key, section in itertools.groupby(f, itp_sections):
            # print(list(section))
            itp_file.append(list(section))
    # get headers from the input itp:
    for sect in itp_file:
        if sect[0].startswith('[ '):
            # print(elm[0])
            input_itp_headers.append(sect[0])
    # find missing headers:
    missing_headers = list(set(itp_go_sections) - set(input_itp_headers))
    # write out the (modified) itp file
    with open(file_pref + '_' + fnames[6], 'w') as f:
        for section in itp_file:
            if 'moleculetype' in section[0]:  # rewrite this section entirely with the right molecule name
                f.write(section[0])
                f.write(file_pref + '    1\n\n')
            elif 'atoms' in section[0]:  # how to filter for sections that will be expanded
                if mod_enabled:  # only write the first N lines of section (N=atoms in 1st chain)
                    for ndx in range(single_chain_mods[0]+1):
                        f.write(section[ndx])
                else:            # otherwise, write the entire section as in the input itp
                    for line in section:
                        f.write(line)
                # add the "include" line with the VS list:
                f.write('#include "' + file_pref + '_' + fnames[2] + '"\n\n')
            elif 'virtual_sitesn' in section[0]:  # check if the input itp already contains the VS section
                if mod_enabled:  # only write lines with indices from 1st chain
                    for line in section:
                        if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                            line = line.split()
                            line = [ int(ndx) for ndx in line]  # turn each element in list line into int
                            # unless any of the ints > N atoms in one chain (line[1] is func=2, safe to assume is true):
                            if not any(ndx > single_chain_mods[0] for ndx in line):
                                # number of indices on each may vary, use simpler formatting:
                                line = [str(ndx) for ndx in line]  # turn them back into strings...
                                f.write('  '.join(line) + '\n')     # ...and write to file

                        else:
                            f.write(line)
                else:            # if mod_enabled=False, write all lines of the section as in the input itp
                    for line in section:
                        f.write(line)
                f.write('#include "' + file_pref + '_' + fnames[3] + '"\n\n')  # [ virtual_sitesn ]
            elif 'exclusions' in section[0]:
                if mod_enabled:
                    for line in section:
                        if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                            line = line.split()
                            line = [ int(ndx) for ndx in line]  # turn each element in list line into int
                            # unless any of the ints > N atoms in one chain:
                            if not any(ndx > single_chain_mods[0] for ndx in line):
                                # number of indices on each line varies, use simpler formatting:
                                line = [str(ndx) for ndx in line]  # turn them back into strings...
                                f.write('  '.join(line) + '\n')    # ...and write to file
                        else:
                            f.write(line)
                else:
                    for line in section:
                        f.write(line)
                f.write('#include "' + file_pref + '_' + fnames[4] + '"\n\n')
            # now for the sections that only need to be shortened for the single-chain output:
            elif mod_enabled and 'position_restraints' in section[0]:
                for line in section:
                    if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                        line = line.split()
                        # posres: 0=index, 1=functype (1), 2,3,4= x,y,z constants
                        line = [ int(ndx) for ndx in line]
                        if not line[0] > single_chain_mods[0]:
                            s2print = '%d  %d  %d  %d  %d\n' % (line[0], line[1], line[2], line[3], line[4])
                            f.write(s2print)
                    else:
                        f.write(line)
            elif mod_enabled and 'bonds' in section[0]:
                for line in section:
                    if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                        line = line.split()
                        # bonds: 0,1=indices, 2=functype (1), 3,4=bond, const
                        line = [int(line[0]), int(line[1]), int(line[2]), float(line[3]), int(line[4])]
                        if not (line[0] > single_chain_mods[0] or line[1] > single_chain_mods[0]):
                            s2print = '%d  %d  %d  %.3f %d\n' % (line[0], line[1], line[2], line[3], line[4])
                            f.write(s2print)
                    else:
                        f.write(line)
            elif mod_enabled and 'constraints' in section[0]:
                for line in section:
                    if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                        line = line.split()
                        # constraints: 0,1=indices, 2=functype (1), 3=dist
                        line = [int(line[0]), int(line[1]), int(line[2]), float(line[3])]
                        if not (line[0] > single_chain_mods[0] or line[1] > single_chain_mods[0]):
                            s2print = '%d  %d  %d  %.3f\n' % (line[0], line[1], line[2], line[3])
                            f.write(s2print)
                    else:
                        f.write(line)
            elif mod_enabled and 'angles' in section[0]:
                for line in section:
                    if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                        if any(x == ';' for x in line):
                            line = line.split(';')
                            line = line[0].split()  # now the output won't include in-line comments
                        else:
                            line = line.split()
                        # angles: 0-2=indices, 3=functype (2:G96, 10: restricted bending), 4,5 = angle, const
                        line = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4]), float(line[5])]
                        if not (line[0] > single_chain_mods[0] or line[1] > single_chain_mods[0]
                                or line[2] > single_chain_mods[0]):
                            s2print = '%d  %d  %d  %d  %.2f  %.1f\n' % (line[0], line[1], line[2], line[3], line[4], line[5])
                            f.write(s2print)
                    else:
                        f.write(line)
            elif mod_enabled and 'dihedrals' in section[0]:  # both proper and improper dihedrals are processed here
                for line in section:
                    if not (line.startswith('[') or line.startswith(';') or line.startswith('\n') or line.startswith('#')):
                        # if line contains a comment: remove comment
                        if any(x == ';' for x in line):
                            line = line.split(';')
                            line = line[0].split()
                        else:
                            line = line.split()
                        # proper dihedrals: 0-3=indices, 4=functype (1), 5,6,7 = angle, const, multiplicity
                        # improper dihedrals: 0-3=indices, 4=functype (2), 5,6 = angle, const
                        if line[4] == '1':  # proper dihedral
                            line = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]),
                                    float(line[5]), float(line[6]), int(line[7])]
                            # filter indices for the first chain only:
                            if not (line[0] > single_chain_mods[0] or line[1] > single_chain_mods[0]
                                    or line[2] > single_chain_mods[0] or line[3] > single_chain_mods[0]):
                                # omit "pretty" formatting to allow more flexibility
                                s2print = '%d  %d  %d  %d  %d  %.2f  %.2f  %d\n' % (
                                    line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7]
                                )
                                f.write(s2print)
                        elif line[4] == '2':  # improper dihedral
                            line = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]),
                                    float(line[5]), float(line[6])]
                            # filter indices for the first chain only:
                            if not (line[0] > single_chain_mods[0] or line[1] > single_chain_mods[0]
                                    or line[2] > single_chain_mods[0] or line[3] > single_chain_mods[0]):
                                s2print = '%d  %d  %d  %d  %d  %.2f  %.2f\n' % (
                                    line[0], line[1], line[2], line[3], line[4], line[5], line[6]
                                )
                                f.write(s2print)
                        else:
                            continue  # something went wrong with formatting, skip this line

                    else:
                        f.write(line)
            else:  # catch-all for the rest of the text in itp
                for line in section:
                    f.write(line)
    # if there are missing sections, include them at the end:
        if missing_headers:
            if 'virtual_sitesn' in missing_headers[0]:
                f.write('\n' + missing_headers[0])
                f.write('#include "' + file_pref + '_' + fnames[3] + '"\n')
                missing_headers.pop(0)
            elif 'exclusions' in missing_headers[0]:
                f.write('\n' + missing_headers[0])
                f.write('#include "' + file_pref + '_' + fnames[4] + '"\n')
                missing_headers.pop(0)
            # check if anything left in missing_headers:
            if missing_headers: # if yes, then it must be excusions
                f.write('\n' + missing_headers[0])
                f.write('#include "' + file_pref + '_' + fnames[4] + '"\n')

    # write updated .top file from scratch:
    with open(file_pref+ '_' + fnames[7], 'w') as f:
        f.write('#define GO_VIRT\n')
        f.write('#include "martini_v3.0.0_go.itp"\n')
        f.write('#include "' + file_pref + '_' + fnames[6] + '"\n\n')
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
out_pdb, single_chain_mods = assign_chain_ids(system_pdb_data, args.bb_cutoff, pdb_chain_ids, args.chain_sort, args.chain_file)
# group sym_pairs into intra and inter based on their chain IDs
sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter = sym_pair_sort(sym_pairs, out_pdb, single_chain_mods)
# retrieve sigma-epsilon values for each BB involved in intra-Go bonds (for D virtual sites)
sigma_d, eps_d = get_bb_pair_sigma_epsilon(args.i, args.nb, sym_pairs_inter)


# write the updated pdb file (VS A-D):
vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb = update_pdb(args.moltype, out_pdb, resnr_intra, resnr_inter)
excl_b, excl_c, excl_d, intra_pairs, inter_pairs = get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra,
                                                                  sym_pairs_inter)
# write the updated itp/top files:
write_include_files(args.moltype, args.missres, args.go_eps_intra,
                    args.go_eps_inter, c6c12, sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs,
                    inter_pairs, virtual_sites, upd_out_pdb, fnames, sigma_d, eps_d)
write_main_top_files(args.moltype, args.i, fnames)
