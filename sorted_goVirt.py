#!/usr/bin/env python

import argparse
import numpy as np
import math
import re

import pandas as pd  # needed for get_bb_pair_sigma_epsilon()

##################### FUNCTIONS #####################
def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='File containing the coarse-grained structure of the protein in pdb format.')
    parser.add_argument('-f', help='File containing the contact analysis of the (atomistic) protein structure obtained from the webserver http://info.ifpan.edu.pl/~rcsu/rcsu/index.html.')
    parser.add_argument('-i', help='File containing the topology of coarse-grained protein in itp format.')
    parser.add_argument('--nb', help='File containing martini_go.ff in itp format.')
    parser.add_argument('--moltype', default='mol',
                        help='Molecule name used as prefix in your output file names and the virtual bead names (default: mol). If you will combine your Go-like model with a coarse-grained protein generated with martinize2, you must use the same name as specified with the --govs-moltype flag of martinize2!')
    parser.add_argument('--go_eps_inter', type=float, default=9.414,
                        help='Dissociation energy [kJ/mol] of the Lennard-Jones potential used in the Go-like model (default: 9.414).')
    parser.add_argument('--go_eps_intra', type=float, default=9.414,
                        help='Dissociation energy [kJ/mol] of the Lennard-Jones potential used in the Go-like model (default: 9.414).')
    parser.add_argument('--cutoff_short', type=float, default=0.3,
                        help='Lower cutoff distance [nm]: contacts with a shorter distance than cutoff_short are not included in the Go-like interactions (default: 0.3).')
    parser.add_argument('--cutoff_long', type=float, default=1.1,
                        help='Upper cutoff distance [nm]: contacts with a longer distance than cutoff_long are not included in the Go-like interactions (default: 1.1).')
    parser.add_argument('--Natoms', type=int,
                        help='Number of coarse-grained beads in the protein excluding the virtual Go beads.')
    parser.add_argument('--missres', type=int, default=0,
                        help='Number of missing residues at the beginning of the atomistic pdb structure which is needed if the numbering of the coarse-grained structure starts at 1 (default: 0).')
    parser.add_argument('--bb_cutoff', type=int, default=10,
                        help='Max distance (in A) allowed between next-neighbor BBs (default: 10 A).')
    args = parser.parse_args()
    return args


# get_settings() initializes "global" vars (names of output files, some numerical values)
def get_settings():
    # some variables
    seqDist = 4         # minimal distance in the sequence to add a elastic bond (ElNedyn=3 [Perriole2009]; Go=4 [Poma2017])
    missAt = 0          # todo: is this used? number of missing atoms at the beginning of pdb structure
                        # (this has to result in the correct atom number when added to "k_at" compared to the .itp file)
    c6c12 = 0           # if set to 1, the C6 and C12 term are expected in the .itp file; if set to 0, sigma and go_eps are used

    # names of the output included itp files:
    fnames = ['atomtypes_go.itp',
              'nonbond_params_go.itp',
              'atoms_go.itp',
              'virtual_sitesn_go.itp',
              'exclusions_go.itp',
              'viz_go.itp']

    return seqDist, missAt, c6c12, fnames


# read_data() parses data from the .map file (output of the rCSU server), stores it in temporary files
# returns lists: indBB (list of xyz coords of BB), map_OVrCSU (list: resID resID distance...), pdb_data
def read_data(cg_pdb, file_contacts):
    # read the pdb file: mind the fixed file format!
    pdb_data = [ ]
    indBB = [ ]  # separate from pdb_data[] because indBB needs to be a numpy array
    with open(cg_pdb, 'r') as file:
        # create a 2d array with all relevant data: pdb_data columns 1-3,5-8
        #    here: omitted chain_id (column 4)
        for line in file:
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

    return indBB, map_OVrCSU, pdb_data


# get_go() calculates and filters the Go pairs according to requirements (AT server map data -> CG structure)
# returns: sym_pairs = [0:indBB, 1:indBB, 2:sigma, 3:eps, 4:resnr, 5:resnr, 6:distance, 7:sigma_const]
#                     0, 1 - exclusions (molecule.itp)   2,3 - ignored   4-7 - go_table (martini.itp)
# todo: get rid of Vii and Wii - they're not used anyway!
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

    return sym_pairs

########## INTRA-INTER SORTING PROCEDURES ##########
# create a new 2d list: pdb_data + last column with chain IDs
# out_pdb can be used to analyze and sort inter/intra bonds by checking the resnr + chain ID combo of the VSite
def assign_chain_ids(pdb_data, bb_cutoff):
    # set the cutoff distance
    max_dist = pow(bb_cutoff, 2)  # (squared, A) distance between two consecutive residues in backbone
    ######## find chain "heads"
    # compute squared distances between sequential BBs
    # input: system_pdb_data (2d list), output: new_chain_begins (1d list)
    system_BB_only = [ ]
    for line in pdb_data:
        if line[1] == 'BB':  # select only BB lines:
            system_BB_only.append(line)
    new_chain_begins = [ ]  # list of atomid's which begin a new chain
    # calculate squared distances and create an index list of chain beginnings:
    for index, line in enumerate(system_BB_only):
        if index+1 < len(system_BB_only):  # if-clause prevents going out of range for the last BB pair
            # calculate square of distance between all sequential BBs
            dist = math.pow((system_BB_only[index][4] - system_BB_only[index+1][4]), 2) \
                   + math.pow((system_BB_only[index][5] - system_BB_only[index+1][5]), 2) \
                   + math.pow((system_BB_only[index][6] - system_BB_only[index+1][6]), 2)
            if dist > max_dist:
                new_chain_begins.append(system_BB_only[index+1][0])
    # new_chain_begins[ ] contains atom indices of chains 2 to n, 1st chain by default starts with 0
    print(new_chain_begins)

    ######## assign the IDs based on the indices of chain "heads"
    chain_flag = 0  # this variable will change as script progresses down the list of residues
    system_pdb_data_out = [ ]
    current_switch = new_chain_begins.pop(0)
    for line in system_pdb_data:
        if line[0] == current_switch:
            chain_flag += 1
            if new_chain_begins:  # avoid popping an empty list: if only 1 element in list, this step is omitted
                current_switch = new_chain_begins.pop(0)
        line[-1] = chain_flag
        system_pdb_data_out.append(line)
    return system_pdb_data_out


# sym_pair_sort() separates sym_pairs into sym_pairs_intra and sym_pairs_inter based on the output of out_pdb
# function input parameters: sym_pairs, out_pdb
# function output: sym_pairs_intra, sym_pairs_inter
# Q: separate intras further chain-wise - needed or not?
# different epsilon entries: taken from the script input (--go_eps_intra, --go_eps_inter)
def sym_pair_sort(sym_pairs, out_pdb):
    sym_pairs_intra = [ ]
    sym_pairs_inter = [ ]
    for index, pair in enumerate(sym_pairs):
        index_i = sym_pairs[index][0]
        index_j = sym_pairs[index][1]
        if out_pdb[index_i][-1] == out_pdb[index_j][-1]:
            sym_pairs_intra.append(pair)
        else:
            sym_pairs_inter.append(pair)
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
    # print('INTRA pairs')
    # for line in sym_pairs_intra:
    #     print(line)
    # print('INTER pairs')
    # for line in sym_pairs_inter:
    #     print(line)
    return sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter


# new pdb file with all relevant particles: CG structure, VS[A-D]
# input: system_pdb_data
# output: updated pdb file (implicitly), exclusions, virtual sites mapping
# todo: better name for this function
def update_pdb(file_pref, out_pdb, resnr_intra, resnr_inter):
    # starting atomnr: last index in out_pdb+1:
    atomnr = out_pdb[-1][0]
    upd_out_pdb = out_pdb.copy()
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
    #print(resnr_intra)
    #for line in vwb_excl:
    #    print(line)

    # write an updated pdb file:
    with open(file_pref + '_cg_go.pdb', 'w') as f:
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


def get_bb_pair_sigma_epsilon(itp_filename, martini_file, sym_pairs_inter, missAt):

    # 1. extract "atom index - atomtype" information from the [ atoms ] section of molecule.itp
    atoms_section = []
    with open(itp_filename, 'r') as file:
        match = False  # logical switch allows to read only lines between (excluding) the two matched lines
        for line in file:
            if re.search(r'\[ atoms ]', line):
                match = True
                continue
            elif re.search(r'\[ position_restraints ]', line): # or line == '\n'
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
    #9438           P2         P2  4.700000e-01  4.060000e+00
    #9698           P2        SP2  4.300000e-01  3.770000e+00

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
    #0    P6 P6  4.700000e-01  4.990000e+00
    #1    P6 P5  4.700000e-01  4.730000e+00

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
def write_include_files(file_pref, missAt, indBB, missRes, Natoms, go_eps_intra, go_eps_inter, c6c12,
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
    with open(fnames[0], 'w') as f:
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
                                                                                      str(int(sym_pairs_intra[k][0]) +missAt),
                                                                                      str(int(sym_pairs_intra[k][1]) +missAt),
                                                                                      sym_pairs_intra[k][6])
                f.write(s2print)
                # VWB-VWB pair: BB -go_eps_intra
                s2print = ' %s_B%s  %s_B%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n' % (file_pref,
                                                                                      str(int(sym_pairs_intra[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_intra[k][5])),
                                                                                      sym_pairs_intra[k][7],
                                                                                      -go_eps_intra + 0.00001,  # avoid exact val
                                                                                      str(int(sym_pairs_intra[k][
                                                                                                  0]) + missAt),
                                                                                      str(int(sym_pairs_intra[k][
                                                                                                  1]) + missAt),
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
                                                                                      str(int(sym_pairs_inter[k][0]) +missAt),
                                                                                      str(int(sym_pairs_inter[k][1]) +missAt),
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
                                                                                      str(int(sym_pairs_inter[k][0]) +missAt),
                                                                                      str(int(sym_pairs_inter[k][1]) +missAt),
                                                                                      sym_pairs_inter[k][6])
                f.write(s2print)
    # add the name of the created .itp file into the wrapper go-table_VirtGoSites.itp
    with open(fnames[1],'w') as f:  # this itp name file was written by martinize2
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
    # store the entire molecule.itp as a list (1d, 1 element = 1 line as str)
    input_itp = [ ]
    with open(molecule_itp, 'r') as f:
        for line in f:
            input_itp.append(line)
    # separate the itp file into blocks based on where the additional lines need to go
    # here, 3 additions must be made: 1. end of [ atoms ] (before [ posres ]), 2. before [ angles ], 3. at the EOF
    block_1 = input_itp[:input_itp.index('[ position_restraints ]\n')]
    if block_1[-1]=='\n':  # the newline will be re-added later
        del block_1[-1]
    block_2 = input_itp[input_itp.index('[ position_restraints ]\n'):input_itp.index('[ angles ]\n')]
    if block_2[-1]=='\n':  # the newline will be re-added later
        del block_2[-1]
    block_3 = input_itp[input_itp.index('[ angles ]\n'):]
    if block_3[-1]=='\n':  # the newline will be re-added later
        del block_3[-1]

    # insert needed lines between blocks and save the new itp file:
    with open(file_pref + '_go.itp', 'w') as f:
        for line in block_1:
            f.write(line)
        f.write('#include "' + file_pref + '_' + fnames[2] + '"\n\n')  # [ atoms ]
        for line in block_2:
            f.write(line)
        #todo: check if .itp file already has the virtual_sitesn secion!
        f.write('\n[ virtual_sitesn ]\n#include "' + file_pref + '_' + fnames[3] + '"\n\n')  # [ virtual_sitesn ]
        for line in block_3:
            f.write(line)
        f.write('#include "' + file_pref + '_' + fnames[4] + '"\n')  # [ exclusions ]

    # write updated .top file from scratch:
    with open(file_pref+'_go.top', 'w') as f:
        f.write('#define GO_VIRT\n')
        f.write('#include "martini_v3.0.0_go.itp"\n')
        f.write('#include "' + file_pref + '_go.itp"\n\n')
        f.write('[ system ]\n'+ file_pref + ' complex with Go bonds\n\n')
        f.write('[ molecules ]\n' + file_pref + '     1 \n') # check if 1 can be a variable



##################### MAIN #####################
# parse input arguments, initialize some vars:
args = user_input()
# write temp files, initialize more vars:
seqDist, missAt, c6c12, fnames = get_settings()
# read contact map data and store it in lists:
indBB, map_OVrCSU, system_pdb_data = read_data(args.s, args.f)
# write symmetric unsorted Go pairs
sym_pairs = get_go(indBB, map_OVrCSU, args.cutoff_short, args.cutoff_long, args.go_eps_intra, seqDist, args.missres)

# sort Go pairs into intra and inter sub-lists:
out_pdb = assign_chain_ids(system_pdb_data, args.bb_cutoff)
# group sym_pairs into intra and inter based on their chain IDs
sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter = sym_pair_sort(sym_pairs, out_pdb)
# retrieve sigma-epsilon values for each BB involved in intra-Go bonds (for D virtual sites)
sigma_d, eps_d = get_bb_pair_sigma_epsilon(args.i, args.nb, sym_pairs_inter, missAt)


# write the updated pdb file (VS A-D):
vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb = update_pdb(args.moltype, out_pdb, resnr_intra, resnr_inter)
excl_b, excl_c, excl_d, intra_pairs, inter_pairs = get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra,
                                                                  sym_pairs_inter)
# write the updated itp/top files:
write_include_files(args.moltype, missAt, indBB, args.missres, args.Natoms, args.go_eps_intra,
                    args.go_eps_inter, c6c12, sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs,
                    inter_pairs, virtual_sites, upd_out_pdb, fnames, sigma_d, eps_d)
write_main_top_files(args.moltype, args.i, fnames)
