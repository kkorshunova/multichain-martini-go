#!/usr/bin/env python

import argparse
import subprocess
import numpy as np
import math
import re

import pandas as pd  # needed only for the get_bb_pair_sigma_epsilon() function...

##################### FUNCTIONS #####################
def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='File containing the coarse-grained structure of the protein in pdb format.')
    parser.add_argument('-f', help='File containing the contact analysis of the (atomistic) protein structure obtained from the webserver http://info.ifpan.edu.pl/~rcsu/rcsu/index.html.')
    parser.add_argument('-i', help='File containing the topology of coarse-grained protein in itp format.')
    parser.add_argument('--moltype', default='molecule_0',
                        help='Molecule name used as prefix in your output file names and the virtual bead names (default: molecule_0). If you will combine your Go-like model with a coarse-grained protein generated with martinize2, you must use the same name as specified with the --govs-moltype flag of martinize2!')
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


# get_settings() initializes temporary file names and some vars
def get_settings():
    # names for temporary files: - todo: delete after end of script run
    file_OV = 'OV.map'
    file_rCSU = 'rCSU.map'

    # some other rudimentary variables
    header_lines = 0
    seqDist = 4         # minimal distance in the sequence to add a elastic bond (ElNedyn=3 [Perriole2009]; Go=4 [Poma2017])
    cols = [5, 9, 10]   # colums of interest in the OV and rCSU contact map file
    missAt = 0          # todo: is this used? number of missing atoms at the beginning of pdb structure
                        # (this has to result in the correct atom number when added to "k_at" compared to the .itp file)
    c6c12 = 0           # if set to 1, the C6 and C12 term are expected in the .itp file; if set to 0, sigma and go_eps are used
    return file_OV, file_rCSU, header_lines, seqDist, cols, missAt, c6c12


# read_data() parses data from the .map file (output of the rCSU server), stores it in temporary files
# returns lists: indBB (list of xyz coords of BB), nameAA (resname list), map_OVrCSU (list: resID resID distance...)
# todo: get rid of subprocess calling if possible
def read_data(cg_pdb, file_contacts, file_OV, file_rCSU, header_lines, cols):
    # preparation of temporary files for reading
    subprocess.call("grep '1 [01] [01] [01]' " + file_contacts + " > " + file_OV, shell=True)
    subprocess.call("echo '' >> " + file_OV, shell=True)
    subprocess.call("grep '0 [01] [01] 1' " + file_contacts + " > " + file_rCSU, shell=True)
    subprocess.call("echo '' >> " + file_rCSU, shell=True)

    # read the pdb file
    pdb_data = [ ]
    indBB = [ ]  # separate from pdb_data[] because indBB needs to be a numpy array
    with open(cg_pdb, 'r') as file:
        # create a 2d array with all relevant data: pdb_data columns 1-3,5-8
        #    here: omitted chain_id (column 4)
        for line in file:
            line = line.split()
            if line[0] == 'ATOM' and line[2] != 'CA':  # double check we are using only the relevant pdb records
                # later: more flexible ways to filter out unnecessary VSs written by martinize?
                # save only relevant columns in pdb_data + add column for chain id:
                pdb_data.append(
                    [int(line[1]), line[2], line[3], int(line[5]), float(line[6]), float(line[7]), float(line[8]), 0])
                #      atomnr     atomname   resname      resnr      x         y         z       chain_id
                #  e.g.: [1, 'BB', 'GLY', 1, -26.214, 5.188, -11.96, 0]
                if line[2] == 'BB':
                    indBB.append([int(line[1]), float(line[6]), float(line[7]), float(line[8])])
                    # indBB:           atomnr           x              y              z
            else:
                continue  # skips irrelevant lines (e.g. CONECT if it's present in file)
    indBB = np.array(indBB)

    # read OV contact map
    with open(file_OV,'r') as fid:
        dat = fid.readlines()
    dat = dat[header_lines:-1]
    #print('Number of contacts read from your OV contact map file: ' + str(len(dat)))

    map_OVrCSU = []
    row = []
    for k in range(0, len(dat)):
        tmp = dat[k]
        tmp = tmp.replace('\t',' ')
        tmp = tmp.split()
        for l in cols:
            row.append(float(tmp[l]))
        map_OVrCSU.append(row)
        row = []

    # read rCSU contact map
    with open(file_rCSU,'r') as fid:
        dat = fid.readlines()
    dat = dat[header_lines:-1]
    #print('Number of contacts read from your rCSU contact map file: ' + str(len(dat)))

    for k in range(0, len(dat)):
        tmp = dat[k]
        tmp = tmp.replace('\t',' ')
        tmp = tmp.split()
        for l in cols:
            row.append(float(tmp[l]))
        map_OVrCSU.append(row)
        row = []

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
        if (map_OVrCSU[k][2] > cutoff_short) and (map_OVrCSU[k][2] < cutoff_long) and ( abs(map_OVrCSU[k][1]-map_OVrCSU[k][0]) >= seqDist ):
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

    #for line in sym_pairs:
    #    print(line)
    #print(len(sym_pairs))
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
    #print(len(sym_pairs_intra), len(sym_pairs_inter))
    #for line in sym_pairs_intra:
    #    print(line)
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
    #for line in upd_out_pdb:
    #    print(line)

    # write an updated pdb file:
    with open('updated_' + file_pref + '_cg.pdb', 'w') as f:
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
    return excl_b, excl_c, excl_d, intra_pairs, inter_pairs


def get_bb_pair_sigma_epsilon(itp_filename, martini_file, sym_pairs_inter, missAt):
    temp = []
    output_raw = []
    atoms_section = []

    # extract "atom index - atomtype" information from the [ atoms ] section of molecule.itp
    with open(itp_filename, 'r') as file:
        match = False

        for line in file:
            if re.search(r'\[ atoms ]', line):
                match = True
                # print(line)
                continue
            elif re.search(r'\[ position_restraints ]', line):
                match = False
                # print(line)
                continue
            elif match:
                output_raw.append(line.split())
    # clean up the raw output
    for entry in output_raw:
        if entry:
            temp.append([int(entry[0]), entry[1], int(entry[2]), entry[3], entry[4]])
            #                       atom index    atomtype      resnr       resname   atomname
    for entry in temp:
        if entry[4] == 'BB':
            atoms_section.append(entry)
    # create a dictionary: key: atomindex; value: atomtype
    atoms_section = np.array(atoms_section)
    atomtype_dict = dict(zip(atoms_section[:,0], atoms_section[:,1])) # {'1': 'Q5', '2': 'P2', '4': 'SP2', ...}
    #print(atomtype_dict)

    # create a list of atomtype pairs to match the inter Go bonds:
    bb_index_pairs = [ ]
    for bbpair in range(len(sym_pairs_inter)):
        bb_index_pairs.append([str(sym_pairs_inter[bbpair][0] + missAt), str(sym_pairs_inter[bbpair][1] + missAt)])
        # elements of bb_index_pairs list must be strings for dictionary to work below

    # replace indices (keys in dict) with atomtypes (vals in dict) using list comprehension:
    bb_atomtype_pairs = [ ]
    for pair in bb_index_pairs:
        replaced_pair = [x if x not in atomtype_dict else atomtype_dict[x] for x in pair]
        bb_atomtype_pairs.append(replaced_pair)
        #print(pair)
        #print(replaced_pair)
    # temp: list of unique atomtypes found in pairs (for filtering the huge martini list below):
    # bb_atomtypes = list(np.concatenate(bb_atomtype_pairs).flat)
    # bb_atomtypes = list(set(bb_atomtypes)) # removed duplicates, unique entries: ['SP2a', 'SP1', 'P2', 'Q5', 'SP2']

    # merge pairs of atomtypes into a single string to use as a matching key in dataframe dictionary:
    bb_atomtype_pairs_merged = []
    for bbpair in bb_atomtype_pairs:
        k = bbpair[0] + ' ' + bbpair[1]
        bb_atomtype_pairs_merged.append(k)
    # bb_atomtype_pairs_merged=['Q5 Q5', 'P2 P2', 'P2 P2',...]

    # find the sigma-epsilon values for the bb_atomtype_pairs in the database (martini_v3.0.0.itp - [ nonbond_params])
    # open martini_v3 file:
    martini_nonbond = [ ]
    with open(martini_file, 'r') as f:
        match = False
        for line in f:
            if re.search(r'\[ nonbond_params ]', line):
                match = True
                continue
            elif len(line.split()) < 5 or line == '\n': # end of [ nonbond_params ]: empty line or other N(columns)
                match = False
                continue
            elif match: # for lines inside the [ nonbond_params ] section: matching and writing?
                martini_nonbond.append(line.split())  # contains the entire nonbond section of martini
                # alternatively: filter - if column 0 contains any of the bb_atomtypes:
                #test = [bbtype for bbtype in line.split()[0] if any(bt in bbtype for bt in bb_atomtypes)]
                #print(test)

    #### todo: this entire section needs to be reworked
    # create a dataframe for filtering by 2 columns... :(
    martini_nonbond = np.array(martini_nonbond)
    #bb_atomtype_pairs = np.array(bb_atomtype_pairs)
    #print(martini_nonbond[:,0])
    df1 = pd.DataFrame({ 'BB1' : martini_nonbond[:,0], 'BB2' : martini_nonbond[:,1],
                                'sigmaBB' : martini_nonbond[:,3], 'epsBB' : martini_nonbond[:,4] })
    df1 = df1[df1.set_index(['BB1','BB2']).index.isin(bb_atomtype_pairs)]
    # dataframe df1 contains a list of _unique_ pairs BB1-BB2 with sigma and eps value pairs

    # concatenate the unique df1 with its BB1-BB2 swapped:
    df2 = df1[['BB2','BB1','sigmaBB', 'epsBB']]
    df = pd.concat([df1, df2.rename(columns={'BB2':'BB1', 'BB1':'BB2'})], ignore_index=True)
    # merge BB1 and BB2 into a single column to use as a dictionary
    df1['BB pairs'] = df1['BB1'].astype(str) + ' ' + df1['BB2'].astype(str)
    df['BB pairs'] = df['BB1'].astype(str) + ' ' + df['BB2'].astype(str)

    # rearrange columns in df so when it is transposed the 'BB pairs' is 1st line
    df1 = df1[['BB pairs', 'sigmaBB', 'epsBB']]
    df = df[['BB pairs', 'sigmaBB', 'epsBB']]
    # create a dictionary out of the df1:
    sig_eps_dict2 = df1.set_index('BB pairs').T.to_dict('list')
    print(sig_eps_dict2)
    sig_eps_dict = df.set_index('BB pairs').T.to_dict('list')
    print(sig_eps_dict)

    # use the dictionary to write a 2d list:
    final_bb_entries = [ ]
    for i in bb_atomtype_pairs_merged:
        temp_line = [x if x not in sig_eps_dict else sig_eps_dict[x] for x in i]
        final_bb_entries.append(temp_line)
        if (i in sig_eps_dict):
            print(i, 'true')
        else:
            print(i, 'false')
    #### todo: end of  section



    return atoms_section, bb_atomtype_pairs


########## FILE WRITING PROCEDURES ##########
# todo: routines same as in old version, just repeated for sym_pars_intra/inter separately
# todo: check WHY does this function use sym_pairs_inter/intra w/o them being input params????
def write_include_files(file_pref, missAt, indBB, missRes, Natoms, go_eps_intra, go_eps_inter, c6c12,
                sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs, inter_pairs,
                        virtual_sites, upd_out_pdb):
    # main.top -> martini_v3.0.0.itp [ nonbonded_params ]-> go-table_VirtGoSites.itp -> $MOL_go-table_VirtGoSites.itp
    with open(file_pref + '_go-table_VirtGoSites.itp', 'w') as f:
        f.write('; OV + symmetric rCSU contacts \n')
        if (c6c12 == 1):  # this setting uses sigma/eps computed using Vii, Wii
            f.write('; not implemented yet\n')
        else:  # default setting, uses sigma + go_eps_*
            f.write('; INTRA section: A-B pairs (+/- go_eps_intra) \n')
            for k in range(0, len(sym_pairs_intra)):
                # VWA-VWA pair: BB go_eps_intra
                s2print = " %s_A%s  %s_A%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n" % (file_pref,
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
                s2print = " %s_B%s  %s_B%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n" % (file_pref,
                                                                                      str(int(sym_pairs_intra[k][4])),
                                                                                      file_pref,
                                                                                      str(int(sym_pairs_intra[k][5])),
                                                                                      sym_pairs_intra[k][7],
                                                                                      -go_eps_intra+0.00001, # avoid exact val
                                                                                      str(int(sym_pairs_intra[k][
                                                                                                  0]) + missAt),
                                                                                      str(int(sym_pairs_intra[k][
                                                                                                  1]) + missAt),
                                                                                      sym_pairs_intra[k][6])
                f.write(s2print)
            f.write('; INTER section: C (go_eps_inter) \n')
            for k in range(0, len(sym_pairs_inter)):
                s2print = " %s_C%s  %s_C%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n" % (file_pref,
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
                # todo: replace sym_pairs_inter[k][7] and go_eps_inter  with BB sigma/eps values
                s2print = " %s_D%s  %s_D%s    1  %.10f  %.10f  ;  %s  %s  %.3f \n" % (file_pref,
                                                                                    str(int(sym_pairs_inter[k][4])),
                                                                                    file_pref,
                                                                                    str(int(sym_pairs_inter[k][5])),
                                                                                    sym_pairs_inter[k][7],
                                                                                    -go_eps_inter+0.00001, # avoid exact val
                                                                                    str(int(sym_pairs_inter[k][0]) +missAt),
                                                                                    str(int(sym_pairs_inter[k][1]) +missAt),
                                                                                    sym_pairs_inter[k][6])
                f.write(s2print)
    # add the name of the created .itp file into the wrapper go-table_VirtGoSites.itp
    with open('go-table_VirtGoSites.itp','w') as f:
        s2print = "#include \"%s_go-table_VirtGoSites.itp\"\n" % file_pref
        f.write(s2print)

    # main.top -> martini_v3.0.0.itp [ atomtypes ]-> BB-part-def_VirtGoSites.itp -> $MOL_BB-part-def_VirtGoSites.itp
    # here: sets of $MOL_[A-D] VSites
    with open(file_pref + '_BB-part-def_VirtGoSites.itp', 'w') as f:
        f.write('; protein BB virtual particles \n')
        f.write('; INTRA particles\n')
        for k in resnr_intra:
            s2print = "%s_A%s 0.0 0.000 A 0.0 0.0 \n" % (
            file_pref, str(k + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        for k in resnr_intra:
            s2print = "%s_B%s 0.0 0.000 A 0.0 0.0 \n" % (
                file_pref, str(k + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        f.write('; INTER particles\n')
        for j in resnr_inter:
            s2print = "%s_C%s 0.0 0.000 A 0.0 0.0 \n" % (
                file_pref, str(j + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
        for j in resnr_inter:
            s2print = "%s_D%s 0.0 0.000 A 0.0 0.0 \n" % (
                file_pref, str(j + missRes))  # residue index adapted due to missing residues
            f.write(s2print)
    # add the name of the created .itp file into the wrapper go-table_VirtGoSites.itp
    with open('BB-part-def_VirtGoSites.itp', 'w') as f:
        s2print = "#include \"%s_BB-part-def_VirtGoSites.itp\"\n" % file_pref
        f.write(s2print)

    # main.top -> molecule.itp [ atoms ] -> $MOL_atoms_VirtGoSites.itp
    with open(file_pref + '_atoms_VirtGoSites.itp', 'w') as f:
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
            s2print = "%3d %s_%s%-3d %6d %3s %-3s %-5d 0.0\n" % (entry[0], file_pref, suffix, entry[3], entry[3],
                                                               entry[2], entry[1], entry[0])
            # atomnr atomtype=(filepref_[A-D]resnr) resnr resname atomname chrggrp=atomnr q=0.0
            f.write(s2print)

    # main.top -> molecule.itp [ virtual_sitesn ] -> $MOL_virtual_sites_VirtGoSites.itp
    with open(file_pref + '_virt_sitesn_VirtGoSites.itp', 'w') as f:
        f.write('; VS index - funct - constructing atom index/indices\n')
        for pair in virtual_sites:
            s2print = "%4d 1 %3d\n" % (pair[0], pair[1])
            f.write(s2print)

    # main.top -> molecule.itp [ exclusions ] -> $MOL_exclusions_VirtGoSites.itp
    # exclusion pairs sorted by intra-inter:
    with open(file_pref + '_exclusions_VirtGoSites.itp', 'w') as f:
        f.write('; [ exclusions for intra BB sites ]\n')
        f.write('; atomnr atomnr  -  resnr resnr\n')
        for line in sym_pairs_intra:
            s2print = " %d  %d  \t ;  %d  %d \n" % (line[0], line[1], line[4], line[5])
            f.write(s2print)
        f.write('; [ exclusions for intra VWB sites ]\n')
        for ind in range(len(excl_b)):
            s2print = " %d  %d  \t ;  %d  %d \n" % (excl_b[ind][0], excl_b[ind][1],
                                                    intra_pairs[ind][0], intra_pairs[ind][1])
            f.write(s2print)
        f.write('; [ exclusions for inter VWC sites ]\n')
        for ind in range(len(excl_c)):
            s2print = " %d  %d  \t ;  %d  %d \n" % (excl_c[ind][0], excl_c[ind][1],
                                                    inter_pairs[ind][0], inter_pairs[ind][1])
            f.write(s2print)
        f.write('; [ exclusions for inter VWD sites ]\n')
        for ind in range(len(excl_d)):
            s2print = " %d  %d  \t ;  %d  %d \n" % (excl_d[ind][0], excl_d[ind][1],
                                                    inter_pairs[ind][0], inter_pairs[ind][1])
            f.write(s2print)

    # visualize the go-bonds:


# modifies the .top and .itp written by martinize2 (w/o -go-vs flag), inserts "#include" lines
def write_main_top_files(file_pref, molecule_itp):
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
    with open('updated_'+molecule_itp, 'w') as f:
        for line in block_1:
            f.write(line)
        f.write('#include "'+file_pref+'_atoms_VirtGoSites.itp"\n\n')  # this can also be a line-for-line list
        for line in block_2:
            f.write(line)
        f.write('\n[ virtual_sitesn ]\n#include "'+file_pref+'_virt_sitesn_VirtGoSites.itp"\n\n')
        for line in block_3:
            f.write(line)
        f.write('#include "'+file_pref+'_exclusions_VirtGoSites.itp"\n')

    # write updated .top file from scratch:
    with open('updated_'+file_pref+'.top', 'w') as f:
        f.write('#define GO_VIRT\n')
        f.write('#include "martini_v3.0.0.itp"\n')
        f.write('#include "updated_'+molecule_itp+'"\n\n')  # todo: check for name consistency after cleanup
        f.write('[ system ]\n'+ file_pref + ' complex with Go\n\n')
        f.write('[ molecules ]\n' + file_pref + '     1 \n') # check if 1 can be a variable

    # later: consistent martinize2 output:
    # with Go-flag: inserts "#include"-lines into the cg top,itps BUT adds not needed lines in pdb and molecule.itp
    # solution -> cut off end of pdb and add own lines
    # without Go-flag: need to alter cg top/itp AND pdb file
    # for now: no Go-flag + pre-created martini_v3.0.0.itp with wrappers included already


##################### MAIN #####################
# parse input arguments, initialize some vars:
args = user_input()
# write temp files, initialize more vars:
file_OV, file_rCSU, header_lines, seqDist, cols, missAt, c6c12 = get_settings()
# read contact map data and store it in lists:
indBB, map_OVrCSU, system_pdb_data = read_data(args.s, args.f, file_OV, file_rCSU, header_lines, cols)
# write symmetric unsorted Go pairs
sym_pairs = get_go(indBB, map_OVrCSU, args.cutoff_short, args.cutoff_long, args.go_eps_intra, seqDist, args.missres)

# sort Go pairs into intra and inter sub-lists:
out_pdb = assign_chain_ids(system_pdb_data, args.bb_cutoff)
# group sym_pairs into intra and inter based on their chain IDs
sym_pairs_intra, sym_pairs_inter, resnr_intra, resnr_inter = sym_pair_sort(sym_pairs, out_pdb)
# todo: this entire function
#atoms_section, bb_atomtype_pairs = get_bb_pair_sigma_epsilon('insulin.itp', 'martini_v3.0.0.itp', sym_pairs_inter, missAt)


# write the updated pdb file (VS A-D):
vwb_excl, vwc_excl, vwd_excl, virtual_sites, upd_out_pdb = update_pdb(args.moltype, out_pdb, resnr_intra, resnr_inter)
excl_b, excl_c, excl_d, intra_pairs, inter_pairs = get_exclusions(vwb_excl, vwc_excl, vwd_excl, sym_pairs_intra,
                                                                  sym_pairs_inter)
# write the updated itp/top files:
write_include_files(args.moltype, missAt, indBB, args.missres, args.Natoms, args.go_eps_intra,
                    args.go_eps_inter, c6c12, sym_pairs_intra, sym_pairs_inter, excl_b, excl_c, excl_d, intra_pairs,
                    inter_pairs, virtual_sites, upd_out_pdb)
write_main_top_files(args.moltype, args.i)
