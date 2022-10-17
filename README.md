# [temp name]: sorted_goVirt
## Description:
This script is the generalization of the Martini-Go scheme (originally developed by Poma et al., 2017) for multichain CG protein complexes.

New features include:
 - *inter*molecular and *intra*molecular Go-bonds are clearly distinguished, with two separate $\varepsilon$-values defined by the user
 - separation of Go-bonds is realized via the combination of multiple virtual site (VS) and exclusion groups
 - flexible chain-identification procedure

## Repository content:
- `sorted_goVirt.py`: the script
- `martini_v3.0.0_go.itp`: modified martini force field to be used with the script
- `docs/`: directory containing the flowchart of the "sorted_goVirt" script (to be updated)
- `test_sytems/`: directory containing example test system files (to be updated)

## How it works:
- the script requires no installation, simply change file permissions (e.g.: `chmod u+x`)
- required Python packages:
  - argparse
  - numpy
  - math
  - re
  - pandas
  - itertools
- example commands:
  - Distance-based chain identification:
  ```
  python3 ./sorted_goVirt.py -s mol_cg.pdb -i mol_cg.itp -f mol.map --nb martini_v3.0.0_go.itp --moltype mol --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 0 --bb_cutoff 10
  ```
  - User-defined chain identification:
  ```
  python3 ./sorted_goVirt.py -s mol_cg.pdb -i mol_cg.itp -f mol.map --nb martini_v3.0.0_go.itp --moltype mol --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 2 --chain_file mol_chain_ids.txt
  ```

### Input files:
- `-s` structure pdb file (output of `martinize2`):
  - when running `martinize2`, do NOT use `-govs-include` and `-govs-moltype` flags
  - maximal number of CG particles (ATOM records in the input pdb file): 99999
- `-i` itp file (output of `martinize2`), contains the structure's topology
- `-f` contact map file: from [this server](http://info.ifpan.edu.pl/~rcsu/rcsu/index.html)
- `--nb` martini force field itp file
- (Opt.) `--chain_file` plain text file containing a single column of chain identifiers for the input pdb file (1 line = 1 ATOM record in the pdb, same order), used with  `--chain_sort 2`
### Other option flags:
- `--moltype` string used as prefix in atomtypes of virtual sites, as well as in output file names (default: 'mol')
- `--go_eps_inter`, `--go_eps_intra` $\varepsilon$-values [kJ/mol] of the *inter*molecular and *intra*molecular Go-bonds, respectively (default: 9.414)
- `--chain_sort` chain sorting method: 0 = distance-based (default), 1 = pdb chain-ID based, 2 = user input based
- (Opt.) `--bb_cutoff` maximal distance (in A) allowed between next-neighbor backbone CG particles (default: 10 A), used with  `--chain_sort 0`
- (\*Opt.) `--cutoff_short`, `--cutoff_long` lower and upper cutoff distances [nm], only contacts within the cutoff limits are included in the Go interactions (current defaults: 0.3 and 1.1)
- (\*Opt.) `--missres` number of missing residues at the beginning of the atomistic pdb structure (legacy variable, default: 0)
### Output files:
- `mol_go.top` main system topology file 
- `mol_go.itp` main include topology file
All auxiliary itp file names follow the same convention: prefix (defined in `--moltype`) + itp section name + `go`
- `mol_atomtypes_go.itp`, `atomtypes_go.itp` list of VS atomtype definitions + wrapper itp file
- `mol_nonbond_params_go.itp`, `nonbond_params_go.itp` list of VS nonbonded interaction parameters + wrapper itp file
- `mol_atoms_go.itp`
- `mol_virtual_sitesn_go.itp`
- `mol_exclusions_go.itp`
- \* `mol_viz_go.itp` (for visualization purposes only, not used in the main topology)
### How to choose a suitable chain sorting method:
 - identification of separate chains in the input protein structure can be done in three ways:
   1. if structure isn't missing residues: automatic, distance-based detection
   2. if input pdb has correct chain-ID records: read from pdb
   3. catch-all option: user-defined chain-ID supplied in a separate, single-column plain text file (one entry per line, same order as in the input structure pdb, N(lines)=N(ATOM records)) 

