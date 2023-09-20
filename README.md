# [temp name]: sorted_goVirt
## Description:
This script is the generalization of the Martini-Go scheme (originally developed by Poma et al., 2017) for multichain CG protein complexes.

UPDATED (230919) features include:
 - *inter*molecular and *intra*molecular Go-bonds are clearly distinguished, with two separate $\varepsilon$-values defined by the user
 - separation of Go-bonds is realized via the combination of multiple virtual site (VS) and exclusion groups
 - flexible chain identification procedure
 - NEW: single chain output for homopolymers (make sure to use `modulo-ft` branch)

## Repository content:
- `sorted_goVirt.py`: the script
- `martini_v3.0.0_go.itp`: modified martini force field to be used with the script
- `docs/`: directory containing documentation (to be updated soon)
- `test_sytems/`: UPDATED directory with new example test system for a homopolymer (2BEG)

## How it works:
- the script requires no installation: simply change file permissions to make it executable (e.g.: `chmod u+x`)
- required Python packages:
  - argparse
  - numpy
  - math
  - re
  - pandas
  - itertools
- CURRENT example commands:
  - chain identification from the provided .pdb file (`--chain_sort 1`):
  ```commandline
  python3 ./sorted_goVirt.py -s 2beg_m1_cg.pdb -i molecule_0.itp -f 2beg_m1_clean_renumbered.map --nb martini_v3.0.0_go.itp --moltype 2beg --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 1 --bb_cutoff 10
  ```
- OLD version examples:
  - Distance-based chain identification:
  ```commandline
  python3 ./sorted_goVirt.py -s mol_cg.pdb -i mol_cg.itp -f mol.map --nb martini_v3.0.0_go.itp --moltype mol --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 0 --bb_cutoff 10
  ```
  - User-defined chain identification:
  ```commandline
  python3 ./sorted_goVirt.py -s mol_cg.pdb -i mol_cg.itp -f mol.map --nb martini_v3.0.0_go.itp --moltype mol --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 2 --chain_file mol_chain_ids.txt
  ```

### Input files:
- `-s` structure pdb file (output of `martinize2` script):
  - when running `martinize2`, do NOT use `-govs-include` and `-govs-moltype` flags
  - maximal number of CG particles (ATOM records in the input pdb file): 99999
- `-i` itp file (output of `martinize2` script), contains the structure's topology
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

All auxiliary itp file names follow the same convention: prefix (defined via `--moltype`) + itp section name + `go_mono` (OLD ver.: `go`)

(to be updated) For the organization of the output files, refer to this [flowchart](https://github.com/kkorshunova/multichain-martini-go/blob/master/docs/MGo_top_A1.pdf).

### How to choose a suitable chain sorting method:
Identification of separate chains in the input protein structure can be done in three ways:
 1. (risky) if structure isn't missing residues: automatic, distance-based detection
 2. if input pdb has correct chain-ID records: read from pdb
 3. catch-all option: user-defined chain-ID supplied in a separate, single-column plain text file (one entry per line, same order as in the input structure pdb, N(lines)=N(ATOM records)) 

### To-do list:
- [ ] make the choice of the analyzed single chain user-controlled (currently: 1st chain by default)
- [ ] sym_pair_sort(): improve and test INTER pair sorting
- [ ] get_bb_pair_sigma_epsilon(): improve analysis and filtering of the martini.itp
- [ ] (Opt.) better Go-bond visualization