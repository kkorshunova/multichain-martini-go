# "Snake Oil"
## Description:
This script is a variation of the Martini-Go scheme (originally developed by Poma et al., 2017) for multichain CG protein homopolymers.

Features include:
 - *inter*molecular and *intra*molecular Go-bonds are clearly distinguished, with two separate $\varepsilon$-values defined by the user
 - separation of Go-bonds is realized via the combination of multiple virtual site (VS) and exclusion groups
 - flexible chain identification procedure
 - single chain output for homopolymers

## Repository content:
- `sorted_goVirt.py`: the script
- `martini_v3.0.0_go.itp`: modified martini force field to be used with the script
- `docs/`: directory containing documentation (to be updated soon)
- `test_sytems/`: UPDATED directory with new example test system for a homopolymer (2BEG)

## How it works:
- the script requires no installation: simply change file permissions to make it executable (e.g.: `chmod u+x`)
- required additional Python packages:
  - numpy
  - pandas
- Example commands:
  - chain identification from the provided .pdb file (`--chain_sort 1`):
  ```commandline
  python3 ./sorted_goVirt.py -s 2beg_m1_cg.pdb -i molecule_0.itp -f 2beg_m1_clean_renumbered.map --nb martini_v3.0.0_go.itp --moltype 2beg --go_eps_inter 6.0 --go_eps_intra 12.0 --chain_sort 1 --bb_cutoff 10 --chain_id B
  ```

### Input files:
- `-s` structure pdb file (output of `martinize2` script):
  - when running `martinize2`, do NOT use `-govs-include` and `-govs-moltype` flags. Example `martinize2` command:
  ```commandline
  martinize2 -f 2beg_m1_clean_renumbered_conected.pdb -o 2beg_m1_cg.top -x 2beg_m1_cg.pdb -dssp /usr/bin/dssp -p backbone -ff martini3001 -merge A,B,C,D,E -scfix -cys auto
  ```
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
- `--chain_id B` select the output chain for which the contacts will be analyzed

### Output files:
Prefix is defined via `--moltype` flag (`mol` by default)
- prefix + `_go_mono.top`: main system topology file 
- prefix + `_go_mono.itp`: main include topology file
- prefix + itp section name + `_go_mono.itp`: all auxiliary itp file names

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