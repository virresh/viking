# VIKING

This repository has two main components:

1. code - This folder contains the code for generating VIKING and VIKING^s perturbations. Additionally it also contains sample implementations of baselines and evaluation methods for node classification and link prediction tasks.

2. data - This folder contains the three datasets used for reporting the given results.

### Requirements for running the code

Make sure you have installed node2vec and LINE binaries and they are avalaible on path.
(i.e `node2vec` and `line` should not print error on terminal).
If you face errors while running line, make sure you have all required c-libraries on path.
Also ensure `pytorch` and `dgl` library is installed.


### Running instructions
From the root directory of this repository, call main.py as follows:
```bash
python code/simple_all.py cora lp
```

Since the results are quite long, it's ideal to save output to a file and later on tail the file for required output.
