# Graph Edit Networks - Reproducibility Challenge

## Introduction
This repository contains all the code for the purpose of the Reproducibility challenge 2021/2022.

Paper that was reproduced is 

* Paaßen, B., Grattarola, D., Zambon, D., Alippi, C., and Hammer, B. (2021).
  Graph Edit Networks. Proceedings of the Ninth International Conference on
  Learning Representations (ICLR 2021). [Link][Paa2021]


Copyright (C) 2020-2021  
Benjamin Paaßen  
The University of Sydney  
Daniele Grattarola, Daniele Zambon  
Università della Svizzera italiana

Most of the code is forked from https://gitlab.com/bpaassen/graph-edit-networks, SHA = 588dffec.

## Installation Instructions

All software enclosed in this repository is written in Python 3. To train
graph edit networks, you additionally require [PyTorch][pytorch]
(Version >= 1.4.0; torchvision or cuda are not required). To train tree edit
networks, you additionally require [edist][edist] (Version >= 1.1.0), which
in turn requires [numpy][numpy] (Version >= 1.17).

To run the kernel time series prediction ([Paaßen et al., 2018][Paa2018])
baseline, you require [edist][edist] (Version >= 1.1.0) and
[scikit-learn][sklearn] (Version >= 0.21.3).

These packages are also sufficient to run the experiments.
All dependencies are available on pip.


## File structure
File structure is as follows:
* `env` folder for the environment setup.
* The code for each individual class of data is in it's separated folder. Each folder also contains the test file
for the code in the current directory.
    - `aditional_baselines`,`boolean_formulae`, `degree_rules`, `game_of_life`, `graph_edit`, `hep_th`, `peano_addition`, `pytorch_graph_edit_networks`,
`pytorch_tree_edit_networks`.
* `train_eval_notebooks`, where you can find 4 notebooks for training and evaluationg the models that were used for the paper.
* `visualisation` folder, where is the script that generates the visualisations used in the paper.

## Reproducing the experiments

Reproducing our experiments is possible by running the four included ipython
notebooks in `train_eval_notebooks` folder.  All notebooks should run without any additional preparation.
Installing the dependencies listed above should suffice. Note that slight
deviations may occur due to different sampling.
In the remainder of this section, we list the different experiments and their
notebooks with the expected results in each case.

### Graph dynamical systems

`graph_dynamical_systems.ipynb` contains the experiments on the three graph
dynamical systems (edit cycles, degree rules, and game of life). 

This file also contains the experiments that are described in 4.2.1 and 4.2.2.

### Tree dynamical systems

`pytorch_ten.ipynb` runs a tree edit network on the two tree dynamical systems
(boolean and peano addition).

This file also contains the experiments that are described in 4.2.3.


### Runtimes

`hep_th_runtimes.ipynb` runs graph edit networks on realistic graphs from the
HEP-Th dataset and reports the runtime needed. Then, we compute a log-log fit
of runtime versus graph size. 

This file also contains the experiments that are described in 4.2.4.


## Contents

In more detail, the following files are contained in this repository (in
alphabetical order):

* `baseline_models.py` : An implementation of [variational graph autoencoders (VGAE; Kipf and Welling, 2016)][Kipf2016]
  for time series prediction.
* `boolean_formulae.py` : A Python script generating the Boolean dataset and
  its teaching protocol.
* `boolean_formulae_test.py` : A unit test for `boolean_formulae.py`.
* `degree_rules.py` : A Python script generating the degree rules dataset and
  its teaching protocol.
* `degree_rules_test.py` : A unit test for `degree_rules.py`.
* `game_of_life.py` : A Python script generating the game of life dataset and
  its teaching protocol.
* `game_of_life_test.py` : A unit test for `game_of_life.py`.
* `graph_dynamical_systems.ipynb` : An ipython notebook containing the graph
  edit cycles, degree rules, and game of life experiments.
* `graph_edit_cycles.py` : A Python script generating the graph edit cycles
  dataset and its teaching protocol.
* `graph_edits.py` : A Python implementation of graph edits.
* `hep-th` : A directory containing the HEP-Th dataset as used in this paper,
  including the preprocessing script used.
* `hep_th.py` : A Python script to load the HEP-Th dataset and its teaching
  protocol.
* `hep_th_runtimes.csv` : A table of runtime results obtained on the HEP-Th
  dataset.
* `hep_th_runtimes.ipynb` : An ipython notebook to generate the runtime
  results.
* `hep_th_runtimes.png` : An image file displaying the runtime results obtained
  on the HEP-Th dataset.
* `hep_th_test.py` : A unit test for `hep_th.py`.
* `peano_addition.py` : A Python script generating the Peano dataset and
  its teaching protocol.
* `peano_addition_test.py` : A unit test for `peano_addition.py`.
* `pygraphviz_interface.py` : An auxiliary file to draw graphs using graphviz.
* `pytorch_graph_edit_networks.py` : An implementation of graph edit networks
  and the according loss function as reported in the paper.
* `pytorch_graph_edit_networks_test.py` : A unit test for
  `pytorch_graph_edit_networks.py`.
* `pytorch_ten.ipynb` : An ipython notebook containing the Boolean and the
  Peano experiments.
* `pytorch_tree_edit_networks.py` : An implementation of tree edit networks
  including a general-purpose teaching protocol (with the caveats described
  above).
* `pytorch_tree_edit_networks_test.py` : A unit test for
  `pytorch_tree_edit_networks.py`.
* `README.md` : this file.
* `visualisation-R_script.R` : Visualisation script.



[edist]:https://gitlab.ub.uni-bielefeld.de/bpaassen/python-edit-distances "edist homepage."
[numpy]:https://numpy.org/ "numpy homepage."
[pytorch]:https://pytorch.org/ "PyTorch homepage."
[sklearn]:https://scikit-learn.org/stable/ "scikit-learn homepage"
[Bou2017]:https://bougleux.users.greyc.fr/articles/ged-prl.pdf "Bougleux, Brun, Carletti, Foggia, Gaüzère, and Vento (2017). Graph edit distance as a quadratic assignment problem. Pattern Recognition Letters, 87, 38-46. doi:10.1016/j.patrec.2016.10.001."
[Kipf2016]:http://bayesiandeeplearning.org/2016/papers/BDL_16.pdf "Kipf, and Welling (2016). Variational Graph Auto-Encoders. Proceedings of the NIPS 2016 Workshop on Bayesian Deep Learning."
[Paa2018]:https://arxiv.org/abs/1704.06498 "Paaßen, Göpfert, and Hammer (2018). Time Series Prediction for Graphs in Kernel and Dissimilarity Spaces. Neural Processing Letters, 48, 669-689. doi:10.1007/s11063-017-9684-5."
[Paa2021]:https://openreview.net/forum?id=dlEJsyHGeaL "Paaßen, B., Grattarola, D., Zambon, D., Alippi, C., and Hammer, B. (2021). Graph Edit Networks. Proceedings of the Ninth International Conference on Learning Representations (ICLR 2021)"
[Zhang1989]:https://doi.org/10.1137/0218082 "Zhang, and Shasha (1989). Simple Fast Algorithms for the Editing Distance between Trees and Related Problems. SIAM Journal on Computing, 18(6), 1245-1262. doi:10.1137/0218082"
