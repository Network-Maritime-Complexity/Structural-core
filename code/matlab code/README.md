## Folder description

* [CM_randomization](./CM_randomization)  

This folder contains the code and data to generate the CM null models for the GLSN, which are used in the subsection titled "(2) Geographical constraints", in the section titled "Supplementary note 10: Influence of the constraints on the structural-core organization of the GLSN".

* [Fig 1d](./Fig%201d)  

This folder contains the code and data to reproduce *Fig 1d* in the subsection titled "Data for the GLSN construction".

* [Fig 7](./Fig%207)  

This folder contains the code and data to reproduce *Fig 7* in the subsection titled "Gateway-hub structural core".

* [Supplementary Fig 4](./Supplementary%20Fig%204)   

This folder contains the code and data to reproduce the supplementary figure titled "Supplementary Fig. 4 Statistical significance of the angular separation index".

* [Supplementary Fig 15](./Supplementary%20Fig%2015)  

This folder contains the code and data to reproduce the supplementary figure titled "Supplementary Fig. 15 Representation of the modular communities and their correspondent structural cores in the hyperbolic space".

## Other publicly released matlab codes

The following publicly released matlab code was also used in the study.

* *randmio_und_connected.m*

Code from the Brain Connectivity Toolbox: https://sites.google.com/site/bctnet/null

This function randomizes an undirected network, while preserving the degree distribution. The function also ensures that the randomized network maintains connectedness, the ability for every node to reach every other node in the network. We used this function to generate configuration null models for the real network under study.

* *latmio_und_connected.m*

Code from the Brain Connectivity Toolbox: https://sites.google.com/site/bctnet/null

This function "latticizes" an undirected network, while preserving the degree distribution. The function also ensures that the randomized network maintains connectedness, the ability for every node to reach every other node in the network. We used this function to generate equivalent lattice networks for the real network under study.

* *randomizer_bin_und.m*

Code from the Brain Connectivity Toolbox: https://sites.google.com/site/bctnet/null

This function randomizes a binary undirected network, while preserving the degree distribution. The function directly searches for rewirable edge pairs (rather than trying to rewire edge pairs at random), and hence avoids long loops and works especially well in dense matrices. We used this function to generate null configuration models for the real network under study.

* *plpva.m*

Code from http://www.santafe.edu/~aaronc/powerlaws/

This function calculates the p-value for a given power-law fit to some data. We used this function to test whether or not the degree distribution of the real network under study is a power law.

* *topological_measures_wide_analysis*

Code from https://github.com/biomedical-cybernetics/topological_measures_wide_analysis

We used this code to calculate the LCP-corr value of the real network under study, which is reported in *Fig. 2 Basic topological properties of the GLSN* and also in *Supplementary Fig. 1 Basic topological properties of the GLSN of 2017*.

* *rich_club*

Code from https://github.com/biomedical-cybernetics/rich_club

We used this code to calculate the nodes' <a href="https://www.codecogs.com/eqnedit.php?latex=\rho&space;_{\mathrm{CM}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\rho&space;_{\mathrm{CM}}" title="\rho _{\mathrm{CM}}" /></a>, a normalized rich-club coefficient proposed by Muscoloni and Cannistraci (Rich-clubness test: how to determine whether a complex network has or doesn’t have a rich-club? arXiv Prepr. arXiv1704.03526 (2017).)
