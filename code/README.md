# Overview

The code in the folder "[Article code](./Article%20code)" and in the folder "[Supplementary information code](./Supplementary%20information%20code)" allows one to reproduce the quantitative results reported in the manuscript. We also provide example data to demo the code. The figure below shows how the code is structured.

<div align="center">
<img src="Overview.jpg" width="900px">
</div>

## Folder description

* [Article code](./Article%20code)  
  This folder holds scripts for reproducing the quantitative results reported in the *Article* manuscript. After running a script, you will get a son folder "output" inside this folder "[Article code](./Article%20code)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Article%20code/Expected%20output)". Please find out below [how to use](#article-code).

* [Supplementary information code](./Supplementary%20information%20code)  
  This folder holds scripts for reproducing the quantitative results reported in the *Supplementary Information* manuscript. After running a script, you will get a son folder "output" inside this folder "[Supplementary information code](./Supplementary%20information%20code)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Supplementary%20information%20code/Expected%20output)". Please find out below [how to use](#supplementary-information-code).

* [Demo](./Demo)  
  This fold contains a small example data to test the code. After running a script, you will get a son folder "output" inside this folder "[Demo](./Demo)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Demo/Expected%20output)". Please find out below [how to use](#demo).


* [data](./data)  
  This folder already contains in two separate son folders all the empirical data adopted in our study. Additionally, we suggest you download (in this link: https://doi.org/10.6084/m9.figshare.12136236.v1) the following 8 zip files named: "1000 Community divisions.7z", "1000 equivalent random networks.7z", "note5.7z", "note9.7z", "note10_1.7z", "note10_2.7z", "note10_3.7z", "note11.7z". Then please unzip them and put them inside this folder "[data](./data)". Note: these zip files contain important process data of our study, which were generated based on the adopted empirical data and were used in many computational experiments; for instance, various null configuration models for the empirical network. Therefore, before you start reproducing the results, we expect that you should have inside this folder "[data](./data)" 10 son folders: "[GLSN data](./data/GLSN%20data)", "[Other data](./data/Other%20data)", "1000 Community divisions", "1000 equivalent random networks", "note5", "note9", "note10_1", "note10_2", "note10_3", "note11".

* [matlab code](./matlab%20code)  
  This folder contains the matlab code we had implemented for the present study. For detailed instructions, please find inside this folder a file named [README.md](./matlab%20code/README.md).
  
# System Requirements 

## OS Requirements

These scripts have been tested on *Windows10* operating system. The Python package should be compatible with *Windows10* operating system.

### Installing Python on Windows

Before setting up the package, users should have Python version 3.6 or higher, and several packages set up from Python 3.6. The latest version of python can be downloaded from the official website: https://www.python.org/

The installation shall be completed in about 10 minutes.

## Hardware Requirements 

The package requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 8+ GB  
CPU: 4+ cores, 3.4+ GHz/core

The runtimes are generated in a computer with the recommended specs (8 GB RAM, 4 cores@3.4 GHz) and internet of speed 1 Gbps.

# Installation Guide

## Package dependencies

Users should install the following packages prior to running the code:

```
matplotlib==3.0.2
networkx==2.2
numpy==1.15.4
openpyxl==2.5.3
xlwt==1.3.0
xlrd==1.2.0
pandas==0.23.4
powerlaw==1.4.6
python-louvain==0.13
scikit-learn==0.19.1
scipy==1.1.0
seaborn==0.9.0
```

To install all packages on *windows* from *cmd*, please open the *cmd* window in the root folder, type:

```
pip3 install -r requirements.txt
```

The package should take approximately 5 minutes to be installed on a recommended computer. 

If you want to install only one of the packages, use:

```
pip3 install pandas==0.23.4
```

# Demo

The folder "[Demo](./Demo)" contains a script [`run.py`](./Demo/run.py). Note: this script is for testing the code, and one should not try to interpret the results generated during the test process.

Demo data description: 

```
# Nodes: 147
# Edges: 2463
Density: 0.230
```

Please open the *cmd* window in the root folder, then use:

```
cd Demo

python run.py
```

Time consumption of testing the code in a normal computer: 2 minutes.

# How To Use

## Article code

The folder "[Article code](./Article%20code)" contains a script [`run.py`](./Article%20code/run.py). To reproduce the quantitative results reported in the *Article* manuscript, please open the *cmd* window in the root folder, then use:

```
cd Article code

python run.py <parts_of_the_manuscript>  # This will reproduce the quantitative results reported in the corresponding parts_of_the_manuscript
```

Positional argument:

+ *parts_of_the_manuscript*: select the manuscript parts whose results you want to reproduce; multiple parts can be selected at a same time, and by default all parts are selected. Parts are listed as follows:
     - Basic_topological_properties_and_economic_small_world_ness: for reproducing the results in the subsection titled "Basic topological properties and economic small-world-ness".
     - Multiscale_modularity_and_hubs_diversity: for reproducing the results in the subsection titled "Multiscale modularity and hubs diversity".
     - Gateway_hub_structural_core: for reproducing the results in the subsection titled "Gateway-hub structural core".
     - Structural_embeddedness_and_economic_performance_of_ports: for reproducing the results in the subsection titled "Structural embeddedness and economic performance of ports".
     - Structural_core_and_international_trade: for reproducing the results in the subsection titled "Structural core and international trade".

Possible usage, for example:

```
python run.py  # To reproduce the quantitative results reported in the main article

python run.py Basic_topological_properties_and_economic_small_world_ness  # To reproduce the quantitative results reported in the subsection titled "Basic topological properties and economic small-world-ness"

python run.py Basic_topological_properties_and_economic_small_world_ness Multiscale_modularity_and_hubs_diversity  # To reproduce the quantitative results reported in the subsection titled "Basic topological properties and economic small-world-ness" and in the subsection titled "Multiscale modularity and hubs diversity".
```

Note: After performing the code, the results will be saved in a folder labeled with the corresponding *parts_of_the_manuscript*, inside the folder "output" in the folder "[Article code](./Article%20code)".

### Code performance

<div align="center">
<img src="Code performance (Article).jpg" width="750px">
</div>

**Warning:** It will take approximately 3 hours to reproduce the quantitative results reported in the main article manuscript.

## Supplementary information code

**Note**: Before you run the code, please download (via the link: https://doi.org/10.6084/m9.figshare.12136236.v1) the follwing 8 zip files named: "1000 Community divisions.7z", "1000 equivalent random networks.7z", "note5.7z", "note9.7z", "note10_1.7z", "note10_2.7z", "note10_3.7z", "note11.7z". Then please unzip them and put them inside the folder "[data](./data)". Therefore, we expect that you should have inside this folder "[data](./data)" 10 son folders: "[GLSN data](./data/GLSN%20data)", "[Other data](./data/Other%20data)", "1000 Community divisions", "1000 equivalent random networks", "note5", "note9", "note10_1", "note10_2", "note10_3", "note11".

The folder "[Supplementary information code](./Supplementary%20information%20code)" contains a script [`run.py`](./Supplementary%20information%20code/run.py). To reproduce the quantitative results reported in the *Supplementary Information* manuscript, please open the *cmd* window in the root folder, then use:

```
cd Supplementary information code

python run.py <iters> <parts_of_the_manuscript>  # This will reproduce the quantitative results reported in the selected parts_of_the_manuscript, based on the number of iters of the related experiments (if applicable).
```

Positional arguments:

+ *iters*: by default it is 1000; it should be a positive integer within 1000.

+ *parts_of_the_manuscript*: select the manuscript parts whose results you want to reproduce; multiple parts can be selected at a same time, and by default all parts are selected. Parts are listed as follows:

     - Supplementary_Fig_2: for reproducing the result file titled "Supplementary Fig. 2 Proportional distribution of intra- and inter- module links in different range of geographical length".
     - Supplementary_Fig_3: for reproducing the result file titled "Supplementary Fig. 3 Statistical significance of the structural core in the real GLSN of 2015".
     - Supplementary_Fig_6: for reproducing the result file titled "Supplementary Fig. 6 Rich-club coefficients of world ports".
     - Supplementary_Fig_7: for reproducing the result file titled "Supplementary Fig. 7 Overlap between the rich club and the structural core of the GLSN".
     - note1: for reproducing the results in the section titled "Supplementary note 1: Statistical significance of the economic small-world-ness of the GLSN".
     - note5: for reproducing the results in the section titled "Supplementary note 5: Robustness of empirical findings on the structural-core organization of the GLSN to the non-detrimental property of the Louvain algorithm in community division".
     - note6: for reproducing the results in the section titled "Supplementary note 6: Gateway-hub structural core organization of the GLSN at modular level".
     - note8: for reproducing the results in the section titled "Supplementary note 8: Significant importance of core connections in supporting long-distance maritime transportation; calculations are based on great-circle distance".
     - note9: for reproducing the results in the section titled "Supplementary note 9: Robustness of the structural-core organization of the GLSN across multiple datasets".
     - note10_1: for reproducing the results in the subsection titled "(1) Constraints of the number of shipping routes".
     - note10_2: for reproducing the results in the subsection titled "(2) Geographical constraints".
     - note10_3: for reproducing the results in the subsection titled "(3) The constraints of the economy of liner shipping network".
     - note11: for reproducing the results in the section titled "Supplementary note 11: Existence of a structural core of the GLSN is not the same as small-world distance scaling".

Possible usage, for example:

```
python run.py 10 note5  # To reproduce the results reported in the Supplementary note 5, based on 10 iterations of the corresponding experiments.

python run.py 10 note5 note6  # To reproduce the results reported in the Supplementary note 5 and Supplementary note 6, based on 10 iterations of the corresponding experiments.

python run.py note11  # To reproduce the results reported in the Supplementary note 11, based on 1000 iterations of the corresponding experiments.

python run.py 10  # To reproduce the results reported in the supplementary information manuscript, based on 10 iterations of the corresponding experiments.
```

After performing the code, the results will be saved in a folder labeled with the corresponding *parts_of_the_manuscript*, inside the folder "output" in the folder "[Supplementary information code](./Supplementary%20information%20code)".

### Code performance

<div align="center">
<img src="Code performance (Supplementary information).jpg" height="350px">
</div>

**Warning:** It shall take approximately **13 days** to reproduce the quantitative results reported in the supplementary information manuscript, if all computational experiments are conducted for 1000 iterations in a normal computer.

**Note:** 

The code contains the python function round(): https://docs.python.org/3.6/library/functions.html?highlight=round#round. The behavior of round() for floats can be surprising: for example, round(2.675, 2) gives 2.67 instead of the expected 2.68. This is not a bug: it’s a result of the fact that most decimal fractions can’t be represented exactly as a float. See Floating Point Arithmetic: Issues and Limitations: https://docs.python.org/3.6/tutorial/floatingpoint.html#tut-fp-issues for more information.

# Contact

* Mengqiao Xu: <stephanie1996@sina.com>
* Qian Pan: <qianpan_93@163.com>

# Acknowledgement

We appreciate two lab members for carefully testing the code：

- Jia Song: <songjiavv@163.com>
- Wen Li: <245885195@qq.com>
