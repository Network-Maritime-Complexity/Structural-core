# Overview

The code in the folder "[Article code](./Article%20code)" and in the folder "[Supplementary notes code](./Supplementary%20notes%20code)" allows one to reproduce the quantitative results reported in the manuscript. We also provide example data to demo the code. The figure below shows how the code is structured.

<div align="center">
<img src="Overview.jpg" width="900px">
</div>

## Folder description

* [Article code](./Article%20code)  
  This folder holds scripts for reproducing the quantitative results reported in the *Article* manuscript. After running a script, you will get a son folder "output" inside this folder "[Article code](./Article%20code)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Article%20code/Expected%20output)". Please find out below [how to use](#article-code).

* [Supplementary notes code](./Supplementary%20notes%20code)  
  This folder holds scripts for reproducing the quantitative results reported in the *Supplementary Notes* manuscript. After running a script, you will get a son folder "output" inside this folder "[Supplementary notes code](./Supplementary%20notes%20code)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Supplementary%20notes%20code/Expected%20output)". Please find out below [how to use](#supplementary-notes-code).

* [Demo](./Demo)  
  This fold contains a small example data to test the code. After running a script, you will get a son folder "output" inside this folder "[Demo](./Demo)". Results will be saved in the son folder "output". For your information, expected results are also available in another son folder "[Expected output](./Demo/Expected%20output)". Please find out below [how to use](#demo).


* [data](./data)  
  This folder already contains in two separate son folders all the empirical data adopted in our study. Additionally, we suggest you download the following zip file named *downloaded data files.zip*, unzip it, and put it inside this folder "[data](./data)". Note that: this zip file contains important process data of our study, which were generated based on the adopted empirical data and were used in many computational experiments; for instance, various null configuration models for the empirical network. This zip file is available at: https://figshare.com/s/2167acb4e1f0106ccf3d. Therefore, before you start reproducing the results, we expect that you should have inside this folder "[data](./data)" three son folders: "[GLSN data](./data/GLSN%20data)", "[Other data](./data/Other%20data)", and "downloaded data files".

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

The folder "[Demo](./Demo)" contains a script [`run.py`](./Demo/run.py). Note that: this script is for testing the code, and one should not try to interpret the results generated during the test process.

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
     - Degree_centrality_distribution_and_assortativity: for reproducing the results in the subsection titled "Degree centrality distribution and assortativity".
     - Economic_small_world_ness: for reproducing the results in the subsection titled "Economic small-world-ness".
     - Multiscale_modularity_and_hubs_diversity: for reproducing the results in the section titled "Multiscale modularity and hubs diversity".
     - Defining_structural_core: for reproducing the results in the subsection titled "Defining structural core".
     - Topological_centrality_of_the_structural_core: for reproducing the results in the subsection titled "Topological centrality of the structural core".
     - Significant_importance_of_core_connections: for reproducing the results in the subsection titled "Significant importance of core connections in supporting long-distance maritime transportation".
     - Structural_embeddedness_and_economic_performance_of_world_ports: for reproducing the results in the section titled "Structural embeddedness and economic performance of world ports".
     - Structural_core_and_the_global_trade: for reproducing the results in the section titled "Structural core and the global trade".

Possible usage, for example:

```
python run.py  # To reproduce the quantitative results reported in the main article

python run.py Degree_centrality_distribution_and_assortativity  # To reproduce the quantitative results reported in the subsection titled "Degree centrality distribution and assortativity"

python run.py Degree_centrality_distribution_and_assortativity Defining_structural_core  # To reproduce the quantitative results reported in the subsection titled "Degree centrality distribution and assortativity" and in the subsection titled "Defining structural core".
```

Note: After performing the code, the results will be saved in a folder labeled with the corresponding *parts_of_the_manuscript*, inside the folder "output" in the folder "[Article code](./Article%20code)".

### Code performance

<div align="center">
<img src="Code performance (Article).jpg" width="750px">
</div>

**Warning:** It will take approximately 11 hours to reproduce the quantitative results reported in the main article manuscript.

## Supplementary notes code

**Note**: Before you run the code, please download (via the link: https://figshare.com/s/2167acb4e1f0106ccf3d) the *downloaded data files.zip*, unzip it and put it inside the folder "[data](./data)". Therefore, we expect that you should have inside this folder "[data](./data)" three son folders: "[GLSN data](./data/GLSN%20data)", "[Other data](./data/Other%20data)", and "downloaded data files".

The folder "[Supplementary notes code](./Supplementary%20notes%20code)" contains a script [`run.py`](./Supplementary%20notes%20code/run.py). To reproduce the quantitative results reported in the *Supplementary Notes* manuscript, please open the *cmd* window in the root folder, then use:

```
cd Supplementary notes code

python run.py <iters> <parts_of_the_manuscript>  # This will reproduce the quantitative results reported in the selected parts_of_the_manuscript, based on the number of iters of the related experiments (if applicable).
```

Positional arguments:

+ *iters*: by default it is 1000; it should be a positive integer within 1000.

+ *parts_of_the_manuscript*: select the manuscript parts whose results you want to reproduce; multiple parts can be selected at a same time, and by default all parts are selected. Parts are listed as follows:

     - note2: for reproducing the results in the section titled "Supplementary note 2: Statistical significance of the economic small-world-ness of the GLSN".
     - note3: for reproducing the results in the section titled "Supplementary note 3: Geographical length of inter-port links in the GLSN".
     - note4: for reproducing the results in the section titled "Supplementary note 4: Gateway-hub-based structural core organization of the GLSN at modular level".
     - note5: for reproducing the results in the section titled "Supplementary note 5: Robustness of empirical findings on the structural-core organization of the GLSN to the non-detrimental property of the Louvain algorithm in community division".
     - note10: for reproducing the results in the section titled "Supplementary note 10: Rich-club coefficients of world ports".
     - note11: for reproducing the results in the section titled "Supplementary note 11: Robustness of the structural-core organization of the GLSN across multiple datasets".
     - note12_1: for reproducing the results in the subsection titled "(1) Constraints of the number of shipping routes".
     - note12_2: for reproducing the results in the subsection titled "(2) Geographical constraints".
     - note12_3: for reproducing the results in the subsection titled "(3) The constraints of the economy of liner shipping network".
     - note13: for reproducing the results in the section titled "Supplementary note 13: Existence of a structural core of the GLSN is not the same as small-world distance scaling".
     - note14: for reproducing the results in the section titled "Supplementary note 14: Significant importance of core connections in supporting long-distance maritime transportation; calculations are based on great-circle distance".

Possible usage, for example:

```
python run.py 10 note5  # To reproduce the results reported in the Supplementary note 5, based on 10 iterations of the corresponding experiments.

python run.py 10 note5 note2  # To reproduce the results reported in the Supplementary note 2 and Supplementary note 5, based on 10 iterations of the corresponding experiments.

python run.py note11  # To reproduce the results reported in the Supplementary note 11, based on 1000 iterations of the corresponding experiments.

python run.py 10  # To reproduce the results reported in the supplementary notes manuscript, based on 10 iterations of the corresponding experiments.
```

After performing the code, the results will be saved in a folder labeled with the corresponding *parts_of_the_manuscript*, inside the folder "output" in the folder "[Supplementary notes code](./Supplementary%20notes%20code)".

### Code performance

<div align="center">
<img src="Code performance (Supplementary notes).jpg" height="350px">
</div>

**Warning:** It shall take approximately **12 days** to reproduce the quantitative results reported in the supplementary notes manuscript, if all computational experiments are conducted for 1000 iterations in a normal computer.

**Note:** 

The code contains the python function <a href='https://docs.python.org/3.6/library/functions.html?highlight=round#round'>round()</a>. The behavior of round() for floats can be surprising: for example, round(2.675, 2) gives 2.67 instead of the expected 2.68. This is not a bug: it’s a result of the fact that most decimal fractions can’t be represented exactly as a float. See <a href='https://docs.python.org/3.6/tutorial/floatingpoint.html#tut-fp-issues'>Floating Point Arithmetic: Issues and Limitations</a> for more information.

# Contact

* Mengqiao Xu: <stephanie1996@sina.com>
* Qian Pan: <qianpan_93@163.com>

# Acknowledgement

We appreciate two lab members for carefully testing the code：

- Jia Song: <songjiavv@163.com>
- Wen Li: <245885195@qq.com>
