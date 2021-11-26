# Project discontinued

This project has been discontinued and will not be maintained. 
If you are by chance interested in the same question (see About section), I suggest you ask the Borgwardt-Lab to also publish their CNN code since could not be found as of July 2021. Feel free to drop me a note, if you found out anything. 

# About 

This code base is used to evaluate Lottery Ticket Hypothesis type experiments with a focus on the network structure of the winning tickets. Currently, I am using a topological method, the Neural Persistence, for evaluation of winning tickets created with the OpenLTH framework. 

# Installation 

## Pre-installation actions 

Set up any virtual environments you want to use. If you want to install our code and the OpenLTH framework in the same environment, your `python` version can be at most `3.8`, due to dependencies of the OpenLTH framework (as of 05/2021). You need not install both in the same environment, since my code does not directly use OpenLTH code (as of 07/2021). I nevertheless recommend using `pyton 3.8` since I developed the code base in it.

The code expects the dependencies (below) to live in the `./deps` folder, please install them there. 

## Install the Neural Persistence framework 

Clone code from [https://github.com/BorgwardtLab/Neural-Persistence](https://github.com/BorgwardtLab/Neural-Persistence). 

Neural-Persistence uses `Tensorflow 1.6`. However, I am only interested in the content of `Neural-Persistence/src/tda.py` which requires the library `Aleph` to be installed, but not `Tensorflow`. Therefore, you can either follow the Docker file installation (untested) or just pick out the relevant lines for `Aleph` from the Docker file.

## Install the Lottery Ticket Hypothesis framework (optional) 

(You do not need to do this, if you already have OpenLTH running on your machine -- I only work on the outputs of the OpenLTH framework.)

Clone code from [https://github.com/facebookresearch/open_lth](https://github.com/facebookresearch/open_lth).

`python3 setup.py install`


# License 

This code base uses the MIT license. Please refer to [docs/license.md](https://github.com/katjahauser/no_blank_slate/blob/master/docs/license.md) for details. 
