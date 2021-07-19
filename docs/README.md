# Installation 

## Pre-installation actions

Set up any virtual environments, etc. you want to use. `Python` version max. `3.8`, due to dependencies of the LTH framework (as of 05/2021).

## Install the Neural Persistence framework

We are only interested in the contents of tda.py which requires `Aleph` to be installed . Picking out the necessary lines from the Dockerfile worked for me to this end. 

Please note that the Neural Persistence framework otherwise seems to be using an older version of tensorflow, e.g. the deprecated tensorflow.examples. 

## Install the Lottery Ticket Hypothesis framework

`python3 setup.py install`

