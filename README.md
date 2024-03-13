# Networks of Unmanned Vehicles Package

This package contains a collection of methods and algorithms developed during my PhD at University of Buenos Aires,
tilted Reconfigurable Networks of Unmanned Vehicles.

# Preparation

First, download the project:

```bash
$ mkdir -p $HOME/repo
$ cd $HOME/repo/
$ git clone https://gitlab.com/fpresenza/uv_network.git
```

## Dependencies
Some of the dependencies used work are (all work with python2.7):

 * numpy>=1.23.5
 * matplotlib>=3.7.1
 * scipy>=1.10.1
 * numba>=0.56.4

## Installation
First time it must be installed as:

```bash
$ cd $HOME/repo/uv_network
$ python3 setup.py sdist bdist_wheel
$ pip3 install ./dist/uvnpy-0.0.1.tar.gz
```

Then, once changes are done, it is updated with:
```bash
$ cd $HOME/repo/uv_network
$ pip install .
```

In a python console, it can be imported as:

```
>>> import unvpy
>>> help(uvnpy)
```
