# QSDVR
A Differentiable volume renderer base on quadric surface grid.

## Prerequisites

* Cuda 11.2+
* Python 3.8+
* Pip 10+

## Setup
First make sure all the Prerequisites are installed in your operating system.

For the main python library, just clone this repository and pip install.

```bash
git clone https://github.com/595744412/qsdvr.git
cd ./qsdvr
pip install -r requirements.txt
pip install .
```
## Test
To render the init model, run

```bash
python3 ./train/test.py
```