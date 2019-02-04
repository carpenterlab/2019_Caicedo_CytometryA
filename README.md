# Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images

This repository contains a toolkit to evaluate the accuracy of segmentation algorithms. 
These tools were used in our study of deep learning segmentation methods for nuclei in fluorescent images: 
https://www.biorxiv.org/content/10.1101/335216v3.full

There are several directories with tools for segmentation and Python code for evaluation of segmentations. 
The dataset used in our evaluation can be found here: 
https://data.broadinstitute.org/bbbc/BBBC039/

In our benchmark, we also evaluated a DeepCell model on this same data.
Checkout our docker container with the adapted configuration for generating segmentations with DeepCell:
https://hub.docker.com/r/jccaicedo/deepcell/


## Contents

### `unet4nuclei`: nucleus segmentation using U-Net

Set of notebooks for training and testing a U-net architecture. 
The essential notebooks are numbered and ordered by functionalities: 
0) download dataset, 1) prepare and normalize data, 
2) train a U-Net model, 3) run predictions, and 4) make evaluation.

There are other notebooks and scripts, but the ones with a number prefix are the essentials.
The `examples` directory contains example configuration files with pre-defined parameters for running the notebooks.
It's important to update the path to the root directory to make an initial run. 
Everything else can be tuned later according to your needs.

### Evaluation toolkit

Example code for running evaluation is implemented in the notebook #4 of the previous package.
The actual evaluation functions are implemented in the `unet4nuclei/utils/evaluation.py` module.
This module can be reused in other scripts if necessary, or notebook #4 can be adapted to generate an evaluation report.

### CellProfiler pipelines

As part of our benchmark, we created two CellProfiler pipelines to compare against deep learning models.
These pipelines and the associated scripts can be found in the CellProfiler directory.

### Random Forest segmentations

We used Ilastik to create a machine-learning-based workflow to generate segmentations using the same annotations in our dataset.
The scripts adapt the data and Ilastik files to run experiments and generate probability masks.
The resulting segmentations can be evaluated using the same toolkit and notebook #4 of the `unet4nuclei` directory.

### Cell cycle measurements

We implemented a Z'-factor test to quantify the impact of segmentations on cell cycle when applying certain drugs to cells.
The segmentations are initially generated with U-Net or CellProfiler, and then the DNA content of single cells is measured.
We analyze the distribution of DNA content in population of cells to detect cell cycle disruption.

