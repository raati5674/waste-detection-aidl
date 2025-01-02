# Waste Segmentation and detection AIDL project

This repository contains the code developed by Martí Fabregat, Rafel Febrer, Ferran Miró-Gea and Miguel Ortiz in the scope of AIDL postgraduate course in UPC (Universitat Politècnica de Catalunya). With supervision of Amanda Duarte.

A vision transformer (ViT) is trained and evaluated to segment and classify waste. The model has been trained using [TACO Dataset](http://tacodataset.org) by Pedro F Proença and Pedro Simões. For more details check the paper: https://arxiv.org/abs/2003.06975

## Getting started

### Requirements 

To install the required python packages simply type
```
pip3 install -r requirements_TACO.txt
```
### Download the dataset

To download the dataset images simply use:
```
python download.py
```

### Exploratory data analysis

Explore the notebook ``demo.pynb``, modified version of the original notebook from the [TACO Repository](https://github.com/pedropro/TACO) that inspects the dataset.
The dataset is in COCO format. It contains the source pictures, anotations and labels. For more details related with the datasource please refer to [TACO Repository](https://github.com/pedropro/TACO).

### Split annotations in train, validation and test

To split the annotations for training and evaluation use ``split_dataset.py``. It has several optional flags.
```
python split_dataset.py --dataset_dir ./data [--test_percentage <0.1>] [--val_percentage <0.1>] [--seed <123>] [--verbose <False>]
```
Use ``--test_percentage`` if you want to use a test split different than default 0.1 (10%).
Use ``--val_percentage`` if you want to use a validation split different than default 0.1 (10%).
Use ``--seed`` if you want to have a different random output. Default 123.
Use ``--verbose`` if you want to have printed text on the console during execution.





