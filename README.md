# Retinal OCT recognition experiments with Transfer Learning and Semi Supervised Learning

## Table Of Contents
- [1. About this repository](#1-about-this-repository)
- [2. Introduction](#2-introduction)
- [3. The process](#3-the-process)
- [4. The Datasets and Data Wrangling](#4-the-datasets-and-data-wrangling)
  - [4.1. The Data Wrangling Notebooks](#41-the-data-wrangling-notebooks)
    - [4.1.1. data_wrangling_from_kaggle_dataset.ipynb](#411-data_wrangling_from_kaggle_datasetipynb)
    - [4.1.2. data_wrangling_from_mendeley.ipynb](#412-data_wrangling_from_mendeleyipynb)
    - [4.1.3. data_wrangling_comparison.ipynb](#413-data_wrangling_comparisonipynb)
  - [4.2. Data Wrangling conclusions](#42-data-wrangling-conclusions)
- [5. Baseline experiments](5-baseline-experiments)
  - [5.1. Baseline, Resnet50, Full Dataset, Training All Layers](#51-baseline-resnet50-full-dataset-training-all-layers)
  - [5.2. Baseline, Resnet50, Reduced Dataset, Training All Layers](#52-baseline-resnet50-reduced-dataset-training-all-layers)
  - [5.3. Baseline, Resnet50, Reduced Dataset, Training Last 34 Layers](#53-baseline-resnet50-reduced-dataset-training-last-34-layers)


## 0. Requirements
This project was done in January 2022. It uses TensorFlow 2.

## 1. About this repository
*[↑ TOC](#table-of-contents)*

This project was done by:
- Felipe Caballero (https://github.com/caballerofelipe/)
- Gabriel Santos Elizondo (https://github.com/Gsantos4)
- George Prounis (https://github.com/prounis)

It was the Capstone Project for the Machine Learning Engineer course by [FourthBrain.ai](https://www.fourthbrain.ai/).

Check out the [Report](./report.pdf) and the [Presentation](./presentation.pdf).

## 2. Introduction
*[↑ TOC](#table-of-contents)*

Retinal OCTs (Optical Coherence Tomography) are images created by scanning the back of the eye using specified equipment. An ophthalmologist can detect different eye conditions to treat them. Detecting and treating these conditions in early stages usually means eye recovery and no vision loss.

In the project, three abnormal conditions were present alongside normal retinas. Machine learning was used for condition recognition in the images.

**We improved performance on a limited (5%) labeled dataset with a semi-supervised learning approach (+0.9% F1).**

## 3. The process
*[↑ TOC](#table-of-contents)*

The project was done in these steps:
- Data Wrangling
- Baseline experiments
- Transfer Learning after training the model on COVID Dataset ([*COVID-CTset: A Large COVID-19 CT Scans dataset*](https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset)).
- Semi supervised learning using only 5% of labeled data.

## 4. The Datasets and Data Wrangling
*[↑ TOC](#table-of-contents)*

The proposed project pointed to a Kaggle Dataset (link below). But this data advertised a 5.81GB dataset and when downloaded it was more than 10GB. So we decided to review the files. We end up using the Mendeley Dataset (link below), further explained below in [Data Wrangling conclusions](#data-wrangling-conclusions)

- Mendeley (The one used during the project)
  - [Version 2 of the project](https://data.mendeley.com/datasets/rscbjbr9sj/2).
  - [Version 3 of the project](https://data.mendeley.com/datasets/rscbjbr9sj/3).
  - [Direct download](https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded).
- Kaggle
  - [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/kermany2018/).
  - [Direct download](https://www.kaggle.com/paultimothymooney/kermany2018/download) if registered in Kaggle.
  - Kaggle API command: `kaggle datasets download -d paultimothymooney/kermany2018`


### 4.1. The Data Wrangling Notebooks
*[↑ TOC](#table-of-contents)*

The data wrangling was done in the three notebooks listed next.

#### 4.1.1. data_wrangling_from_kaggle_dataset.ipynb
*[↑ TOC](#table-of-contents)*

[`data_wrangling/kaggle/data_wrangling_from_kaggle_dataset.ipynb`](./data_wrangling/kaggle/data_wrangling_from_kaggle_dataset.ipynb)

In this notebook we created a Pandas DataFrame to store all the information about the files in the [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/kermany2018/), incluiding their md5 hash, this was done for comparison purpuses.

The result for this notebook is a CSV file included with it.

#### 4.1.2. data_wrangling_from_mendeley.ipynb
*[↑ TOC](#table-of-contents)*

[`data_wrangling/mendeley/data_wrangling_from_mendeley.ipynb`](./data_wrangling/mendeley/data_wrangling_from_mendeley.ipynb)

In this notebook we created a Pandas DataFrame to store all the information about the files in the [Mendeley Dataset](https://data.mendeley.com/datasets/rscbjbr9sj/2), incluiding their md5 hash, this was done for comparison purpuses.

The results for this notebook are two CSV file included with it:
- The first CSV (`mendeley_filelist.csv`) contains information on all files.
- The second CSV (`mendeley_filelist_combo_cond_md5.csv`) removing duplicates (duplicates are considered when two or more files have the same condition and md5).

#### 4.1.3. data_wrangling_comparison.ipynb
*[↑ TOC](#table-of-contents)*

[`data_wrangling/data_wrangling_comparison.ipynb`](./data_wrangling/data_wrangling_comparison.ipynb)

The objective of this notebook was to compare both datasets to see if they had the same unduplicated notebooks.

Both datasets contained the same unduplicated files. Se from this point on we used the Mendeley one.

### 4.2. Data Wrangling conclusions
*[↑ TOC](#table-of-contents)*

We use the [Mendeley Dataset](https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/5699a1d8-d1b6-45db-bb92-b61051445347/file_downloaded).

<figure align="center"><img src="./images/data_distribution.png" alt="Dataset class (conditions) distribution" style="width:50%"><figcaption align = "center">Dataset class (conditions) distribution</figcaption></figure>

We consider duplicates when two or more files have the same condition and md5.

The Mendeley dataset we used (as the Kaggle one) contains a total of 84,484 files, of which 7,357 are duplicates. We ended with 77,127 usable files. We used a Pandas DataFrame to store these files’ information and to do further work.

## 5. Baseline experiments

### 5.1. Baseline, Resnet50, Full Dataset, Training All Layers

### 5.2. Baseline, Resnet50, Reduced Dataset, Training All Layers

### 5.3. Baseline, Resnet50, Reduced Dataset, Training Last 34 Layers



## To be reviewed
### Implementation details
#### The Notebooks
- `v1 OCT TensorFlow Resnet 50 Model.ipynb`  
First attempt to train a Resnet50 for classification of the different conditions. Using TensorFlow's default parameters.
- `v2 OCT TensorFlow Resnet 50 Model.ipynb`  
Second attempt to train a Resnet50 for classification of the different conditions. Using tested parameters and earning Rate reduction on plateau.
- <font color="red">GEORGE'S NOTEBOOK WELL FORMATTED</font>  
Transfer learning experiment, a Resnet50 trained on an OCT dataset and then used that model as transfer learning for a reduced OCT dataset.
- <font color="red">GABRIEL'S NOTEBOOK WELL FORMATTED</font>  
Notebook to define hyperparameters for the classification problem of the OCT dataset.
- `v2_reduced OCT TensorFlow Resnet 50 Model.ipynb`  
Some later experiments use a reduced training dataset, this serves as a baseline for those experiments.
