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
- [5. Model and Hyperparameter selection](#5-model-and-hyperparameter-selection)
- [6. Baseline experiments](#6-baseline-experiments)
  - [6.1. Baseline experiments (1), Resnet50, Full Dataset, Training All Layers](#61-baseline-experiments-1-resnet50-full-dataset-training-all-layers)
  - [6.2. Baseline experiments (2), Resnet50, Reduced Dataset, Training All Layers](#62-baseline-experiments-2-resnet50-reduced-dataset-training-all-layers)
  - [6.3. Baseline experiments (3), Resnet50, Reduced Dataset, Training Last 34 Layers](#63-baseline-experiments-3-resnet50-reduced-dataset-training-last-34-layers)
  - [6.4. v2_model OCT TensorFlow Resnet 50 Model.ipynb](#64-v2model-oct-tensorflow-resnet-50-modelipynb)
  - [6.5. v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb](#65-v2frozenreduced5percent-oct-tensorflow-resnet-50-modelipynb)
- [7. Transfer Learning](#7-transfer-learning)
  - [7.1. The Dataset](#71-the-dataset)
  - [7.2. The Notebooks](#72-the-notebooks)
- [8. Semi-Supervised Learning](#8-semi-supervised-learning)
- [9. Extra Files](#9-extra-files)
  - [9.1 get_stats_on_test_set.py](#91-getstatsontestsetpy)




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
- Data Wrangling ([4. The Datasets and Data Wrangling](#4-the-datasets-and-data-wrangling)).
- Model and Hyperparameter selection ([5. Model and Hyperparameter selection](#5-model-and-hyperparameter-selection)).
- Baseline experiments ([6. Baseline experiments](#6-baseline-experiments)).
- Transfer Learning after training the model on COVID Dataset ([7. Transfer Learning](#7-transfer-learning)).
- Semi supervised learning using only 5% of labeled data ([8. Semi-Supervised Learning](#8-semi-supervised-learning)).





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

<div align="center" style="max-width: 450px;">

|Dataset class (conditions) distribution|
|:-:|
|![Dataset class (conditions) distribution](./images/data_distribution.png)|

</div>

We consider duplicates when two or more files have the same condition and md5.

The Mendeley dataset we used (as the Kaggle one) contains a total of 84,484 files, of which 7,357 are duplicates. We ended with 77,127 usable files. We used a Pandas DataFrame to store these files’ information and to do further work.





## 5. Model and Hyperparameter selection
*[↑ TOC](#table-of-contents)*





## 6. Baseline experiments
*[↑ TOC](#table-of-contents)*

Many experiments were done for our baseline, these are listed in the below sections.

All experiments were done using 30 epochs, to allow us to have comparable results.

Not all of these experiments were presented in the [Report](./report.pdf) and in the [Presentation](./presentation.pdf), we presented what was more relevant and what made more sense in terms of comparison to the following experiments. The important baselines are:
- [6.4. v2_model OCT TensorFlow Resnet 50 Model.ipynb](#64-v2model-oct-tensorflow-resnet-50-modelipynb)
- [6.5. v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb](#65-v2frozenreduced5percent-oct-tensorflow-resnet-50-modelipynb)

### 6.1. Baseline experiments (1), Resnet50, Full Dataset, Training All Layers
*[↑ TOC](#table-of-contents)*

The idea behind these experiments was to have a baseline with a fully trained Neural Network using the complete dataset.

Two experiments were done:
- (**_**) [`baselines/resnet50_full_dataset_training_all_layers/v1 OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_full_dataset_training_all_layers/v1%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**x**) [`baselines/resnet50_full_dataset_training_all_layers/v2_model OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_full_dataset_training_all_layers/v2_model%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)

**_**: Not used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).
<br>**x**: Used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).

### 6.2. Baseline experiments (2), Resnet50, Reduced Dataset, Training All Layers
*[↑ TOC](#table-of-contents)*

The idea behind these experiments was to have a baseline with a fully trained Neural Network a reduced dataset.

Four experiments were done:
- (**_**) [`baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_10percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_10percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_20percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_20percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_30percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_all_layers/v2_reduced_30percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)

**_**: Not used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).
<br>**x**: Used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).

### 6.3. Baseline experiments (3), Resnet50, Reduced Dataset, Training Last 34 Layers
*[↑ TOC](#table-of-contents)*

The idea behind these experiments was to have a baseline with a Neural Network trained on only the last 34 layers and a reduced dataset.

Four experiments were done:
- (**x**) [`baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_10percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_10percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_20percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_20percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)
- (**_**) [`baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_30percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_30percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)

**_**: Not used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).
<br>**x**: Used for the [Report](./report.pdf) and the [Presentation](./presentation.pdf).

### 6.4. v2_model OCT TensorFlow Resnet 50 Model.ipynb
*[↑ TOC](#table-of-contents)*

[`baselines/resnet50_full_dataset_training_all_layers/v2_model OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_full_dataset_training_all_layers/v2_model%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)

This notebook used the hyperparameters obtained in model and hyperparameter selection. It was used for comparison purposes in the [Report](./report.pdf) and in the [Presentation](./presentation.pdf).

About this model:
- The weights were initialized with the ImageNet weights and all layers were trained.
- Trained on the complete the training set.

Below are some images with stats about this model.

<div align="center" style="max-width: 450px;">

|v2_model OCT TensorFlow Resnet 50 Model.ipynb, classification report|
|:-:|
|![v2_model OCT TensorFlow Resnet 50 Model.ipynb, classification report](./images/v2_model%20OCT%20TensorFlow%20Resnet%2050%20Model_classification_report.png)|

</div>

<div align="center" style="max-width: 450px;">

|v2_model OCT TensorFlow Resnet 50 Model.ipynb, confusion matrix|
|:-:|
|![v2_model OCT TensorFlow Resnet 50 Model.ipynb, confusion matrix](./images/v2_model%20OCT%20TensorFlow%20Resnet%2050%20Model_confusion_matrix.png)|

</div>

<div align="center" style="max-width: 450px;">

|v2_model OCT TensorFlow Resnet 50 Model.ipynb, classification with heatmap|
|:-:|
|![v2_model OCT TensorFlow Resnet 50 Model.ipynb, classification with heatmap](./images/v2_model%20OCT%20TensorFlow%20Resnet%2050%20Model_classification_heatmap.png)|

</div>

### 6.5. v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb
*[↑ TOC](#table-of-contents)*

[`baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb`](./baselines/resnet50_reduced_dataset_training_last34_layers/v2_frozen_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model.ipynb)

This notebook used the hyperparameters obtained in model and hyperparameter selection. It was used for comparison purposes in the [Report](./report.pdf) and in the [Presentation](./presentation.pdf).

About this model:
- The weights were initialized with the ImageNet weights and all layers were frozen except the last 34 for training.
- Trained on only 5% of the training set.

Below are some images with stats about this model.

<div align="center" style="max-width: 450px;">

|v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, classification report|
|:-:|
|![v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, classification report](./images/v2_frozen_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model_classification_report.png)|

</div>

<div align="center" style="max-width: 450px;">

|v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, confusion matrix|
|:-:|
|![v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, confusion matrix](./images/v2_frozen_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model_confusion_matrix.png)|

</div>

<div align="center" style="max-width: 450px;">

|v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, classification with heatmap|
|:-:|
|![v2_frozen_reduced_5percent OCT TensorFlow Resnet 50 Model.ipynb, classification with heatmap](./images/v2_frozen_reduced_5percent%20OCT%20TensorFlow%20Resnet%2050%20Model_classification_heatmap.png)|

</div>





## 7. Transfer Learning
*[↑ TOC](#table-of-contents)*

### 7.1. The Dataset
*[↑ TOC](#table-of-contents)*

([*COVID-CTset: A Large COVID-19 CT Scans dataset*](https://www.kaggle.com/mohammadrahimzadeh/covidctset-a-large-covid19-ct-scans-dataset))

### 7.2. The Notebooks
*[↑ TOC](#table-of-contents)*





## 8. Semi-Supervised Learning
*[↑ TOC](#table-of-contents)*





## 9. Extra Files
*[↑ TOC](#table-of-contents)*

## 9.1 get_stats_on_test_set.py
*[↑ TOC](#table-of-contents)*

[`extra_files/get_stats_on_test_set.py`](./extra_files/get_stats_on_test_set.py)

This file was used to obtain the main stats using different models.

To show help:
```
python extra_files/get_stats_on_test_set.py -h
```

The help message:
```shell
usage: get_stats_on_test_set.py [-h] -m MODEL_FILE_PATH -c MENDELEY_CSV_PATH
                                [--image-size IMAGE_SIZE]
                                [--base-path BASE_PATH]
                                [--test-fraction TEST_FRACTION] [--gpu GPU]

Show metrics on the 5% test set using a given model.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_FILE_PATH    Model path.
  -c MENDELEY_CSV_PATH  Mendeley csv file location.
  --image-size IMAGE_SIZE
                        Image size tuple e.g.: (500,500).
  --base-path BASE_PATH
                        Directory where the OCT2017 is located (container of
                        OCT2017).
  --test-fraction TEST_FRACTION
                        Amount of data from total used as test set.
  --gpu GPU             GPU where to run the model.
```
