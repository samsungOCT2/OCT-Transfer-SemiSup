#!/usr/bin/env python3

import math
import os
import time
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as kbackend
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Argparse stuff
import sys
import argparse  # https://docs.python.org/3/library/argparse.html

# https://github.com/tensorflow/tensorflow/issues/50487#issuecomment-1012785682
import atexit

def obtain_stats(model_file_path, mendeley_csv_path, BASE_PATH='.', IMAGE_SIZE=(256,256), SPLIT_TEST_FRACTION=0.05, GPU=None):

    # Read CSV into Pandas DataFrame
    print(f'* Loading CSV [{str(mendeley_csv_path)}].')
    df_original = pd.read_csv(mendeley_csv_path, index_col=0)
    # print(df_original.head(2))
    df = pd.concat([df_original, pd.get_dummies(df_original["condition"])], axis=1)
    # df.sample(5, random_state=42)

    # Split into train/test
    # https://stackoverflow.com/a/70573258/1071459
    print(f'* Splitting data (test={SPLIT_TEST_FRACTION}).')
    df_test = df.sample(frac=SPLIT_TEST_FRACTION, axis=0, random_state=42)
    # get everything but the test sample
    df_train = df.drop(index=df_test.index)
    # # df_train_reduced = df_train.sample(frac=REDUCED_FRACTION, axis=0, random_state=42)
    # print(f"Examples by set: df_test: {df_test.shape[0]} / df_train: {df_train.shape[0]}")

    # Load model
    print(f'* Model location [{model_file_path}].')
    # print(f'tf.config.list_physical_devices("GPU"): {tf.config.list_physical_devices("GPU")}')
    if GPU is not None:
        print(f'* Sprecifying GPU [{GPU}].')
        strategy = tf.distribute.MirroredStrategy([GPU])
        with strategy.scope():
            functional_model_for_testing = tf.keras.models.load_model(model_file_path)
    else:
        print('* No GPU specified.')
        functional_model_for_testing = tf.keras.models.load_model(model_file_path)

    # Create Generator
    # ImageDataGenerator is used to obtain the images from the test set without augmentations
    print(f'* IMAGE_SIZE: [{IMAGE_SIZE}].')
    print('* Create ImageDataGenerator.')
    test_image_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        dtype=tf.float32,
    )
    test_generator = test_image_gen.flow_from_dataframe(
        dataframe=df_test,  # DataFrame
        directory=f"{BASE_PATH}/",
        x_col="file_location",
        y_col="condition",
        color_mode="rgb",  # color
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        shuffle=False,  # IMPORTANT: Set to False to avoid changing the orden from the dataframe
        # batch_size=BATCH_SIZE,  # default 32
    )

    # Create y_test
    print('* Creating y_test.')
    y_test = df_test["condition"].map(test_generator.class_indices).to_numpy()

    # Prediction
    print('* Doing prediction.')
    prediction_full = functional_model_for_testing.predict(test_generator)
    # prediction_full
    prediction_class = np.argmax(prediction_full, axis=-1)

    # Classification report
    print('* Crating classification report.')
    report_printable = classification_report(
        y_test, prediction_class, target_names=test_generator.class_indices
    )
    report_dict = classification_report(
        y_test,
        prediction_class,
        target_names=test_generator.class_indices,
        output_dict=True,
    )

    # Classification report DataFrame
    report_df = pd.DataFrame(report_dict)
    print('-'*30)
    print(report_df[['accuracy', 'macro avg', 'weighted avg']])
    if GPU is not None:
        atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore


def main(argv):
    # Using https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description="Show metrics on the 5% test set using a given model."
    )
    # Not using double `--` to allow bash replacement of ~
    #   This probably can be done with some custom action kwarg
    #   This is a conceptual choise because it can still use ~ without transorming it
    parser.add_argument(
        "-m",
        dest="model_file_path",
        type=str,
        help="Model path.",
        required=True,
    )
    parser.add_argument(
        "-c",
        dest="mendeley_csv_path",
        type=str,
        help="Mendeley csv file location.",
        required=True,
    )
    parser.add_argument(
        '--image-size',
        type=str,
        dest='IMAGE_SIZE',
        help=
        'Image size tuple e.g.: (500,500).',
        default='(256,256)')
    parser.add_argument(
        '--base-path',
        type=str,
        dest='BASE_PATH',
        help=
        'Directory where the OCT2017 is located (container of OCT2017).',
        default='.')
    parser.add_argument(
        '--test-fraction',
        type=float,
        dest='TEST_FRACTION',
        help=
        'Amount of data from total used as test set.',
        default=0.05)
    parser.add_argument(
        '--gpu',
        type=str,
        dest='gpu',
        help=
        'GPU where to run the model.')
    args = parser.parse_args()
    # print(args, end='\n\n')
    slist = args.IMAGE_SIZE.replace("(", "").replace(")", "").split(",")
    IMAGE_SIZE = tuple(int(s) for s in slist)
    obtain_stats(args.model_file_path, args.mendeley_csv_path, IMAGE_SIZE=IMAGE_SIZE, BASE_PATH=args.BASE_PATH, SPLIT_TEST_FRACTION=args.TEST_FRACTION, GPU=args.gpu)


if __name__ == "__main__":
    # print('holalala')
    # print(sys.argv)
    main(sys.argv)
