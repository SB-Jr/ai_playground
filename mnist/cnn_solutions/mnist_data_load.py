import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


def train_test_validation_split(dataframe: pd.DataFrame, train_ratio=0.8, validation_ratio=0.1, seed=100):
    """
    This function returns an equal weighted training and validation data
    All the labels will have same number of examples
    The remaining is put into training data
    Data is shuffled before being segregated into traina, validation and test set
    Data is shuffled after segreagation also
    
    Arguments:
        dataframe: the dataframe that needs to be segregated
        train_ratio: a floating number less than 1
        validation_ratio: a floating number less than 1
        seed: to set the random generator's seed to reproduce results
    
    Return:
        train_data: Dataframe contianing training labels and features
        validation_data: Dataframe containing validation labels and features
        test_data: Dataframe containing testing lables and features
    """
    assert train_ratio + validation_ratio < 1.0
    data = dataframe.copy()
    np.random.seed(seed)
    # basic info gathering to aid partitioning
    labels = data.label.unique()
    min_item_per_label = data.groupby('label').label.count().min()
    # defining the indexes
    train_index = int(min_item_per_label * train_ratio)
    validation_index = int(min_item_per_label * validation_ratio + train_index)
    # randomize the data
    data.sample(frac=1).reset_index(drop=True)
    # sample data
    train_data = pd.DataFrame()
    validation_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for l in labels:
        logging.info('handling label: '+ str(l))
        train_data = train_data.append(data[data.label == l].iloc[:train_index])
        validation_data = validation_data.append(data[data.label == l].iloc[train_index: validation_index])
        test_data = test_data.append(data[data.label == l].iloc[validation_index:])
    # shuffling the data
    train_data.sample(frac=1).reset_index(drop=True)
    test_data.sample(frac=1).reset_index(drop=True)
    validation_data.sample(frac=1).reset_index(drop=True)
    # validating that all the set contains all the labels
    assert len(train_data.label.unique()) == len(labels)
    assert len(test_data.label.unique()) == len(labels)
    assert len(validation_data.label.unique()) == len(labels)
    
    return train_data, validation_data, test_data


def load_data(train_ratio=0.8, validation_ratio=0.1, seed=100) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    loads the MNIST dataset and provides a train, validation and test split
    
    Arguments:
        train_ratio: the ratio of the whole data that has to be reserved for training
        validation_ratio: the ratio of the whole data that has to be reserved for validation
    """
    data_path = '../dataset/train.csv'
    data = pd.read_csv(data_path)
    return train_test_validation_split(data, train_ratio, validation_ratio, seed)


def plot_sample_data(dataset: pd.DataFrame):
    """
    Plots 16 randomly selected MNIST dataset's values in a 4x4 format
    
    Argument
        dataset: a dataframe containing 784 columns for each pixel and the 1st column being the label
    """
    for i in range(16):
        index = random.randint(0, dataset.shape[0])
        subplot = plt.subplot(4, 4, i+1)
        data = dataset.iloc[index]
        subplot.set_title(data.label)
        subplot.imshow(data[1:].values.reshape((28, 28)), cmap='gray')
    plt.show()
    
    
def get_label_and_features_from_dataframe(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Provides Numpy NDArray values for labels and MNIST dataset for model training, validation and testing
    Arguments:
        df: A dataframe whose 1st column is the label and has 784 other columns for each pixel value
    Returns:
        x: Numpy.ndarray containing 2D vector of shape 28x28 for each label
        y: Numpy.ndarray containing class label corresponding to each element in x
    """
    y = df.label.values
    x = df[df.columns[1:]].values.reshape((-1, 28, 28, 1)).astype(np.float)
    return x, y