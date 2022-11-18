import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        self.ids = (213635899, 325324994)

        self.max_label = None
        self.train_points = np.ndarray((1, 1))
        self.train_labels = np.ndarray((1, 1))
        self.reversed_labels_list = np.ndarray((1, 1))

    def minkowski_distance(self, a, b):
        return sum(abs(a - b) ** self.p) ** (1 / self.p)


    def help_f(self , point):
        distances = np.array([self.minkowski_distance(point, train_point) for train_point in self.train_points])
        s = np.array(list(zip(distances, self.train_labels)))
        ordered_neighbors = sorted(s, key=lambda x: (x[0], x[1]))
        return self.find_label(ordered_neighbors)


    def find_label(self, ordered_neighbors: np.ndarray):
        k_neighbors = ordered_neighbors[0:self.k]
        k_labels = [neighbor[1] for neighbor in k_neighbors]
        count_of_labels = np.bincount(k_labels, minlength=self.max_label + 1)
        s = np.array(list(zip(self.reversed_labels_list, count_of_labels)))

        ordered_reversed_labels_count = sorted(s, key=lambda x: (x[1], x[0]), reverse=True)
        return self.max_label - ordered_reversed_labels_count[0][0]




    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        self.train_points = X
        self.train_labels = y
        self.max_label = np.max(self.train_labels)

        self.reversed_labels_list = np.flip(np.array(range(0, self.max_label + 1)))



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        predicted_labels = [self.help_f(point) for point in X]
        return np.array(predicted_labels)




def main():
    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    # k = 1
    # p = 2
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    # X = np.array([[0, 0], [1, 2], [3, 4], [5, 0]])
    # y = np.array([0, 0, 1, 1])

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
