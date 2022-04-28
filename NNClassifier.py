from sklearn.neighbors import KNeighborsClassifier
from utils import load_images_from_folders
from eigenface import Eigenfaces
from fischerfaces import FischerFaces
import numpy as np
import cv2
import os


class Classifier:
    def __init__(self, m, algorithm):
        folder_arman = "resources/arman"
        folder_prof = "resources/prof"
        self.__images = load_images_from_folders([folder_arman, folder_prof])
        self.__images = np.reshape(self.__images,
                                   (self.__images.shape[0], self.__images.shape[1] * self.__images.shape[2]))

        self.__m = m
        self.__e = None
        self.__f = None

        self.__y = np.array([*np.ones(13), *np.zeros(9)]).astype(int)
        self.__algoritm = algorithm
        if self.__algoritm == 'eigen':
            self.__run_eigenfaces()
        elif self.__algoritm == 'fischer':
            self.__run_fischerfaces()

    def __run_fischerfaces(self):
        classes = ["prof", "arman"]
        self.__f = FischerFaces(classes, self.__images, self.__y)
        self.__f.calculate(m=self.__m, debug=False)
        X = np.array([self.__f.get_vector(self.__images[i]) for i in range(len(self.__images))])
        self.__train_classifier(X)

    def __train_classifier(self, X):
        self.__neigh = KNeighborsClassifier(n_neighbors=3)
        self.__neigh.fit(X, self.__y)

    def __run_eigenfaces(self):
        self.__e = Eigenfaces(self.__images)
        self.__e.calculate(m=self.__m, debug=False)
        X = np.array([self.__e.get_vector(self.__images[i]) for i in range(len(self.__images))])
        self.__train_classifier(X)

    def get_vector(self, new_image):
        if self.__algoritm == 'eigen':
            return self.__e.get_vector(new_image)
        elif self.__algoritm == 'fischer':
            return self.__f.get_vector(new_image)

    def predict(self, new_vector):
        predicted = self.__neigh.predict([new_vector])
        if predicted == [1]:
            print("Predicting... Arman")
            return "Arman"
        elif predicted == [0]:
            print("Predicting... Prof")
            return "Prof"
