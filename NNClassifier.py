from sklearn.neighbors import KNeighborsClassifier
from utils import load_images_from_folders
import matplotlib.pyplot as plt
from eigenface import Eigenfaces
from fischerfaces import FischerFaces
import numpy as np
import cv2
import os

class Classifier:
    def __init__(self, m, algorithm, plot=False):

        assert algorithm == 'eigen' or algorithm == 'fischer', "Algorithm has to be either 'eigen' or 'fischer'"
        main_folder = "resources"
        folder_arman = main_folder + "/arman"
        folder_prof = main_folder + "/prof"
        folder_joao = main_folder + "/joao"
        folder_carlos = main_folder + "/carlos"
        self.__images = load_images_from_folders([folder_prof, folder_arman, folder_joao, folder_carlos])
        self.__images = np.reshape(self.__images,
                                   (self.__images.shape[0], self.__images.shape[1] * self.__images.shape[2]))

        self.__m = m
        self.__e = None
        self.__f = None
        j = np.ones(12)+1
        c = np.ones(10)+2
        self.__y = np.array([*np.zeros(9), *np.ones(13), *j, *c]).astype(int)
        self.__algoritm = algorithm

        if self.__algoritm == 'eigen':
            X = self.__run_eigenfaces()
        elif self.__algoritm == 'fischer':
            X = self.__run_fischerfaces()

        if plot:
            self.__plot(X)

        self.__train_classifier(X)

    def __run_fischerfaces(self):
        classes = ["prof", "arman", "joao", "carlos"]
        self.__f = FischerFaces(classes, self.__images, self.__y)
        self.__f.calculate(m=self.__m, debug=False)
        X = np.array([self.__f.get_vector(self.__images[i]) for i in range(len(self.__images))])

        return X

    def __train_classifier(self, X):
        self.__neigh = KNeighborsClassifier(n_neighbors=4)
        self.__neigh.fit(X, self.__y)

    def __plot(self, X):
        class_0 = X[self.__y == 0][:, :2]
        class_1 = X[self.__y == 1][:, :2]
        class_2 = X[self.__y == 2][:, :2]
        class_3 = X[self.__y == 3][:, :2]

        plt.title("Eigenface" if self.__algoritm == 'eigen' else "Fischerface")
        plt.plot(class_0[:,0], class_0[:,1], 'o')
        plt.plot(class_1[:, 0], class_1[:, 1], 'o')
        plt.plot(class_2[:, 0], class_2[:, 1], 'o')
        plt.plot(class_3[:, 0], class_3[:, 1], 'o')

        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        plt.show()

    def __run_eigenfaces(self):
        self.__e = Eigenfaces(self.__images)
        self.__e.calculate(m=self.__m, debug=False)
        X = np.array([self.__e.get_vector(self.__images[i]) for i in range(len(self.__images))])
        return X

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

if __name__ == '__main__':
    c = Classifier(20, 'eigen', plot=True)
