import cv2
import numpy as np
from utils import load_images_from_folder
from eigenface import Eigenfaces


class FischerFaces:

    def __init__(self, folder, classes):
        self.__classes = classes
        prof_images = load_images_from_folder(folder + self.__classes[0])
        arman_images = load_images_from_folder(folder + self.__classes[1])

        # Create dataset and labels
        self.__dataset = np.array([*prof_images, *arman_images])
        self.__labels = np.array([*np.zeros(len(prof_images)), *np.ones(len(arman_images))]).astype(int)

        self.__N = self.__dataset.shape[0]

        self.__w = self.__dataset.shape[1]
        self.__h = self.__dataset.shape[2]

        self.__dataset = np.reshape(self.__dataset, (self.__dataset.shape[0], self.__w * self.__h))
        print("Dataset shape:", self.__dataset.shape)
        print("Labels shape:", self.__labels.shape)

    def calculate(self):
        c = len(self.__classes)

        print("N =", self.__N)
        print("c =", c)

        mean_face = np.mean(self.__dataset, axis=0).astype(np.uint8)
        mean_faces = [np.mean(self.__dataset[self.__labels == i], axis=0).astype(np.uint8) for i in range(len(self.__classes))]

        St = np.zeros_like(self.__dataset[0])
        for i in range(len(self.__dataset)):
            aux = np.reshape(self.__dataset[i], (self.__w * self.__h, 1)) - np.reshape(mean_faces[self.__labels[i]], (self.__w * self.__h, 1))
            aux = np.dot(aux, aux.T)
            St = St + aux

        print("Scatter Matrix St:", St.shape)
        max_m = self.__N-c
        print("Maximum of m:", max_m)
        eigenfaces = Eigenfaces(self.__dataset)
        W = eigenfaces.calculate_eigenfaces(m=10, debug=True)

        ajuda1 = np.dot(W.T, St)
        ajuda = np.dot(ajuda1, W)
        print(ajuda)
        print("ARG MAX", ajuda.shape)
        print(np.argmax(ajuda))


        # Wpca = Eigenfaces()

        Sb = 0

folder = "resources/"
classes = ["prof", "arman"]
f = FischerFaces(folder, classes)
f.calculate()