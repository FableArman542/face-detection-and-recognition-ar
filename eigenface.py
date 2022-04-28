import os
import cv2
import heapq
import numpy as np
import scipy
from scipy.interpolate import interp1d
from utils import load_images_from_folder


class Eigenfaces:

    def __init__(self, images):
        self.__w = 56
        self.__h = 46
        self.__dataset = images

    def __calculate_mean_face(self, images, w, h, to_plot=False):
        mean_face = np.mean(images, axis=0).astype(np.uint8)
        cv2.imwrite("eigenfaces_output/mean_face.jpg", np.reshape(mean_face, (w, h)))
        if to_plot:
            cv2.imshow('Mean Face', np.reshape(mean_face, (self.__w, self.__h)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return mean_face

    def calculate(self, m, debug=False):

        N = self.__dataset.shape[0]
        n = self.__dataset.shape[1]

        self.__mean_face = np.mean(self.__dataset, axis=0)
        A = self.__dataset - self.__mean_face
        A = A.T

        if debug: print("A shape:", A.shape)

        R = np.dot(A.T, A)
        if debug: print("R shape:", R.shape)

        vw, V = np.linalg.eig(R)

        assert m < N-1, "m has to be less than N-1"

        indexes = np.argsort(vw)[::-1][:m]
        newV = np.array([])
        for i in V:
            newV = np.append(newV, np.array([i[j] for j in indexes]))

        V = np.reshape(newV, (N, m))
        if debug: print("V shape:", V.shape)

        self.__W = np.matmul(A, V)
        if debug: print("W shape:", self.__W.shape)
        self.__W = self.__W/np.linalg.norm(self.__W, axis=0)

        return self.__W

    def display_eigenfaces(self):
        cv2.imshow('Mean Face', np.reshape(self.__mean_face, (self.__w, self.__h)))

        for i in range(self.__W.shape[1]):
            ajuda = self.__W[:, i]
            maximum = max(ajuda)
            minimum = min(ajuda)
            m = interp1d([minimum, maximum], [0, 255])
            for j in range(len(ajuda)):
                ajuda[j] = m(ajuda[j])
            cv2.imshow('Test', np.reshape(ajuda, (self.__w, self.__h)).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_vector(self, image):
        y = np.dot(self.__W.T, (image - self.__mean_face))
        return y


if __name__ == "__main__":
    folder = "resources/prof"
    images = load_images_from_folder(folder)
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

    e = Eigenfaces(images)
    W = e.calculate(5, debug=True)

    print("MATRIZ IDENTIDADE")
    print("dot(W.T, W)")
    print(np.dot(W.T, W))

    # e.display_eigenfaces()
