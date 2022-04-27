import os
import cv2
import heapq
import numpy as np
import scipy
from scipy.interpolate import interp1d
from utils import load_images_from_folder


class Eigenfaces:

    @staticmethod
    def reshape_data(data):
        return np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))

    def __init__(self, images):
        self.__w = 56
        self.__h = 46
        self.__dataset = images
        print(self.__dataset.shape)

    def __calculate_mean_face(self, images, w, h, to_plot=False):
        mean_face = np.mean(images, axis=0).astype(np.uint8)
        cv2.imwrite("eigenfaces_output/mean_face.jpg", np.reshape(mean_face, (w, h)))
        if to_plot:
            cv2.imshow('Mean Face', np.reshape(mean_face, (self.__w, self.__h)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return mean_face

    def calculate_eigenfaces(self, m, debug=False):

        N = self.__dataset.shape[0]
        n = self.__dataset.shape[1]

        self.__mean_face = np.mean(self.__dataset, axis=0).astype(np.uint8)
        A = self.__dataset - self.__mean_face
        A = A.T
        if debug: print("A shape:", A.shape)

        R = np.dot(A.T, A)
        if debug: print("R shape:", R.shape)

        vw, V = np.linalg.eig(R)
        print("OLD V SHAPE", V.shape)

        indexes = np.argsort(vw)[::-1][:m]
        newV = np.array([])
        for i in V:
            newV = np.append(newV, np.array([i[j] for j in indexes]))

        V = np.reshape(newV, (N, m))
        if debug: print("V shape:", V.shape)

        self.__W = np.matmul(A, V)
        if debug: print("W shape:", self.__W.shape)
        print("W Nao Normalizado")
        print(self.__W)
        print(self.__W.shape)
        self.__W = np.array([w / np.linalg.norm(w) for w in self.__W])

        print(self.__W.shape)
        print(np.linalg.norm(self.__W, axis=0))
        # print("W Normalizado")
        # print(self.__W)
        # print("W Transposto")
        # print(self.__W.T)

        print("MATRIZ IDENTIDADE")
        print("dot(W.T, W)")
        help = np.dot(self.__W.T, self.__W)
        print(help[0][1])
        print(help[1][2])
        # for i in help:
        #     print(i)

        return self.__W

    def display_eigenfaces(self):
        print(self.__W.shape)
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

    def get_vector(self):
        y = np.dot(self.__W.T, (self.__dataset[0] - self.__mean_face))
        print(y.shape)
        return y


if __name__ == "__main__":
    folder = "resources/arman"
    images = load_images_from_folder(folder)

    e = Eigenfaces(Eigenfaces.reshape_data(images))
    Wpca = e.calculate_eigenfaces(5, debug=True)
    # e.display_eigenfaces()
    # print(e.get_vector())
