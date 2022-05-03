import os
import cv2
import heapq
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils import load_images_from_folder, load_images_from_folders


class Eigenfaces:

    def __init__(self, images):
        self.__w = 56
        self.__h = 46
        self.__dataset = images

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
        V = V[:, indexes]
        if debug: print("V shape:", V.shape)

        self.__W = np.dot(A, V)
        if debug: print("W shape:", self.__W.shape)
        self.__W = self.__W/np.linalg.norm(self.__W, axis=0)

        return self.__W

    def display_eigenfaces(self):
        plt.title("Mean Face")
        plt.imshow(np.reshape(self.__mean_face, (self.__w, self.__h)), cmap='gray')
        plt.show()
        for i in range(self.__W.shape[1]):
            aux = self.__W[:, i]
            maximum = max(aux)
            minimum = min(aux)
            m = interp1d([minimum, maximum], [0, 255])
            for j in range(len(aux)):
                aux[j] = m(aux[j])
            plt.imshow(np.reshape(aux, (self.__w, self.__h)), cmap='gray')
            plt.show()
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def get_vector(self, image):
        y = np.dot(self.__W.T, (image - self.__mean_face))
        return y

    def reconstruct(self, vector):
        image = np.dot(self.__W, vector) + self.__mean_face
        return image


if __name__ == "__main__":

    main_folder = "resources"
    folder_arman = main_folder + "/arman"
    folder_prof = main_folder + "/prof"
    folder_joao = main_folder + "/joao"
    folder_carlos = main_folder + "/carlos"

    images = load_images_from_folders([folder_arman, folder_prof, folder_joao, folder_carlos])
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    print(images.shape)
    e = Eigenfaces(images)
    W = e.calculate(20, debug=True)

    # print("Multiplicação de W transposto com W")
    # print(np.dot(W.T, W))

    # e.display_eigenfaces()

    vector = e.get_vector(images[0])
    reconstructed_image = e.reconstruct(vector)
    reconstructed_image = np.reshape(reconstructed_image, (56, 46))
    error_face = images[0].reshape(56, 46) - reconstructed_image

    # print("Multiplicação do face de erro com a face original")
    # print(np.dot(np.reshape(error_face, (56*46)), np.reshape(images[0], (56*46))))

    # plt.imshow(images[0].reshape(56, 46), cmap='gray')
    # plt.axis('off')
    # plt.show()
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(error_face, cmap='gray')
    plt.axis('off')
    plt.show()

    # cv2.imshow('asd', reconstructed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('Reconstructed Error', error_face)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

