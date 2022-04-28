import cv2
import numpy as np
from eigenface import Eigenfaces
from utils import load_images_from_folder, load_images_from_folders


class FischerFaces:

    def __init__(self, classes, images, labels):
        self.__classes = classes

        self.__w = 56
        self.__h = 46
        self.__mean_face = None

        self.__dataset = images
        self.__labels = labels

        self.__N = self.__dataset.shape[0]

        self.__W = None

    def __calculate_sw(self, mean_faces):
        Sw = np.zeros_like(self.__dataset[0])
        for i in range(len(self.__dataset)):
            aux = np.reshape(self.__dataset[i], (self.__w * self.__h, 1)) - np.reshape(mean_faces[self.__labels[i]],
                                                                                       (self.__w * self.__h, 1))
            aux = np.dot(aux, aux.T)
            Sw = Sw + aux
        return Sw

    def __calculate_sb(self, mean_face, mean_faces):
        Sb = None

        for i in range(len(self.__classes)):
            ni = np.sum(self.__labels == i)
            mean_diff = (np.reshape(mean_faces[i], (self.__w * self.__h, 1)) - np.reshape(mean_face,
                                                                                          (self.__w * self.__h, 1)))
            mean_diff = ni * np.dot(mean_diff, mean_diff.T)
            if Sb is None:
                Sb = mean_diff
            else:
                Sb = Sb + mean_diff

        return Sb

    def calculate(self, m, debug=False):
        c = len(self.__classes)
        max_m = self.__N - c

        if debug:
            print("N =", self.__N)
            print("c =", c)
            print("Maximum of m:", max_m)

        self.__mean_face = np.mean(self.__dataset, axis=0).astype(np.uint8)
        mean_faces = [np.mean(self.__dataset[self.__labels == i], axis=0).astype(np.uint8) for i in
                      range(len(self.__classes))]

        Sw = self.__calculate_sw(mean_faces)
        if debug: print("Sw shape:", Sw.shape)

        Sb = self.__calculate_sb(self.__mean_face, mean_faces)
        if debug: print("Sb shape:", Sb.shape)

        St = Sw + Sb
        if debug: print("St shape:", St.shape)

        eigenfaces = Eigenfaces(self.__dataset)
        Wpca = eigenfaces.calculate(m=m, debug=False)

        new_sb = np.dot(np.dot(Wpca.T, Sb), Wpca)
        new_sw = np.dot(np.dot(Wpca.T, Sw), Wpca)

        if debug:
            print("New Sb shape:", new_sb.shape)
            print("New Sw shape:", new_sw.shape)

        matrix = np.dot(np.linalg.inv(new_sb), new_sb)

        w, v = np.linalg.eig(matrix)
        indexes = np.argsort(w)[::-1][:m]
        newV = np.array([])
        for i in v:
            newV = np.append(newV, np.array([i[j] for j in indexes]))
        Wfld = np.reshape(newV, (v.shape[0], m))
        if debug: print("Wfld shape:", Wfld.shape)

        self.__W = np.dot(Wpca, Wfld)

        if debug:
            print("W shape:", self.__W.shape)

    def get_vector(self, image):
        assert self.__W is not None, "use calculate() first"
        assert self.__mean_face is not None, "use calculate() first"

        y = np.dot(self.__W.T, (image - self.__mean_face))
        return np.real(y)


if __name__ == "__main__":
    classes = ["prof", "arman"]

    folder_arman = "resources/arman"
    folder_prof = "resources/prof"
    images = load_images_from_folders([folder_arman, folder_prof])
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

    labels = np.array([*np.zeros(10), *np.ones(9)]).astype(int)

    f = FischerFaces(classes, images, labels)
    f.calculate(m=10, debug=True)
    ajuda = f.get_vector(images[0])
    print(ajuda)
