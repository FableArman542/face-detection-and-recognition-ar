import cv2
import numpy as np
from eigenface import Eigenfaces
import matplotlib.pyplot as plt
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
            aux = np.reshape(self.__dataset[i], (self.__w * self.__h, 1)) - np.reshape(mean_faces[self.__labels[i]], (self.__w * self.__h, 1))
            aux = np.dot(aux, aux.T)
            Sw = Sw + aux

        return Sw

    def __calculate_sb(self, mean_face, mean_faces):
        Sb = np.zeros(self.__w * self.__h, dtype = np.float64)
        for x in range(len(self.__dataset)):
            ni = np.sum(self.__labels == x)
            mean_diff = np.reshape(self.__mean_faces[self.__labels[x]], (self.__w * self.__h, 1)) - np.reshape(self.__mean_face, (self.__w * self.__h, 1))
            mean_diff = ni * np.dot(mean_diff, mean_diff.T)
            Sb = Sb + mean_diff

        return Sb

    def calculate(self, m, debug=False):
        c = len(self.__classes)
        max_m = self.__N - c
        assert m <= max_m, "m has to be less than " + str(max_m)

        if debug:
            print("N =", self.__N)
            print("c =", c)
            print("Maximum of m:", max_m)

        self.__mean_face = np.mean(self.__dataset, axis=0)
        self.__mean_faces = [np.mean(self.__dataset[self.__labels == i], axis=0) for i in range(len(self.__classes))]

        print("Mean faces", np.array(self.__mean_faces).shape)

        # Sw = self.__calculate_sw(self.__mean_faces)

        Sw = np.zeros(56 * 46, dtype=np.float64)
        for x in range(len(self.__dataset)):
            var_temp = np.reshape(self.__dataset[x], (56 * 46, 1)) - np.reshape(self.__mean_faces[int(self.__labels[x])], (56 * 46, 1))
            Sw = Sw + np.dot(var_temp, var_temp.T)

        if debug: print("Sw shape:", Sw.shape)

        Sb = self.__calculate_sb(self.__mean_face, self.__mean_faces)
        if debug: print("Sb shape:", Sb.shape)

        St = Sw + Sb
        if debug: print("St shape:", St.shape)

        eigenfaces = Eigenfaces(self.__dataset)
        Wpca = eigenfaces.calculate(m=m, debug=False)

        new_sb = np.dot(np.dot(Wpca.T, Sb), Wpca)
        new_sw = np.dot(np.dot(Wpca.T, Sw), Wpca)

        if debug:
            print("W eigen:", Wpca.shape)
            print("New Sb shape:", new_sb.shape)
            print("New Sw shape:", new_sw.shape)

        matrix = np.dot(np.linalg.inv(new_sw), new_sb)

        w, v = np.linalg.eig(matrix)
        indexes = np.argsort(w)[::-1][:m]
        Wfld = v[:, indexes]
        if debug: print("Wfld shape:", Wfld.shape)

        self.__W = np.dot(Wpca, Wfld)

        if debug:
            print("W shape:", self.__W.shape)

    def get_vector(self, image):
        assert self.__W is not None, "use calculate() first"
        assert self.__mean_face is not None, "use calculate() first"

        y = np.dot(self.__W.T, (image - self.__mean_face))
        return np.real(y)

    def reconstruct(self, vector):
        image = np.dot(self.__W, vector) + self.__mean_face
        return np.real(image)

    def plot_mean_faces(self):
        plt.title('Mean Face')
        plt.imshow(np.reshape(self.__mean_face, (self.__w, self.__h)), cmap='gray')
        plt.show()
        for mean in self.__mean_faces:
            plt.imshow(np.reshape(mean, (self.__w, self.__h)), cmap='gray')
            plt.show()


if __name__ == "__main__":
    classes = ["prof", "arman", "joao", "carlos"]

    main_folder = "resources"
    folder_arman = main_folder + "/arman"
    folder_prof = main_folder + "/prof"
    folder_joao = main_folder + "/joao"
    folder_carlos = main_folder + "/carlos"
    images = load_images_from_folders([folder_arman, folder_prof, folder_joao, folder_carlos])
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))

    j = np.ones(12) + 1
    c = np.ones(10) + 2
    labels = np.array([*np.zeros(9), *np.ones(13), *j, *c]).astype(int)

    f = FischerFaces(classes, images, labels)
    f.calculate(m=10, debug=True)
    f.plot_mean_faces()

    # cv2.imshow('Image', np.reshape(images[0], (56, 46)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    vector = f.get_vector(images[0])
    reconstructed_image = f.reconstruct(vector)
    reconstructed_image = np.reshape(reconstructed_image, (56, 46))

    error_face = images[0].reshape(56, 46) - reconstructed_image

    # plt.imshow(reconstructed_image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(error_face, cmap='gray')
    # plt.axis('off')
    # plt.show()
