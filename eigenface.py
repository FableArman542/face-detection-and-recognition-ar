import os
import cv2
import heapq
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.asarray(images)

def calculate_mean_face(images, w, h, to_plot=False):
    mean_face = np.mean(images, axis=0).astype(np.uint8)
    if to_plot:
        cv2.imshow('Mean Face', mean_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return np.reshape(mean_face, (w * h))

images = load_images_from_folder("resources")
w = images.shape[1]
h = images.shape[2]
a = [[2, 3, 4],
     [1, 2, 3]]

dataset = np.reshape(images, (images.shape[0], w * h))
N=dataset.shape[0]
n = dataset.shape[1]

mean_face = calculate_mean_face(images, w, h)

A = np.array([face - mean_face for face in dataset])
A = np.reshape(A, (n, N))
print("A shape:", A.shape)

m = 5
R = np.dot(A.T, A)
print("R shape:", R.shape)
S = np.dot(A, A.T)
print("S shape:", S.shape)

R_mag = np.array([np.linalg.norm(v) for v in R])

# indexes = heapq.nlargest(m, range(len(R_mag)), R_mag.__getitem__)
# V = np.array([R[i] for i in indexes])
# V = np.reshape(V, (N, m))
_, V = np.linalg.eig(R)

V_mag = np.array([np.linalg.norm(v) for v in V])
indexes = heapq.nlargest(m, range(len(V)), V_mag.__getitem__)
V = np.array([V[i] for i in indexes])
V = np.reshape(V, (N, m))
print("V shape:", V.shape)

W = np.dot(A, V)
W = np.array([w/np.linalg.norm(w) for w in W])

print("W shape:", W.shape)

y = W.T*(dataset[0] - mean_face)

for i in y:
    cv2.imshow('Mean Face', np.reshape(i, (w, h)).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print()
print(y)
print("Y shape:", y.shape)
