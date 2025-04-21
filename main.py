import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def one_hot_encoding(labels_dimension):
    indexed_labels = {label: idx for idx, label in enumerate(labels_dimension)} 
    one_hot_labels = []

    for label in indexed_labels:
        label_index = indexed_labels[label]
        row = [1 if j == label_index else 0 for j in range(len(indexed_labels))]
        one_hot_labels.append(row)

    return one_hot_labels

def normalize_image(image_matrix=[]):
    return np.array(image_matrix) / 255

class EMNISTHandler:
    def __init__(self, data_path, mapping_label_path):
        self.data_path = data_path
        self.mapping_path = mapping_label_path
        self.label_mapping = {}
        self.image_labels = []
        self.normalized_image = []
        self._load_data()
        self._load_mapping()

    def __str__(self):
        return (
            f"EMNISTHandler with {len(self.image_matrix)} images, "
            f"{len(self.label_mapping)} labels loaded from:\n"
            f"- Data: {self.data_path}\n"
            f"- Mapping: {self.mapping_path}"
        )

    def _load_data(self):
        data = loadmat(self.data_path)
        image_matrix = data['dataset'][0][0][0][0][0][0]
        self.image_labels = data['dataset'][0][0][0][0][0][1]
        self.normalized_image = normalize_image(image_matrix)

    def _load_mapping(self):
        with open(self.mapping_path, 'r') as f:
            for line in f:
                label, ascii_code = line.strip().split()
                self.label_mapping[int(label)] = chr(int(ascii_code))

    def display_sample_images(self, num_samples=5):
        for i in range(num_samples):
            original_matrix = self.normalized_image[i].reshape(28, 28)
            image = np.fliplr(np.fliplr(original_matrix.T))
            label = self.image_labels[i][0]
            plt.imshow(image, cmap='gray')
            plt.title(f'Label: {self.label_mapping[label]}')
            plt.axis('off')
            plt.show()

emnist = EMNISTHandler(data_path='./matlab/emnist-balanced.mat',mapping_label_path='./matlab/emnist-balanced-mapping.txt')
emnist.display_sample_images(5)