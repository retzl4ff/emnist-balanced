import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from scipy.io import loadmat

def one_hot_encoding(labels_dimension):
    indexed_labels = {label: idx for idx, label in enumerate(labels_dimension)} 
    one_hot_labels = []

    for label in indexed_labels:
        label_index = indexed_labels[label]
        row = [1 if j == label_index else 0 for j in range(len(indexed_labels))]
        one_hot_labels.append(row)

    return one_hot_labels

def normalize_image(image_matrix=[]): #Normalize image matrix with values between 0 and 1
    image = image_matrix.reshape(28, 28)
    image = np.fliplr(np.fliplr(image.T))
    return np.array(image) / 255

class EMNISTHandler:
    def __init__(self, data_path, mapping_label_path): #Initialize variables and call load functions
        self.data_path = data_path
        self.mapping_path = mapping_label_path
        self.label_mapping = {}
        self.images_matrix = []
        self.images_labels = []
        self.shuffled_images = []
        self.shuffled_labels = []
        self._load_data()
        self._load_mapping()

    def _load_data(self): #Load the dataset layer with images and labels
        data = loadmat(self.data_path)
        images_matrix = data['dataset'][0][0][0][0][0][0]
        images_labels = data['dataset'][0][0][0][0][0][1]
        self.images_matrix = images_matrix
        self.images_labels = images_labels

        self.shuffled_images, self.shuffled_labels = self._shuffle_data(images_matrix, images_labels)
        (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = self._split_data(self.shuffled_images, self.shuffled_labels)


    def _load_mapping(self): #Load the mapping file creating a dictionary
        with open(self.mapping_path, 'r') as f:
            for line in f:
                label, ascii_code = line.strip().split()
                self.label_mapping[int(label)] = chr(int(ascii_code))

    def _shuffle_data(self, images_matrix, images_labels):
        new_indexes = np.random.permutation(len(images_matrix))
        shuffled_images = images_matrix[new_indexes]
        shuffled_labels = images_labels[new_indexes]
        return shuffled_images, shuffled_labels
    
    def _split_data(self, shuffled_images, shuffled_labels, train_size=0.8, validation_size=0.1):
        total_size = len(shuffled_images)

        #train and validation indexes
        train_end = int(total_size * train_size)
        validation_end = train_end + int(total_size * validation_size)

        #slicing train, test and validation data
        train_images = shuffled_images[:train_end]
        train_labels = shuffled_labels[:train_end]

        validation_images = shuffled_images[train_end:validation_end]
        validation_labels = shuffled_labels[train_end:validation_end]

        test_images = shuffled_images[validation_end:]
        test_labels = shuffled_labels[validation_end:]

        return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)


    def display_sample_images(self, num_samples=5): #A method to display image samples
        for i in range(num_samples):
            show_image = normalize_image(image_matrix=self.images_matrix[i])
            label = self.images_labels[i][0]
            plt.imshow(show_image, cmap='gray')
            plt.title(f'Label: {self.label_mapping[label]}')
            plt.axis('off')
            plt.show()

emnist = EMNISTHandler(data_path='./matlab/emnist-balanced.mat',mapping_label_path='./matlab/emnist-balanced-mapping.txt')