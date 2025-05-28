import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from scipy.io import loadmat

def one_hot_encoding(labels, num_classes=47):
    one_hot_labels = []

    for label in labels:
        row = [1 if i == label[0] else 0 for i in range(num_classes)]
        one_hot_labels.append(row)

    return np.array(one_hot_labels)


def normalize_image(image_matrix): #Normalize image matrix with values between 0 and 1
    image = image_matrix.reshape(28, 28)
    image = image.T
    return np.array(image) / 255

class NeuralNetwork:
    def __init__(self, data_path, mapping_label_path): #Initialize variables and call load functions
        self.data_path = data_path
        self.mapping_path = mapping_label_path
        self.label_mapping = {}
        self.images_matrix = []
        self.images_labels = []
        self.shuffled_images = []
        self.shuffled_labels = []
        self._load_data()
        # self._load_mapping()

    def _load_data(self): #Load the dataset layer with images and labels
        data = loadmat(self.data_path)
        images_matrix = data['dataset'][0][0][0][0][0][0]
        images_labels = data['dataset'][0][0][0][0][0][1]
        self.images_matrix = images_matrix
        self.images_labels = images_labels

        self.shuffled_images, self.shuffled_labels = self._shuffle_data(images_matrix, images_labels)
        (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = self._split_data(self.shuffled_images, self.shuffled_labels)
        y_train_encoded = one_hot_encoding(train_labels)
        y_validation_encoded = one_hot_encoding(validation_labels)
        y_test_encoded = one_hot_encoding(test_labels)


    """
    def _load_mapping(self): #Load the mapping file creating a dictionary
        with open(self.mapping_path, 'r') as f:
            for line in f:
                label, ascii_code = line.strip().split()
                self.label_mapping[int(label)] = chr(int(ascii_code))
    """

    def _shuffle_data(self, images_matrix, images_labels):
        new_indexes = np.random.permutation(len(images_matrix))
        shuffled_images = images_matrix[new_indexes]
        shuffled_labels = images_labels[new_indexes]
        return np.array(shuffled_images), np.array(shuffled_labels)
    
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

        return (np.array(train_images), np.array(train_labels)), (np.array(validation_images), np.array(validation_labels)), (np.array(test_images), np.array(test_labels))


    def display_sample_images(self, num_samples=5): #A method to display image samples
        for i in range(num_samples):
            show_image = normalize_image(image_matrix=self.images_matrix[i])
            label = self.images_labels[i][0]
            plt.imshow(show_image, cmap='gray')
            plt.title(f'Label: {self.label_mapping[label]}')
            plt.axis('off')
            plt.show()

    def build_model(self):
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(47, activation='softmax')
        ])

emnist = NeuralNetwork(data_path='./matlab/emnist-balanced.mat',mapping_label_path='./matlab/emnist-balanced-mapping.txt')