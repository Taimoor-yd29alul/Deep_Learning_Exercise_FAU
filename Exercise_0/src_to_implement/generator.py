import math
import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self._file_path = file_path
        self._label_path = label_path
        self._batch_size = batch_size
        self._image_size = image_size
        self._rotation = rotation
        self._mirroring = mirroring
        self._shuffle = shuffle

        self.current_epoch_number = 0
        self.batch_number = 0

        self.image_names = [filename for filename in os.listdir(self._file_path)]
        with open(self._label_path, 'r') as json_file:  # storing images labels dictionary
            self.labels_dictionary = json.load(json_file)

        if self._shuffle:
            self.shuffle_images()

        # print(self.labels_dictionary)
        #print(self.image_names)

    def shuffle_images(self):
        np.random.shuffle(self.image_names)

    def next(self):
        images_batch = []
        labels_array = []

        start = self.batch_number * self._batch_size
        end = start + self._batch_size

        if ((self.batch_number * self._batch_size) + self._batch_size) > len(self.image_names):
            if self._shuffle:
                self.shuffle_images()

            self.current_epoch_number += 1
            self.batch_number = 0
        else:
            self.batch_number += 1

        # print("Epoch number: ", self.current_epoch_number)
        # print("Batch number: ", self.batch_number)
        # print("Start number: ", start)
        # print("End number: ", end)
        # print("\n")

        for i in range(start, end):
            image_to_load_index = i % len(self.image_names)
            image_file_path = os.path.join(self._file_path, self.image_names[image_to_load_index])
            #print(image_file_path)

            img = np.load(image_file_path)
            img = skimage.transform.resize(img, (self._image_size[0], self._image_size[1]))

            img = self.augment(img)

            #print(img.shape)
            images_batch.append(img)

            image_key = self.image_names[image_to_load_index].split(".")[0]
            #print(image_key)
            labels_array.append(self.labels_dictionary.get(image_key))

        images_batch_array = np.array(images_batch)
        return images_batch_array, labels_array

    def augment(self, img):
        if self._mirroring:
            img = np.fliplr(img)

        if self._rotation:
            random_rotation = np.random.choice([90, 180, 270])
            img = scipy.ndimage.rotate(img, random_rotation, reshape=False)

        return img

    def current_epoch(self):
        return self.current_epoch_number

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict.get(x)

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()

        cols = math.ceil(math.sqrt(len(images)))  # Set columns to the ceiling of the square root of elements
        rows = math.ceil(len(images) / cols)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # Adjust figsize as needed
        current_img_index = 0

        for row in range(rows):
            for column in range(cols):
                if current_img_index < len(images):
                    ax = axes[row, column]
                    image_label = self.class_name(labels[current_img_index])
                    ax.set_title(image_label.capitalize())
                    ax.axis('off')
                    ax.imshow(images[current_img_index])

                else:
                    ax = axes[row, column]
                    ax.axis('off')
                current_img_index += 1

        plt.show()


image_generator = ImageGenerator('./exercise_data/', './Labels.json', 50, [50, 50, 3],
                                 False, False, True)
#image_generator.show()
#print(image_generator.current_epoch())
