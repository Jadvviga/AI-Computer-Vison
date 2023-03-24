import tensorflow as tf

import os
import numpy as np
import math


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir_path,
                 labels_file_path,
                 batch_size=25,
                 target_dim=(224, 224),
                 img_preprocessing_func=None,
                 arr_preprocessing_func=None,
                 do_augmentation=False,
                 do_shuffling=False):
        self.img_dir_path = img_dir_path
        self.labels_dict = self.import_labels(labels_file_path)
        self.filenames_list = list(self.labels_dict.keys())

        self.batch_size = batch_size
        self.num_of_batches = math.ceil(len(self.labels_dict) / self.batch_size)

        self.target_dim = target_dim
        self.n_classes = len(set(self.labels_dict.values()))

        self.arr_preprocessing_func = arr_preprocessing_func
        self.img_preprocessing_func = img_preprocessing_func
        self.do_augmentation = do_augmentation
        self.do_shuffling = do_shuffling

        self.augmentator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def __len__(self):
        return self.num_of_batches

    def __getitem__(self, batch_index):
        if batch_index >= self.num_of_batches or batch_index < 0:
            raise IndexError(f"Provided index for batch {batch_index}, max batch index is {self.num_of_batches - 1}")

        X, Y = self.load_files(batch_index)

        if self.do_augmentation:
            it = self.augmentator.flow(X, batch_size=len(X), shuffle=False)
            X = it.next()

        if self.do_shuffling:
            permutation = np.random.permutation(len(X))
            X, Y = X[permutation], Y[permutation]

        return X, Y

    @staticmethod
    def import_labels(label_file):
        labels = dict()

        import csv
        with open(label_file) as fd:
            csvreader = csv.DictReader(fd)

            for row in csvreader:
                labels[row['filename']] = int(row['label'])
        return labels

    def load_files(self, batch_index):
        # assumes correct batch index
        # calculate batch size here
        if batch_index == self.num_of_batches - 1:
            batch_size = len(self.labels_dict) % self.batch_size
        else:
            batch_size = self.batch_size

        X = np.empty((batch_size, *self.target_dim, 3))
        Y = np.empty(batch_size, dtype=int)

        batch_filenames = self.filenames_list[self.batch_size * batch_index:
                                              self.batch_size * batch_index + batch_size]
        for idx, fname in enumerate(batch_filenames):
            img_path = os.path.join(self.img_dir_path, fname)
            img = tf.keras.preprocessing.image.load_img(img_path)

            if self.img_preprocessing_func:
                img = self.img_preprocessing_func(img)

            x = tf.keras.preprocessing.image.img_to_array(img)
            x = tf.keras.preprocessing.image.smart_resize(x, size=self.target_dim)

            if self.arr_preprocessing_func:
                x = self.arr_preprocessing_func(x)

            X[idx, :] = x
            Y[idx] = self.labels_dict[fname] - 1

        return X, tf.keras.utils.to_categorical(Y, num_classes=self.n_classes)


if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    for x, y in DataGenerator(IMAGES_TEST_PATH, LABELS_TEST):
        print(len(x), len(y))
