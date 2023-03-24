# Read in label file and return a dictionary {'filename' : label}.
#

import tensorflow as tf

import os
import numpy as np

def import_labels(label_file):
    labels = dict()

    import csv
    with open(label_file) as fd:
        csvreader = csv.DictReader(fd)

        for row in csvreader:
            labels[row['filename']] = int(row['label'])
    return labels

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_root_dir, labels_dict, batch_size, target_dim, preprocess_func=None, use_augmentation=True):
        self._labels_dict = labels_dict
        self._img_root_dir = img_root_dir
        self._batch_size = batch_size
        self._target_dim = target_dim
        self._preprocess_func = preprocess_func
        self._n_classes = len(set(self._labels_dict.values()))
        self._fnames_all = list(self._labels_dict.keys())
        self._use_augmentation = use_augmentation

        if self._use_augmentation:
            self._augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self._fnames_all)) / self._batch_size)

    def on_epoch_end(self):
        self._indices = np.arange(len(self._fnames_all))
        np.random.shuffle(self._indices)

    def __getitem__(self, index):
        indices = self._indices[index * self._batch_size:(index+1)*self._batch_size]

        fnames = [self._fnames_all[k] for k in indices]
        X,Y = self.__load_files__(fnames)

        return X,Y

    def __load_files__(self, batch_filenames):
        X = np.empty((self._batch_size, *self._target_dim, 3))
        Y = np.empty((self._batch_size), dtype=int)

        for idx, fname in enumerate(batch_filenames):
            img_path = os.path.join(self._img_root_dir, fname)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self._target_dim)
            x = tf.keras.preprocessing.image.img_to_array(img)

            if self._preprocess_func is not None:
                x = self._preprocess_func(x)

            X[idx,:] = x
            Y[idx] = self._labels_dict[fname]-1

        if self._use_augmentation:
            it = self._augmentor.flow(X, batch_size=self._batch_size, shuffle=False)
            X = it.next()


        return X, tf.keras.utils.to_categorical(Y, num_classes=self._n_classes)


if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    train_dict = import_labels(LABELS_TRAIN)
    test_dict = import_labels(LABELS_TEST)

    INPUT_SHAPE = (100, 100, 3)
    print(len(test_dict))
    for x, y in DataGenerator(img_root_dir="data/test",
                                                  labels_dict=test_dict,
                                                  batch_size=50,
                                                  target_dim=INPUT_SHAPE[:2],
                                                  preprocess_func=None,
                                                  use_augmentation=False):
        print(len(x))