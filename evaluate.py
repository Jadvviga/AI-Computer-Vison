import data_gen_new
import utils
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    MODELS_PATH = "models"

    model_filename = utils.choose_file_to_load(MODELS_PATH)

    input_shape = utils.model_filename_parse_dimension(model_filename)

    labels_train = utils.import_labels(LABELS_TRAIN)
    labels_test = utils.import_labels(LABELS_TEST)

    data_generator_train = data_gen_new.DataGenerator(img_dir_path=IMAGES_TRAIN_PATH,
                                                      labels_dict=labels_train,
                                                      batch_size=25,
                                                      target_dim=input_shape,
                                                      arr_preprocessing_func=None,
                                                      img_preprocessing_func=None,
                                                      do_shuffling=False,
                                                      do_augmentation=False)

    data_generator_test = data_gen_new.DataGenerator(img_dir_path=IMAGES_TEST_PATH,
                                                     labels_dict=labels_test,
                                                     batch_size=25,
                                                     target_dim=input_shape,
                                                     arr_preprocessing_func=None,
                                                     img_preprocessing_func=None,
                                                     do_shuffling=False,
                                                     do_augmentation=False)

    model = tf.keras.models.load_model(filepath=os.path.join(MODELS_PATH, model_filename))
    model.evaluate(x=data_generator_train, verbose=2)

    y_predict = model.predict(x=data_generator_test)
    y_predict = tf.argmax(input=y_predict, axis=1)

    # classification report
    target_names = []
    for i in range(1, 103):
        target_names.append(f"flower_{i}")

    print(list(labels_test.values()))
    x = list(np.array(y_predict))
    print(x)
    # for some fuking reason it eauther doent work beacous of
    # b. even if i put porper #of batches, its throws error that it read 103 classes instead of 102
    report = classification_report(y_true=list(labels_test.values()), y_pred=y_predict, target_names=target_names)

    print(report)
