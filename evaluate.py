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

    test_labels = data_gen_new.DataGenerator.import_labels(LABELS_TEST)
    train_labels = data_gen_new.DataGenerator.import_labels(LABELS_TRAIN)
    

    print('here1')

    data_generator_train = data_gen_new.DataGenerator(img_dir_path=IMAGES_TRAIN_PATH,
                                                      labels_file_path=LABELS_TRAIN,
                                                      batch_size=25,
                                                      target_dim=input_shape,
                                                      arr_preprocessing_func=None,
                                                      img_preprocessing_func=None,
                                                      do_shuffling=False,
                                                      do_augmentation=False)

    data_generator_test = data_gen_new.DataGenerator(img_dir_path=IMAGES_TEST_PATH,
                                                     labels_file_path=LABELS_TEST,
                                                     batch_size=25,
                                                     target_dim=input_shape,
                                                     arr_preprocessing_func=None,
                                                     img_preprocessing_func=None,
                                                     do_shuffling=False,
                                                     do_augmentation=False)

    print('here2')
    model = tf.keras.models.load_model(filepath='models/model1_notshuffled.h5')
    print('here3')
    model.evaluate(x=data_generator_train, verbose=2)

    
    y_predict = model.predict(x=data_generator_test)
    y_predict = tf.argmax(input=y_predict, axis=1)


    # classification report
    target_names = []
    for i in range (1,103):
        target_names.append(f"flower_{i}")


    print(list(test_labels.values()))
    x = list(np.array(y_predict))
    print(x)
    #for some fuking reason it eauther doent work beacous of
    #a. the probelm with batches described earlier
    #b. even if i put porper #of batches, its throws error that it read 103 classes instead of 102
    report = classification_report(y_true=list(test_labels.values()), y_pred=y_predict, target_names=target_names)

    print(report)







