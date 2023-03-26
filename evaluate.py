import data_gen_new
import utils
import os
import tensorflow as tf
import numpy as np
import pprint
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib

if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    MODELS_PATH = "models"
    METRICS_PATH = "metrics"

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
    #model.evaluate(x=data_generator_test, verbose=2)

    y_predict = model.predict(x=data_generator_test)
    y_predict_argmax = tf.argmax(input=y_predict, axis=1)

    # classification report
    target_names = [f"flower_{i}" for i in range(1, 103)]
    report = classification_report(y_true=list(labels_test.values()),
                                   y_pred=y_predict_argmax,
                                   target_names=target_names,
                                   output_dict=True)
    pprint.pprint(report)

    report_filename = "clfreport_" + os.path.splitext(model_filename)[0] + ".json"
    with open(os.path.join(METRICS_PATH, report_filename), "w") as outfile:
        json.dump(report, outfile, indent=2)

    # confiusion matrix
    cm = confusion_matrix(list(labels_test.values()), y_predict_argmax)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(include_values=False, xticks_rotation='vertical')
    matplotlib.pyplot.show()
