import data_gen_new
import model
import utils
import tensorflow as tf

if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    INPUT_SHAPE = (224, 224, 3)

    labels_train = utils.import_labels(LABELS_TRAIN)
    labels_test = utils.import_labels(LABELS_TEST)

    data_generator_train = data_gen_new.DataGenerator(img_dir_path=IMAGES_TRAIN_PATH,
                                                      labels_dict=labels_train,
                                                      batch_size=10,
                                                      target_dim=INPUT_SHAPE[:2],
                                                      arr_preprocessing_func=None,
                                                      img_preprocessing_func=None,
                                                      do_shuffling=False,
                                                      do_augmentation=False)

    data_generator_test = data_gen_new.DataGenerator(img_dir_path=IMAGES_TEST_PATH,
                                                     labels_dict=labels_test,
                                                     batch_size=10,
                                                     target_dim=INPUT_SHAPE[:2],
                                                     arr_preprocessing_func=None,
                                                     img_preprocessing_func=None,
                                                     do_shuffling=False,
                                                     do_augmentation=False)
    model = model.makeModel(INPUT_SHAPE)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, restore_best_weights=True)

    history = model.fit(data_generator_train, validation_data=data_generator_test, epochs=5)
    model_filename = f"testing_model_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}.h5"
    model.save(f"models/{model_filename}")

    utils.make_plots_from_history(history, model_filename)

    
