import data_gen_new
import model
import tensorflow as tf

if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    INPUT_SHAPE = (224, 224, 3)

    data_generator_train = data_gen_new.DataGenerator(img_dir_path=IMAGES_TRAIN_PATH,
                                                      labels_file_path=LABELS_TRAIN,
                                                      batch_size=25,
                                                      target_dim=INPUT_SHAPE[:2],
                                                      arr_preprocessing_func=None,
                                                      img_preprocessing_func=None,
                                                      do_shuffling=False,
                                                      do_augmentation=False)

    data_generator_test = data_gen_new.DataGenerator(img_dir_path=IMAGES_TEST_PATH,
                                                     labels_file_path=LABELS_TEST,
                                                     batch_size=25,
                                                     target_dim=INPUT_SHAPE[:2],
                                                     arr_preprocessing_func=None,
                                                     img_preprocessing_func=None,
                                                     do_shuffling=False,
                                                     do_augmentation=False)

    model = tf.keras.models.load_model(filepath='models/model2.hp')
    model.evaluate(x=data_generator_train, verbose=2)
