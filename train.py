import data_gen_new
import model


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
    model = model.makeModel2(INPUT_SHAPE)

    model.fit(data_generator_train, validation_data=data_generator_test, epochs=1)
    model.save(f"models/model_name_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}.h5")
