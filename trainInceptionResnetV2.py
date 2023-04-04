import data_gen_new
import model
import utils
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense

if __name__ == '__main__':
    LABELS_TEST = "data/test_labels.csv"
    LABELS_TRAIN = "data/train_labels.csv"

    IMAGES_TEST_PATH = "data/test"
    IMAGES_TRAIN_PATH = "data/train"

    PLOTS_PATH = "metrics/plots"

    INPUT_SHAPE = (224, 224, 3)

    labels_train = utils.import_labels(LABELS_TRAIN)
    labels_test = utils.import_labels(LABELS_TEST)

    data_generator_train = data_gen_new.DataGenerator(img_dir_path=IMAGES_TRAIN_PATH,
                                                      labels_dict=labels_train,
                                                      batch_size=32,
                                                      target_dim=INPUT_SHAPE[:2],
                                                      arr_preprocessing_func=preprocess_input, #preprocessing for inception_resnet_v2
                                                      img_preprocessing_func=None,
                                                      do_shuffling=False,
                                                      do_augmentation=True)

    data_generator_test = data_gen_new.DataGenerator(img_dir_path=IMAGES_TEST_PATH,
                                                     labels_dict=labels_test,
                                                     batch_size=32,
                                                     target_dim=INPUT_SHAPE[:2],
                                                     arr_preprocessing_func=preprocess_input, #preprocessing for inception_resnet_v2
                                                     img_preprocessing_func=None, 
                                                     do_shuffling=False,
                                                     do_augmentation=False)
    
    base_model = model.makeBaseModelInceptionResNetV2(INPUT_SHAPE)
    
    for layer in base_model.layers:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(102, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    loss_fun = tf.keras.losses.CategoricalCrossentropy()

    #unfreeze last 20 layers
    #restnet
    #50 Epochs
    #Add Dropout Layers 0.5
    #GlobalAveragePooling instead of flatten
    
    model.compile(
    optimizer='adam',
    loss=loss_fun,
    metrics=['accuracy']
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, restore_best_weights=True)

    history = model.fit(data_generator_train, validation_data=data_generator_test, callbacks=[callback], epochs=10)
    model_filename = f"Inception_Resnet_V2_model_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}.h5"
    model.save(f"models/{model_filename}")

    utils.make_plots_from_history(history,PLOTS_PATH, model_filename)

    
