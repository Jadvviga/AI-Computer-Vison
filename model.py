import tensorflow as tf

def makeModel(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10, 10), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))

    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10, 10), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(102, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model

def makeModel2(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(102, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'], optimizer="adam")

    return model

def makeBaseModelVGG16(input_shape):
    #simple VGG16 stump as described in https://keras.io/api/applications/vgg/
    model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=input_shape
    )


    return model

def makeBaseModelVGG19(input_shape):
    #simple VGG16 stump as described in https://keras.io/api/applications/vgg/
    model = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    input_shape=input_shape #input needs to be preprocessed
    )

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model

def makeModelInceptionResNetV2(input_shape):
    #simple VGG16 stump as described in https://keras.io/api/applications/inceptionresnetv2/
    model = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape, #input needs to be preprocessed
    pooling="avg",
    classes=102,
    classifier_activation="softmax",
    )

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model

def makeModelDenseNet121(input_shape):
    #simple VGG16 stump as described in https://keras.io/api/applications/densenet/
    model = tf.keras.applications.DenseNet121(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=input_shape, #input needs to be preprocessed
    pooling="avg",
    classes=102,
    classifier_activation="softmax",
    )

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    pass