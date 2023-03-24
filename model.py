import tensorflow as tf



def makeModel(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10, 10), activation='relu'))
    #model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))

    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10, 10), activation='relu'))
    #model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3)))

    # convert to a fully connected network
    model.add(tf.keras.layers.Flatten())

    # Add a classifier part

    model.add(tf.keras.layers.Dense(150, activation='relu'))
    model.add(tf.keras.layers.Dense(150, activation='relu'))

    #the number of output neurons must be equal to the number of classes
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
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.add(tf.keras.layers.Dense(102, activation='softmax'))

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss_fun, metrics=['accuracy'], optimizer="adam")

    return model


if __name__ == '__main__':
    pass