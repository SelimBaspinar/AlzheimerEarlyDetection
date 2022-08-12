#SELİM BAŞPINAR - 44128127874
#Mehmet Ali BALİ - 63925397150
#Mustafa BÖLÜK - 48301492506

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
import tensorflow as tf
import os

W = 128
H = 128
classNames = {
    'MildDemented': 0,
    'ModerateDemented': 1,
    'NonDemented': 2,
    'VeryMildDemented': 3
}
numberClasses = len(classNames)

def getImages(dir_name='./AlzheimerDataSet',classNames=classNames):
    """read images / labels from directory"""

    Images = []
    Classes = []

    for j in ['/train', '/test']:
        for label_name in os.listdir(dir_name + str(j)):
            cls = classNames[label_name]

            for img_name in os.listdir('/'.join([dir_name + str(j), label_name])):
                img = load_img(
                    '/'.join([dir_name + str(j), label_name, img_name]), target_size=(W, H))
                img = img_to_array(img)

                Images.append(img)
                Classes.append(cls)

    Images = np.array(Images, dtype=np.float32)
    Classes = np.array(Classes, dtype=np.float32)
    Images, Classes = shuffle(Images, Classes, random_state=0)

    return Images, Classes
# get images / labels


Images, Classes = getImages()

# split train / test
indicesTrain, indicesTest = train_test_split(
    list(range(Images.shape[0])), train_size=0.8, test_size=0.2, shuffle=True)

x_train = Images[indicesTrain]
y_train = Classes[indicesTrain]
x_test = Images[indicesTest]
y_test = Classes[indicesTest]

y_train = keras.utils.np_utils.to_categorical(y_train, numberClasses)
y_test = keras.utils.np_utils.to_categorical(y_test, numberClasses)

datagenTrain = ImageDataGenerator(
    # image preprocessing function
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    rotation_range=30,                       # randomly rotate images in the range
    zoom_range=0.1,                          # Randomly zoom image
    width_shift_range=0.1,                   # randomly shift images horizontally
    height_shift_range=0.1,                  # randomly shift images vertically
    horizontal_flip=True,                    # randomly flip images horizontally
    vertical_flip=False,                     # randomly flip images vertically
    validation_split=0.2
)
datagenTest = ImageDataGenerator(

    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
)


IMG_SHAPE = (W, H, 3)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(4, activation='softmax')
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    keras.layers.BatchNormalization(),
    keras.layers.Dense(2048, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    prediction_layer
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpoint = keras.callbacks.ModelCheckpoint("mri_resnet50.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto')

early = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit(datagenTrain.flow(x_train, y_train, batch_size=32, subset='training'), epochs=50,
                    validation_data=datagenTrain.flow(
                        x_train, y_train, batch_size=8, subset='validation'),
                    callbacks=[checkpoint, early])

model.load_weights("mri_resnet50.h5")
test_loss, test_acc = model.evaluate(
    datagenTest.flow(x_test, y_test, batch_size=32), verbose=2)
print(test_acc)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
model.save("resnet50.h5")
predictions = model.predict(datagenTest.flow(x_test, y_test, batch_size=32))
np.array(predictions)
CATEGORIES = ['MildDemented', 'ModerateDemented',
              'NonDemented', 'VeryMildDemented']
rows = 2
columns = 5
fig, axs = plt.subplots(rows, columns, figsize=(25, 10))
axs = axs.flatten()
i = 10
for a in axs:
    Image = Images[i]
    pred_label = predictions.argmax(axis=1)[i]
    actual_label = y_test.argmax(axis=1)[i]
    pred_label = CATEGORIES[pred_label]
    actual_label = CATEGORIES[actual_label]
    label = 'pred: ' + pred_label + ' ' + 'real: ' + actual_label
    a.imshow(np.uint8(Image))
    a.set_title(label)
    i = i + 1

plt.show()