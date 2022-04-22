import matplotlib.pyplot as plt
import sklearn
from keras import backend as K
from keras.callbacks import History
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

# Dimension of images
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

img_height, img_width = 150, 150

train_data_dir = 'rice_leaf_diseases'
validation_data_dir = 'rice_leaf_diseases'
nb_train_samples = 111
nb_validation_samples = 100
epochs = 30
batch_size = 25


if K.image_data_format == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale = 1. /255,
    shear_range = 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)

train_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size= (img_width, img_height),
    batch_size=batch_size,
    classes=['Bacterial leaf blight','Brown Spot','Leaf Smut'],
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size= (img_width, img_height),
    batch_size=batch_size,
    classes=['Bacterial leaf blight','Brown Spot','Leaf Smut'],
    class_mode='categorical')

#########################################################################################
# Model Creation

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss= 'categorical_crossentropy',
              optimizer='rmsprop',
              metrics = ['accuracy'])

#This is the augmentation configuration we will use for training
history = History()
model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,callbacks=[history],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size
)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()



trash = 'trail'
(X_train, y_train), (X_test, y_test) = trash.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("rice.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("rice.h5")
