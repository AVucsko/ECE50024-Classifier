import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import pathlib
import csv

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## Check for gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Load data and create datasets
batch_size = 5
img_height = 180
img_width = 180

data_dir = "separated_faces"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size)
    
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset="validation",
    seed=123,
    image_size=(img_height,img_width),
    batch_size=batch_size)
    
class_names = train_ds.class_names
print(class_names)


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
    
## Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

## Standardize the Data
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]


## Create a basic model
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


## Train the Model
epochs=7
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



## Visualize Training Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
model.save_weights('tflow_models/test1.ckpt.weights.h5')
print('Saved!')


## Run the model on new data
new_im_path = 'test_faces/'
#outputs = np.zeros((4977,2))
with open('answers2.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id','Category'])
    for idx in range(0,4977):
        print(idx)
        img = tf.keras.utils.load_img(new_im_path + str(idx) + '.jpg', target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array,0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        #outputs[idx] = [idx, class_names[np.argmax(score)]]
        writer.writerow([idx, class_names[np.argmax(score)]])

#print(outputs)
