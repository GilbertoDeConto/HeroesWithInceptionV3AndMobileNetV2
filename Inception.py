import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from pathlib import Path
import os.path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.inception_v3 import InceptionV3

image_dir = Path('data/')

dirNames = os.listdir(image_dir)

cat = []
jpgnames = []
for filename in dirNames:
    jpgname = os.listdir('data/' + filename)
    for jpg in jpgname:
        jpgnames.append('data\\' + filename + '\\' + jpg)
        cat.append(filename)
    
    
df_train = pd.DataFrame({
    'file': jpgnames,
    'cat': cat
})

train, test = train_test_split(df_train, train_size=0.7, shuffle=True, random_state=1)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
    validation_split=0.3
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train,
    x_col='file',
    y_col='cat',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    #seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train,
    x_col='file',
    y_col='cat',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    #seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test,
    x_col='file',
    y_col='cat',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

model = InceptionV3(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

model.trainable = False 

inputs = model.input
x = Dense(128, activation='sigmoid')(model.output)

x = Dense(128, activation='sigmoid')(x)
outputs = Dense(8, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=25,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

res = model.evaluate(test_images, verbose=0)
predictions = np.argmax(model.predict(test_images), axis=1)

class_names = list(test_images.class_indices.keys())


var = {0: 'black widow', 1: 'captain america', 2: 'doctor strange', 3: 'hulk', 4: 'ironman', 5: 'loki', 6: 'spider-man', 7: 'thanos'}
img, labels = test_images[0]
img = img[11:19]
img = (img + 1) / 2
plt.figure(figsize=(20, 15))
row = np.math.ceil(len(img) / 4)
for i, image in enumerate(img):
    plt.subplot(row, 4, (i+1))
    plt.title(var[labels[i+11].argmax()], fontsize=30)
    plt.text(0, 256, var[predictions[i+11]], color='red', fontsize=30)
    plt.axis('off')
    plt.imshow(image)
    
    
    
cm = confusion_matrix(test_images.labels, predictions, labels=np.arange(8))
clr = classification_report(test_images.labels, predictions, labels=np.arange(8), target_names=class_names)

print("Test Accuracy: {:.2f}%".format(res[1] * 100))


plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=np.arange(8) + 0.5, labels=class_names, rotation=90)
plt.yticks(ticks=np.arange(8) + 0.5, labels=class_names, rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Resultados Obtidos:\n----------------------\n", clr)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.legend()

