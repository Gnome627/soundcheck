import librosa
from glob import glob

from tensorflow import keras
import tensorflow as tf

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import utils

import pandas as pd
import numpy as np
import sounddevice
import soundfile
import datetime
import os

def audio_files_to_librosa(audio_files):
    spectrograms = []
    shapes = []
    for af in audio_files:
        y, sr = librosa.load(af)
        D = librosa.stft(y)
        shape = D.shape
        spectrograms.append(D)
        shapes.append(shape)
    return spectrograms, shapes

x, shapes = audio_files_to_librosa(glob('./dataset/*.wav'))
x = np.array(x, dtype=np.float32)
x = x / 255
y = np.array(
       [0, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0,],
       dtype=np.float32
    )
classes = ['ok', 'breaking']

# for i in range(len(x)):
#     print(f'{len(x[i])} : {shapes[i]}')

model = keras.Sequential([
    keras.layers.Flatten(input_shape=shapes[0]),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x, y, epochs=1024)

test_loss, test_acc = model.evaluate(x, y)
print('Test accuracy: ', test_acc)

predictions = model.predict(x)
c = 1
for i in predictions:
    print(f'{c} : predicted={np.argmax(i)}, real={y[c-1]}')
    c += 1

rate = 44100
code = 0.0
file_length = 0

if not os.path.exists('./records'):
    os.mkdir('records')

def listen(sec, default):
    sounddevice.default.device = default
    recording = sounddevice.rec(int(sec * rate), samplerate=rate, channels=1)
    sounddevice.wait()
    
    # save file
    file_stamp = str(datetime.datetime.timestamp(datetime.datetime.now())).split('.')[0]
    file_stamp = f'records/record-{file_stamp}.wav'
    soundfile.write((file_stamp), recording, rate)

    return file_stamp

while 1:
    sound_file = listen(sec=5, default=(1, 3))
    x_test, y_test = audio_files_to_librosa([sound_file])
    x_test = np.array(x_test)
    x_test = x_test / 255
    rt_prediction = model.predict(x_test)
    for i in range(len(rt_prediction)):
        print(f'realtime : predicted={classes[np.argmax(rt_prediction[i])]}')