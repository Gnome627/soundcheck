import librosa
from glob import glob

from tensorflow import keras
import tensorflow as tf

import numpy as np
import sounddevice
import soundfile
import datetime
import os
import random

# Этот скрипт инициализирует нейронную сеть для 
# проверки ауидопотока в реальном времени 
# на принадлежность к категории "шум" или "пение птиц"

print(sounddevice.query_devices())

default_devices = 1, 3
rate = 44100

learning_epochs = 128

def audio_files_preprocess(audio_files):
    data = []
    shapes = []
    for af in audio_files:
        y, sr = librosa.load(af, sr=rate)
        S = tf.signal.stft(y, frame_length=320, frame_step=32)
        S = tf.abs(S)
        # D = librosa.feature.spectral_contrast(y=y, sr=sr)
        D = tf.expand_dims(S, axis=2)
        shape = D.shape
        data.append(D)
        shapes.append(shape)
    return data, shapes

x = glob('./dataset/*.wav')
for af in x:
    print(f'{af[2:]}')

y = [   
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]

classes = ['bird_singing', 'noise']

zipped = list(zip(x, y))
random.shuffle(zipped)
x, y = zip(*zipped)

x, shapes = audio_files_preprocess(x)
x = np.array(
        x,
        dtype=np.float32
    )
x = x / 255

y = np.array(
        y,
        dtype=np.float32
    )

# for i in range(len(x)):
#     print(f'{len(x[i])} : {shapes[i]}')

print(shapes[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(shapes[0])),
    keras.layers.Dense(96, activation='relu'),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x, y, epochs=learning_epochs)

test_loss, test_acc = model.evaluate(x, y)
print('Test accuracy: ', test_acc)

predictions = model.predict(x)
c = 1
for i in predictions:
    print(f'{c} : predicted = {np.argmax(i)}, real = {int(y[c-1])}')
    c += 1

if not os.path.exists('./records'):
    os.mkdir('records')

def listen(t, default_devices):
    sounddevice.default.device = default_devices
    recording = sounddevice.rec(int(t * rate), samplerate=rate, channels=1)
    sounddevice.wait()
    
    # save file
    file_stamp = str(datetime.datetime.timestamp(datetime.datetime.now())).split('.')[0]
    file_stamp = f'records/record-{file_stamp}.wav'
    soundfile.write((file_stamp), recording, rate)

    return file_stamp

while 1:
    sound_file = listen(t=5, default_devices=default_devices)
    x_test, y_test = audio_files_preprocess([sound_file])
    x_test = np.array(x_test, dtype=np.float32)
    x_test = x_test / 255
    rt_prediction = model.predict(x_test)
    for i in range(len(rt_prediction)):
        print(f'realtime : predicted = {classes[np.argmax(rt_prediction[i])]}')