import librosa.display
import matplotlib.pyplot as plt
import numpy
import sounddevice

def melsg(signal, sr, show_sg = False):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr)
    if show_sg:
        plt.figure(figsize=(16,8))
        librosa.display.specshow(
            librosa.power_to_db(mel_spectrogram, ref=numpy.max),
            sr=sr,
            hop_length=512,
            y_axis='hz',
            x_axis='s'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Мелспектрограмма')
        plt.tight_layout()
        plt.show()
    return mel_spectrogram

noise_file = './sound.wav'
signal, sr = librosa.load(noise_file)
tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sr)

c = 0
x = []
for pulse in librosa.beat.plp(y=signal, sr=sr, hop_length=512):
    c += 1
    print(f'{c} : {pulse}')
    x.append(pulse)

mel = melsg(signal, sr, 1)
fig, ax = plt.subplots()
ax.plot(x)
plt.show()


