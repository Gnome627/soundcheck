import sounddevice
import flet
from flet.matplotlib_chart import MatplotlibChart
import datetime
import soundfile
import os
import numpy
import matplotlib.pyplot as plot
import librosa

# print(sounddevice.query_devices()) # sounddevice check
# basic parameters
rate = 44100
code = 0.0
file_length = 0
# creating folder for records
if not os.path.exists('./records'):
    os.mkdir('records')

### function listening to the sound
def listen(sec, default):
    sounddevice.default.device = default
    recording = sounddevice.rec(int(sec * rate), samplerate=rate, channels=1)
    sounddevice.wait()
    
    # save file
    file_stamp = str(datetime.datetime.timestamp(datetime.datetime.now())).split('.')[0]
    file_stamp = f'records/record-{file_stamp}.wav'
    soundfile.write((file_stamp), recording, rate)

    y = 0.0
    for x in recording:
        if round(x[0], 4) > y: y = x[0]
    return round(y * 10, 3), file_stamp

### spectrogram function
def create_mel_sg(sound):
    signal, sr = librosa.load(sound)
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128,
                                                     fmax=20000)
    return mel_spectrogram

### flet webpage main function
def main(page: flet.Page):

    # window settings
    page.window_width=760
    page.window_height=640
    page.window_resizable=0
    page.window_center=True
    page.window_maximizable=False
    page.theme=flet.Theme(color_scheme_seed=flet.colors.BLUE)
    page.theme_mode='LIGHT'

    # sensivity slider
    slider = flet.Slider(
        min=0.0,
        max=50.0,
        divisions=5,
        label='{value}',
        value=10.0
    )

    # colorable indicator
    indicator = flet.Icon(
        name=flet.icons.VOLUME_OFF,
        color=flet.colors.BLACK45,
        size=64
    )

    def indicator_repaint(code_value):
        channel = code_value * (slider.value + 10)
        print(channel)
        if channel > 255: channel = 255
        new_color = (
            f'#{str(hex(int(channel))[2:]).zfill(2)}' +
            f'00{str(hex(255 - int(channel))[2:]).zfill(2)}'
        )
        # print(new_color)
        indicator.color=new_color
        if channel > 100:
            indicator.name=flet.icons.VOLUME_UP
        elif channel < 75:
            indicator.name=flet.icons.VOLUME_MUTE
        else:
            indicator.name=flet.icons.VOLUME_DOWN

        page.update()

    # container for matplotlib spectrogram
    plot_container = flet.Container(
        width=720,
        height=480,
        padding=0,
        border_radius=10,
        content=flet.ProgressBar(color=flet.colors.PRIMARY_CONTAINER)
    )

    def draw_mel_sg_on_container(mel_sg):
        fig, ax = plot.subplots()
        sg_db = librosa.power_to_db(mel_sg, ref=numpy.max)
        img = librosa.display.specshow(
            sg_db,
            sr=rate,
            fmax=20000,
        )
        fig.colorbar(
            img,
            format='%+2.0f dB'
        )
        plot_container.content = MatplotlibChart(fig, expand=True)

        page.update()
        plot.close()

    # adding controls to page
    page.add(
        flet.Row(
            [
                flet.Container(
                    content=indicator,
                    border_radius=10,
                    padding=10
                ),
                flet.Column(
                    [
                        flet.Text('Усиление'),
                        slider
                    ]
                )
            ]
        ),
        flet.Row(
            [
                plot_container
            ]
        )
    )

    page.update()

    ### mainloop
    while 1:
        code, sound_file = listen(sec=5, default=(1, 3))
        sg = create_mel_sg(sound_file)
        draw_mel_sg_on_container(sg)
        indicator_repaint(code)

# start service
flet.app(target=main)


