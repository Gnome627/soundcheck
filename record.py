import sounddevice
import flet

# sounddevice.default.device = 1, 3

### function listening to the sound and return it as float value
# setting the rate
rate = 44100
print(sounddevice.query_devices())

def main(page: flet.Page):
    page.window_width=256
    page.window_height=256
    page.window_resizable=0

    code = 0.0

    indicator = flet.Icon(
        name=flet.icons.CIRCLE,
        size=64
    )
    
    def indicator_repaint(code_value):
        # if code_value >= 3.0:
        #     indicator.color = flet.colors.RED
        # elif code_value >= 2.0:
        #     indicator.color = flet.colors.ORANGE
        # elif code_value >= 1.0:
        #     indicator.color = flet.colors.YELLOW
        # else:
        #     indicator.color = flet.colors.GREEN
        if code_value > 3: code_value = 3.0
        new_color = (
            f'#{str(hex(int(code_value * 85))[2:]).zfill(2)}' +
            f'{str(hex(255 - int(code_value * 85))[2:]).zfill(2)}00'
        )
        print(new_color)
        indicator.color=new_color

    page.add(
        flet.Row(
            [
                flet.Container(
                    content=indicator,
                    padding=10
                )
            ]
        )
    )

    def listen(sec, default):
        sounddevice.default.device = default
        recording = sounddevice.rec(int(sec * rate), samplerate=rate, channels=1)
        sounddevice.wait()
        y = 0.0
        for x in recording:
            if round(x[0], 4) > y: y = x[0]
        return round(y * 10, 3)

    while 1:
        code = listen(0.1, (1, 3))
        print(code)
        indicator_repaint(code)
        page.update()

flet.app(target=main)


