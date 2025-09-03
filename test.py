import pyaudio

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# 打开一个同时具有输入和输出的流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                input_device_index=1,
                frames_per_buffer=CHUNK)
print("* echoing")

# 从麦克风读取数据，并立即写入扬声器
try:
    while True:
        data = stream.read(CHUNK)
        stream.write(data)  # 这会产生一个尖锐的回声
except KeyboardInterrupt:  # 用 Ctrl+C 来停止
    pass

print("* stopped")

stream.stop_stream()
stream.close()
p.terminate()
