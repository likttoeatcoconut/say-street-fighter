import pyaudio
import wave
import time

def record_audio(duration=1, sample_rate=16000, channels=1, chunk=1024, output_file="recorded_audio.wav"):
    """
    录制指定时长的音频并保存为WAV文件

    参数:
        duration: 录制时长（秒）
        sample_rate: 采样率（Hz）
        channels: 声道数
        chunk: 每次读取的帧数
        output_file: 输出文件名
    """
    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"开始录制 {duration} 秒音频...")

    frames = []

    # 计算需要读取的块数
    total_chunks = int(sample_rate / chunk * duration)

    # 录制音频
    for i in range(total_chunks):
        data = stream.read(chunk)
        frames.append(data)

        # 显示进度
        progress = (i + 1) / total_chunks * 100
        print(f"\r录制进度: {progress:.1f}%", end="")

    print("\n录制完成!")

    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 保存为WAV文件
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"音频已保存到: {output_file}")

if __name__ == "__main__":
    # 录制1秒音频，保存为recorded_audio.wav
    record_audio(duration=2, output_file="sl.wav")