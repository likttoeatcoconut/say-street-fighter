import pyaudio
import time
import threading

# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512

# 全局变量
audio_data_buffer = []
buffer_lock = threading.Lock()
data_available = threading.Event()

def audio_callback(in_data, frame_count, time_info, status):
    """
    PyAudio回调函数，处理音频输入和输出
    参数:
        in_data: 输入的音频数据
        frame_count: 帧数量
        time_info: 时间信息
        status: 状态标志
    返回:
        (out_data, pyaudio.paContinue) 元组
    """
    print("123")
    global audio_data_buffer

    # 将输入的音频数据保存到缓冲区
    with buffer_lock:
        audio_data_buffer.append(in_data)
        # 限制缓冲区大小，避免占用过多内存
        if len(audio_data_buffer) > 100:
            audio_data_buffer.pop(0)

    # 设置数据可用事件
    data_available.set()

    # 返回输入的数据作为输出（实现回声效果）
    return (in_data, pyaudio.paContinue)

def main():
    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流，使用回调方式
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback
    )

    # 开始音频流
    stream.start_stream()
    print("开始音频回环，按 Ctrl+C 停止...")

    try:
        # 保持程序运行
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("正在停止...")

    # 停止并关闭流
    stream.stop_stream()
    stream.close()

    # 终止PyAudio
    p.terminate()
    print("程序已停止")

if __name__ == "__main__":
    main()
