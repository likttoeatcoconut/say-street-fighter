import pyaudio
import webrtcvad
import multiprocessing as mp
from queue import Empty
import time
import numpy as np
import keyboard  # 需要安装: pip install keyboard
import threading

import funASR

# ==================== 配置参数 ====================
# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz对于语音识别足够
CHUNK = int(RATE * 0.03)  # 小块数据降低延迟
VAD_AGGRESSIVENESS = 3  # VAD激进程度 (0-3)

# 设备选择 (如果自动选择不正确，可以手动指定)
INPUT_DEVICE_INDEX = None  # 设为None自动选择，或指定具体索引

# 语音指令映射
COMMAND_MAPPING = {
    "hadouken": ["down", "down", "right", "p"],
    "shoryuken": ["right", "down", "right", "p"],
    "sonicboom": ["back", "back", "p"],
    # 添加更多指令...
}

# ==================== 音频采集进程 ====================
def audio_capture_process(audio_queue, stop_event):
    """音频采集进程，负责低延迟录音和VAD检测"""
    print("音频采集进程启动")

    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 如果没有指定设备，尝试查找低延迟设备
    if INPUT_DEVICE_INDEX is None:
        input_device_index = find_low_latency_device(p)
    else:
        input_device_index = INPUT_DEVICE_INDEX

    print(f"使用音频设备: {input_device_index}")

    # 初始化VAD
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # 创建回调函数类实例
    callback_handler = AudioCallbackHandler(vad, audio_queue)

    # 打开音频流 - 使用正确的回调方式
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=False,
        frames_per_buffer=CHUNK,
        stream_callback=callback_handler.callback  # 直接传递回调方法
    )

    stream.start_stream()

    # 等待停止事件
    while not stop_event.is_set():
        time.sleep(0.1)

    # 清理资源
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("音频采集进程结束")


class AudioCallbackHandler:
    """处理音频回调的类"""
    def __init__(self, vad, audio_queue):
        self.vad = vad
        self.audio_queue = audio_queue

    def callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数，处理音频数据"""
        # 使用VAD检测是否有语音
        is_speech = False
        try:
            is_speech = self.vad.is_speech(in_data, RATE)
        except:
            # 有时VAD会因数据长度问题抛出异常，忽略这些帧

            pass

        # 只有当检测到语音时才将数据放入队列
        if is_speech:
            self.audio_queue.put(in_data)
        # self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)


def find_low_latency_device(p):
    """尝试找到低延迟的音频输入设备"""
    default_device = p.get_default_input_device_info()
    return default_device["index"]


# ==================== 语音识别进程 ====================
def speech_recognition_process(audio_queue, command_queue, stop_event):
    """语音识别进程，处理音频并识别指令"""
    print("语音识别进程启动")

    # 初始化语音识别模型
    # 这里应该替换为实际的本地轻量级模型
    # 例如: model = load_your_model()
    model = funASR.FunASR()

    audio_buffer = b''
    silence_frames = 0
    MAX_SILENCE_FRAMES = 5  # 持续10帧静音则认为语音结束

    while not stop_event.is_set():
        try:
            # 非阻塞获取音频数据
            data = audio_queue.get(timeout=0.1)
            audio_buffer += data
            silence_frames = 0  # 重置静音计数

            # 当积累足够长的音频后进行处理
            if len(audio_buffer) > RATE * 1:  # 1秒音频
                # 这里应该调用实际的语音识别模型
                command = model.generate(audio_buffer)

                # 模拟识别过程 - 实际应替换为模型推理
                # command = simulate_speech_recognition(audio_buffer)
                print(f"识别结果: {command}")

                # if command in COMMAND_MAPPING:
                #     print(f"识别到指令: {command}")
                #     command_queue.put(command)
                if "发波" in command or "升龙" in command:
                    # 清空缓冲区
                    audio_buffer = b''
                else:
                    # 清空一半
                    half = (len(audio_buffer) // 4) * 2
                    audio_buffer = audio_buffer[half:]
        except Empty:
            # 队列为空时，增加静音计数
            silence_frames += 1
            if silence_frames > MAX_SILENCE_FRAMES and audio_buffer:
                # 长时间静音，处理缓冲区中剩余的音频
                if len(audio_buffer) > CHUNK * 3:  # 至少有3帧数据
                    command = model.generate(audio_buffer)
                    # command = simulate_speech_recognition(audio_buffer)
                    print(f"识别结果: {command}")

                    # if command in COMMAND_MAPPING:
                    #     print(f"识别到指令: {command}")
                    #     command_queue.put(command)

                audio_buffer = b''
                silence_frames = 0
        except Exception as e:
            print(f"语音识别错误: {e}")

    print("语音识别进程结束")


def simulate_speech_recognition(audio_data):
    """模拟语音识别函数 - 实际应替换为真正的模型推理"""
    # 这里只是模拟，实际应用中应该:
    # 1. 提取音频特征 (MFCC等)
    # 2. 使用训练好的模型进行推理
    # 3. 返回识别结果



    # 简单模拟：随机返回一个指令或None
    import random
    if random.random() > 0.9:  # 10%的概率返回指令
        return random.choice(list(COMMAND_MAPPING.keys()))
    return None


# ==================== 指令执行线程 ====================
def command_execution_thread(command_queue, stop_event):
    """指令执行线程，将语音指令转换为游戏操作"""
    print("指令执行线程启动")

    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
            if command in COMMAND_MAPPING:
                print(f"执行指令: {command}")
                execute_command(COMMAND_MAPPING[command])
        except Empty:
            continue
        except Exception as e:
            print(f"指令执行错误: {e}")

    print("指令执行线程结束")


def execute_command(key_sequence):
    """执行按键序列"""
    for key in key_sequence:
        keyboard.press(key)
        time.sleep(0.05)  # 短暂延迟模拟人手操作
        keyboard.release(key)
        time.sleep(0.05)


# ==================== 主程序 ====================
def main():
    print("启动语音控制街霸系统...")

    # 创建进程间通信队列
    audio_queue = mp.Queue(maxsize=100)  # 音频数据队列
    command_queue = mp.Queue(maxsize=10)  # 指令队列

    # 创建事件用于停止进程
    stop_event = mp.Event()

    try:
        # 创建并启动音频采集进程
        audio_process = mp.Process(
            target=audio_capture_process,
            args=(audio_queue, stop_event)
        )
        audio_process.start()

        # 创建并启动语音识别进程
        recognition_process = mp.Process(
            target=speech_recognition_process,
            args=(audio_queue, command_queue, stop_event)
        )
        recognition_process.start()

        # 在主线程中启动指令执行
        command_thread = threading.Thread(
            target=command_execution_thread,
            args=(command_queue, stop_event)
        )
        command_thread.start()

        # 等待用户中断
        print("系统运行中... 按Ctrl+C停止")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("正在停止系统...")
        stop_event.set()

        # 等待进程结束
        audio_process.join(timeout=2)
        recognition_process.join(timeout=2)
        command_thread.join(timeout=2)

        # 强制终止如果还在运行
        if audio_process.is_alive():
            audio_process.terminate()
        if recognition_process.is_alive():
            recognition_process.terminate()

        print("系统已停止")


if __name__ == "__main__":
    main()