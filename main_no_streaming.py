import pyaudio
import webrtcvad
import multiprocessing as mp
from queue import Empty
import time
import numpy as np
import keyboard  # 需要安装: pip install keyboard
import threading

import funASR
import funASR_no_streaming

# ==================== 配置参数 ====================
# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz对于语音识别足够
CHUNK = int(RATE * 0.01)  # 小块数据降低延迟
VAD_AGGRESSIVENESS = 3  # VAD激进程度 (0-3)

# 设备选择 (如果自动选择不正确，可以手动指定)
INPUT_DEVICE_INDEX = None  # 设为None自动选择，或指定具体索引
# 人物位置 枚举：left right 默认人物在左边
LOCATION = 'left'
# 语音指令映射
COMMAND_MAPPING = {
    "转向": ["@"],
    "波": ["s", ("s", "d"), "d", "o"],
    "升": ["d", ("s", "d"), "s", ("s", "d"), "d", "i"],
    # "波": ["i"],
    # "升": [("d", "i")],
    # "多达": [("a", "i")],
    "迸发": ["y"],
    "下中脚": [("s", "k")],
    "下中脚取消": [("s", "k"), "", "", "h"],
    "下中拳": [("s", "i")],
    "取消": ["h"],
    "重拳": ["o"],
    "中拳": ["i"],
    "中拳重拳": ["i", "", "", "", "", "o"],
    "tc": ["i", "", "", "", "", "o"],
    "跑旋风腿": [("j", "k"), 's', ("s", "a"), ('a', 'j'), 'j'],
    "TC跑旋风腿": ["i", "#14", "o", "#16", ("j", "k"), "#13", 's', ("s", "a"), 'a', 'j'],
    "跳重脚": ['w', "#29", 'l'],
    "绿冲": ['h', ("h", "d"), "h", ("h", "d")],
    "跑升龙": [("j", "k"), 'd', ("d", "s"), ('d', 'i'), 'i'],
    "一个z": ['s', ("s", "d"), 'd', 's', ("s", "d"), 'd', 'i'],
    "多达": ['d', ("d", "s"), 's', ("s", "d"), 'd', 'l'],
    "投他": [("j", "u")],
    "咿呀": ['s', ("s", "d"), 'd', 'k', '#5', 'j'],
    "下中脚复合": ['$下中脚取消', "#15", '$下中拳', "#45", "$TC跑旋风腿"],

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
        self.audio_buffer = b''
        self.silence_frames = 0
        self.MAX_SILENCE_FRAMES = 0

    def callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数，处理音频数据"""
        # 在音频采集层进行VAD检测
        if self.vad.is_speech(in_data, RATE):
            self.audio_buffer += in_data
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            # 即使是静音，我们也重置语音帧计数器，以便下次说话时重新开始计数

        # 如果连续静音时间够长，认为一句话结束
        if self.silence_frames > self.MAX_SILENCE_FRAMES and len(self.audio_buffer) > 0:
            # 将有效的音频数据放入队列
            self.audio_queue.put(self.audio_buffer)
            # 清空缓冲
            self.audio_buffer = b''
            self.silence_frames = 0

        return (None, pyaudio.paContinue)


def find_low_latency_device(p):
    """尝试找到低延迟的音频输入设备"""
    default_device = p.get_default_input_device_info()
    return default_device["index"]


# ==================== 语音识别进程 ====================
def speech_recognition_process(audio_queue, command_queue, stop_event):
    print("语音识别进程启动")
    model = funASR_no_streaming.FunASR(keywords=COMMAND_MAPPING.keys())

    while not stop_event.is_set():
        try:
            # 直接从队列获取有效音频数据
            audio_buffer = audio_queue.get(timeout=0.1)

            print('-' * 10)
            # 计算传入语音的时间长度
            duration = len(audio_buffer) / RATE / 2
            print(f"录音时长: {duration:.2f}秒")
            # 记录识别开始时间
            start_time = time.time()
            result = model.generate(audio_buffer)
            # 计算识别用时
            recognition_time = time.time() - start_time
            print("识别结果:", result)
            print(f"识别用时: {recognition_time:.4f}秒")
            map_to_execution(result, command_queue)

        except Empty:
            continue


def map_to_execution(recognition_result, command_queue):
    """
    映射识别结果到执行函数
    recognition_result:[{'key': 'rand_key_NO6n9JEC3HqdZ', 'text': 'detected 发波 0.21558110781628012'}]
    """
    if not recognition_result:
        return
    result_ = recognition_result[0]
    key = result_['text'].replace(" ", "")
    if key in COMMAND_MAPPING.keys():
        command_queue.put(key)
    # for key in COMMAND_MAPPING.keys():
    #     if key in result_['text']:
    #         # 塞入操作队列
    #         command_queue.put(key)


# ==================== 指令执行线程 ====================
def command_execution_thread(command_queue, stop_event):
    """指令执行线程，将语音指令转换为游戏操作"""
    print("指令执行线程启动")

    while not stop_event.is_set():
        try:
            command = command_queue.get(timeout=0.1)
            print(f"执行指令: {command}")
            # 记录执行开始时间
            start_time = time.time()
            execute_command(command)
            # 计算执行用时
            execution_time = time.time() - start_time
            print(f"执行用时: {execution_time:.4f}秒")
        except Empty:
            continue
        except Exception as e:
            print(f"指令执行错误: {e}")

    print("指令执行线程结束")


def execute_command(command):
    """
    :param command: 指令名称
    :return:
    """
    if command not in COMMAND_MAPPING:
        return
    keys = COMMAND_MAPPING[command]
    for key in keys:
        if isinstance(key, tuple):
            # 处理同时按键的情况
            for k in key:
                press(k)

            # 添加适当的延迟
            time.sleep(0.02)

            # 释放按键
            for k in key:
                release(k)
        elif key:
            if key.__contains__("#"):
                # 处理放帧
                num = int(key.split("#")[1])
                time.sleep(1.0 / 60 * num)
            elif key.__contains__("$"):
                # 处理嵌套指令
                cmd = key.split("$")[1]
                execute_command(cmd)
            elif key == "@":
                LOCATION = 'right'
            else:
                # 处理单个按键的情况
                press(key)
                time.sleep(0.02)
                release(key)
        else:
            time.sleep(0.02)


def press(key):
    keyboard.press(key_map_by_location(key))


def release(key):
    keyboard.release(key_map_by_location(key))


def key_map_by_location(key):
    if LOCATION == 'right':
        # ad互换
        if key == 'd':
            key = 'a'
        elif key == 'a':
            key = 'd'
    return key


# ==================== 主程序 ====================
def main():
    print("启动语音控制街霸系统...")

    # 创建进程间通信队列l
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
