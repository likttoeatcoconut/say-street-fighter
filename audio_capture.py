import pyaudio
import webrtcvad
import time
from multiprocessing import Queue, Event
from config import AUDIO_CONFIG


class AudioCapture:
    def __init__(self, audio_queue: Queue, stop_event: Event):
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.pyaudio = None
        self.vad = None
        self.stream = None
        self.callback_handler = None

    def find_low_latency_device(self):
        """尝试找到低延迟的音频输入设备"""
        default_device = self.pyaudio.get_default_input_device_info()
        return default_device["index"]

    def start(self):
        """启动音频采集"""
        print("音频采集进程启动")
        self.vad = webrtcvad.Vad(AUDIO_CONFIG['VAD_AGGRESSIVENESS'])
        self.pyaudio = pyaudio.PyAudio()
        # 设置输入设备
        input_device_index = (AUDIO_CONFIG['INPUT_DEVICE_INDEX'] if AUDIO_CONFIG['INPUT_DEVICE_INDEX']
                              else self.find_low_latency_device())
        print(f"使用音频设备: {input_device_index}")

        # 创建回调处理器
        self.callback_handler = AudioCallbackHandler(self.vad, self.audio_queue)

        # 打开音频流
        self.stream = self.pyaudio.open(
            format=AUDIO_CONFIG['FORMAT'],
            channels=AUDIO_CONFIG['CHANNELS'],
            rate=AUDIO_CONFIG['RATE'],
            input=True,
            output=False,
            frames_per_buffer=AUDIO_CONFIG['CHUNK'],
            stream_callback=self.callback_handler.callback
        )

        self.stream.start_stream()

        # 等待停止事件
        while not self.stop_event.is_set():
            time.sleep(0.1)

        self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()
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
        if self.vad.is_speech(in_data, AUDIO_CONFIG['RATE']):
            self.audio_buffer += in_data
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        # 如果连续静音时间够长，认为一句话结束
        if self.silence_frames > self.MAX_SILENCE_FRAMES and len(self.audio_buffer) > 0:
            self.audio_queue.put(self.audio_buffer)
            self.audio_buffer = b''
            self.silence_frames = 0

        return (None, pyaudio.paContinue)
