import time
from multiprocessing import Queue, Event
from queue import Empty

import funASR_no_streaming
from config import COMMAND_MAPPING

class SpeechRecognition:
    def __init__(self, audio_queue: Queue, command_queue: Queue, stop_event: Event):
        self.audio_queue = audio_queue
        self.command_queue = command_queue
        self.stop_event = stop_event
        self.model = funASR_no_streaming.FunASR(keywords=COMMAND_MAPPING.keys())

    def start(self):
        """启动语音识别进程"""
        print("语音识别进程启动")

        while not self.stop_event.is_set():
            try:
                audio_buffer = self.audio_queue.get(timeout=0.1)
                self.process_audio(audio_buffer)
            except Empty:
                continue

    def process_audio(self, audio_buffer):
        """处理音频数据"""
        print('-' * 10)
        # 计算传入语音的时间长度
        duration = len(audio_buffer) / 16000 / 2  # RATE/2
        print(f"录音时长: {duration:.2f}秒")

        # 记录识别开始时间
        start_time = time.time()
        result = self.model.generate(audio_buffer)
        recognition_time = time.time() - start_time

        print("识别结果:", result)
        print(f"识别用时: {recognition_time:.4f}秒")
        self.map_to_execution(result)

    def map_to_execution(self, recognition_result):
        """映射识别结果到执行命令"""
        if not recognition_result:
            return
        result_ = recognition_result[0]
        key = result_['text'].replace(" ", "")
        if key in COMMAND_MAPPING.keys():
            self.command_queue.put(key)