
import time

from funasr import AutoModel

# 在回调函数或处理线程中
import numpy as np
import wave
import os
from datetime import datetime


class FunASR:
    def __init__(self, model_name="iic/speech_charctc_kws_phone-xiaoyun", keywords="小云小云,发波,升龙", output_dir="./outputs/debug",
                 device='cuda:0'):
        self.model = AutoModel(
            model=model_name,
            keywords=keywords,
            output_dir=output_dir,
            device=device
        )
        # 确保输出目录存在
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, input):
        # 保存输入音频到WAV文件
        # self._save_to_wav(input)

        # audio_data = np.frombuffer(input, dtype=np.int16)  # 假设是16位格式
        # audio_data = audio_data.astype(np.float32) / 32768.0  # 转换为float32并归一化
        res = self.model.generate(input=input, cache={})
        return res

    def _save_to_wav(self, audio_data):
        """将音频数据保存为WAV文件"""
        # 生成基于时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"audio_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)

        # 创建WAV文件
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)   # 16位音频，2字节
            wav_file.setframerate(8000)  # 采样率
            wav_file.writeframes(audio_data)

        print(f"音频已保存到: {filepath}")