import time

from funasr import AutoModel

# 在回调函数或处理线程中
import numpy as np


class FunASR:
    def __init__(self, model_name="iic/speech_charctc_kws_phone-xiaoyun", keywords="发波,升龙", output_dir="./outputs/debug",
                 device='cuda:0'):
        self.model = AutoModel(
            model=model_name,
            keywords=keywords,
            output_dir=output_dir,
            device=device
        )

    def generate(self, input):
        audio_data = np.frombuffer(input, dtype=np.int16)  # 假设是16位格式
        audio_data = audio_data.astype(np.float32) / 32768.0  # 转换为float32并归一化
        res = self.model.generate(input=audio_data, cache={}, sampling_rate=16000)
        return res
