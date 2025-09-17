import os
import threading
import time

import numpy as np
from funasr import AutoModel


class FunASRStreaming:
    def __init__(self, model_name="paraformer-zh-streaming",
                 chunk_size=None,
                 encoder_chunk_look_back=4,
                 decoder_chunk_look_back=1,
                 output_dir="./outputs/debug",
                 device='cpu'
                 ):
        # 添加启用调试模式的参数
        if chunk_size is None:
            chunk_size = [0, 4, 4]
        self.chunk_size = chunk_size
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.chunk_stride = chunk_size[1] * 960  # 计算块步长
        self.output_dir = output_dir
        # 预加载模型
        print("正在加载流式语音识别模型...")
        start_time = time.time()
        self.model = AutoModel(
            model=model_name,
            device=device
        )
        print(f"流式模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        # 添加缓存机制
        self.cache = {}

    def generate(self, speech_chunk, is_final):
        """

        :param speech_chunk: 二进制音频数据
        :param is_final:
        :return:
        """
        result = self.model.generate(input=speech_chunk, cache=self.cache, is_final=is_final,
                                     chunk_size=self.chunk_size, encoder_chunk_look_back=self.encoder_chunk_look_back,
                                     decoder_chunk_look_back=self.decoder_chunk_look_back)
        return result

    def generate_proxy(self,  speech_chunk, is_final):
        """
        不要求传入数据大小
        :param speech_chunk: 二进制音频数据
        :param is_final: 一句话是否结束
        :return:
        """
        pass
