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
            chunk_size = [0, 1, 4]
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

        self.cache = {}
        # 初始化缓存池
        self.buffer_pool = np.array([], dtype=np.float32)

    def generate(self, speech_chunk, is_final):
        """

        :param speech_chunk: 音频数据，大小要求为 chunk_size[1] * 960
        :param is_final:
        :return:
        """
        result = self.model.generate(input=speech_chunk, cache=self.cache, is_final=is_final,
                                     chunk_size=self.chunk_size, encoder_chunk_look_back=self.encoder_chunk_look_back,
                                     decoder_chunk_look_back=self.decoder_chunk_look_back)
        return result

    def generate_proxy(self, speech_chunk, is_final):
        """
        不要求传入数据大小
        :param speech_chunk: 音频数据
        :param is_final: 一句话是否结束
        :return:
        """
        # 将新数据添加到缓存池
            
        self.buffer_pool = np.concatenate((self.buffer_pool, speech_chunk))
        
        results = []
        # 当缓存池中的数据足够大或者这是最后一块数据时，进行处理
        while len(self.buffer_pool) >= self.chunk_stride or (is_final and len(self.buffer_pool) > 0):
            # 如果是最后一块且剩余数据不足一个完整块，也进行处理
            if is_final and len(self.buffer_pool) < self.chunk_stride:
                # 用零填充到合适的大小
                padded_chunk = np.zeros(self.chunk_stride, dtype=np.float32)
                padded_chunk[:len(self.buffer_pool)] = self.buffer_pool
                chunk_to_process = padded_chunk
                self.buffer_pool = np.array([], dtype=np.float32)  # 清空缓存池
            else:
                # 取出一个完整的块进行处理
                chunk_to_process = self.buffer_pool[:self.chunk_stride]
                self.buffer_pool = self.buffer_pool[self.chunk_stride:]  # 更新缓存池
                
            # 判断是否为最终块（缓存池为空且是最后一块数据）
            is_final_chunk = is_final and len(self.buffer_pool) == 0
            
            # 调用原始generate方法处理数据块
            result = self.generate(chunk_to_process, is_final_chunk)
            results.append(result)
            
        # 如果有结果，返回最后一个结果；否则返回空列表
        return results[-1] if results else []