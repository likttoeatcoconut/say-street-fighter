import time
import numpy as np
import wave
import os
from datetime import datetime
from funasr import AutoModel
import threading

class FunASR:
    def __init__(self, model_name="iic/speech_charctc_kws_phone-xiaoyun", keywords="多达,发波,升龙",
                 output_dir="./outputs/debug", device='cpu', enable_debug=False):
        # 添加启用调试模式的参数
        self.enable_debug = enable_debug

        # 确保输出目录存在（仅在调试模式下需要）
        # self.output_dir = output_dir
        # if self.enable_debug:
        #     os.makedirs(self.output_dir, exist_ok=True)

        # 预加载模型
        print("正在加载语音识别模型...")
        start_time = time.time()
        self.model = AutoModel(
            model=model_name,
            keywords=keywords,
            output_dir=output_dir,
            device=device
        )
        print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

        # 添加缓存机制
        self.cache = {}
        self.last_audio_data = None
        self.last_result = None

        # 添加批量处理队列
        self.audio_queue = []
        self.processing = False
        self.lock = threading.Lock()

    def generate(self, input_data):
        """优化后的生成方法"""
        # 检查是否与上次音频相同（避免重复处理）
        if self.last_audio_data == input_data and self.last_result is not None:
            return self.last_result

        # 更新缓存
        self.last_audio_data = input_data

        # 仅在调试模式下保存音频
        # if self.enable_debug:
        #     self._save_to_wav(input_data)

        try:
            # 使用缓存进行流式识别
            # 在调用模型之前记录开始时间
            start_time = time.time()
            res = self.model.generate(input=input_data, cache=self.cache)
            # 在调用模型之后记录结束时间
            end_time = time.time()
            # 计算耗时
            elapsed_time = end_time - start_time
            print(f"Model inference took {elapsed_time} seconds.")
            self.last_result = res
            return res
        except Exception as e:
            print(f"语音识别错误: {e}")
            # 重置缓存以防错误累积
            self.cache = {}
            return []

    def generate_batch(self, input_data):
        """批量处理音频数据（可选）"""
        with self.lock:
            self.audio_queue.append(input_data)

            # 如果已经有处理线程运行，则直接返回
            if self.processing:
                return None

            self.processing = True

        # 在新线程中处理批量音频
        threading.Thread(target=self._process_batch).start()
        return None

    def _process_batch(self):
        """处理批量音频数据"""
        with self.lock:
            batch_data = self.audio_queue
            self.audio_queue = []

        try:
            # 批量处理音频数据
            if batch_data:
                # 合并音频数据
                combined_audio = b''.join(batch_data)

                # 仅在调试模式下保存音频
                if self.enable_debug:
                    self._save_to_wav(combined_audio)

                # 处理合并后的音频
                res = self.model.generate(input=combined_audio, cache=self.cache)
                self.last_result = res

                # 处理识别结果
                self._handle_results(res)

        except Exception as e:
            print(f"批量语音识别错误: {e}")
            self.cache = {}

        with self.lock:
            self.processing = False

    def _handle_results(self, results):
        """处理识别结果（可根据需要自定义）"""
        for result in results:
            if 'text' in result:
                print(f"识别结果: {result['text']}")
                # 这里可以添加更多处理逻辑

    def _save_to_wav(self, audio_data):
        """将音频数据保存为WAV文件（仅在调试模式下使用）"""
        if not self.enable_debug:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"audio_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)  # 注意：这里应该是16000，不是8000
                wav_file.writeframes(audio_data)

            print(f"音频已保存到: {filepath}")
        except Exception as e:
            print(f"保存音频文件失败: {e}")

    def reset_cache(self):
        """重置缓存（在检测到语音间隔时调用）"""
        self.cache = {}
        self.last_audio_data = None
        self.last_result = None

# 使用示例
if __name__ == "__main__":
    # 初始化模型（禁用调试模式以提高性能）
    asr_model = FunASR(enable_debug=False)

    # 模拟音频处理
    dummy_audio = b'\x00' * 3200  # 200ms的16kHz音频

    # 测试识别速度
    start_time = time.time()
    for i in range(10):
        result = asr_model.generate(dummy_audio)
        print(f"识别 {i+1}: {result}")

    print(f"平均识别时间: {(time.time() - start_time)/10:.3f}秒")