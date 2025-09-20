import time
import wave
import os
from datetime import datetime
from funasr import AutoModel
import threading
import traceback

class FunASR:
    def __init__(self, output_dir="./outputs/debug", device='cpu', keywords=None):
        # 预加载模型

        self.keywords = " ".join(keywords)
        # self.keywords = keywords

        print(self.keywords)
        print("正在加载语音识别模型...")
        start_time = time.time()
        self.model = AutoModel(
            model='paraformer-zh',
            output_dir=output_dir,
            device=device
        )
        print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

    def generate(self, input_data):
        # if self.enable_debug:
        #     self._save_to_wav(input_data)

        try:
            start_time = time.time()
            res = self.model.generate(input=input_data, batch_size_s=300, hotword=self.keywords)
            # 在调用模型之后记录结束时间
            end_time = time.time()
            # 计算耗时
            elapsed_time = end_time - start_time
            print(f"Model inference took {elapsed_time} seconds.")
            return res
        except Exception as e:
            print(f"语音识别错误: {e}")
            traceback.print_exc()
            # 重置缓存以防错误累积
            return []

    def _save_to_wav(self, audio_data):
        """将音频数据保存为WAV文件（仅在调试模式下使用）"""
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


# 使用示例
if __name__ == "__main__":
    # 初始化模型（禁用调试模式以提高性能）
    asr_model = FunASR()
    path = 'recordings/2mk.wav'
    audio_data = wave.open(path, 'rb').readframes(-1)
    rs = asr_model.generate(audio_data)
    print(rs)
