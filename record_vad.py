
import pyaudio
import webrtcvad
import wave
import os
import time
from datetime import datetime

# ==================== 配置参数 ====================
# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz对于语音识别足够
CHUNK = 960 * 4  # 小块数据降低延迟 单位：采样点
VAD_AGGRESSIVENESS = 3  # VAD激进程度 (0-3)

# VAD参数
SAMPLES_PER_VAD_FRAME = 480  # VAD期望的帧长度

class VADRecorder:
    def __init__(self, output_dir="./recordings"):
        """
        初始化VAD录音器

        Args:
            output_dir: 录音文件保存目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化PyAudio
        self.audio = pyaudio.PyAudio()

        # 初始化VAD
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        # 录音状态
        self.is_recording = False
        self.audio_buffer = []
        self.silence_count = 0
        self.MAX_SILENCE_FRAMES = 1

    def is_speech(self, audio_data):
        """
        检测音频数据中是否包含人声

        Args:
            audio_data: 音频数据

        Returns:
            bool: True表示检测到人声，False表示未检测到
        """
        # 计算需要拆分成几段
        num_chunks = len(audio_data) // (SAMPLES_PER_VAD_FRAME * 2)  # 16位音频，每个采样点2字节

        # 将音频数据拆分成多个VAD帧
        for i in range(num_chunks):
            start_idx = i * SAMPLES_PER_VAD_FRAME * 2  # 字节偏移量
            end_idx = start_idx + SAMPLES_PER_VAD_FRAME * 2
            chunk_data = audio_data[start_idx:end_idx]

            if self.vad.is_speech(chunk_data, RATE):
                return True

        return False

    def start_recording(self):
        """开始录音"""
        print("开始录音，检测到人声时将自动录制...")

        # 打开音频流
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        try:
            while True:
                # 读取音频数据
                audio_data = stream.read(CHUNK)

                # 检测是否有人声
                if self.is_speech(audio_data):
                    print("检测到人声，开始录制...")
                    self.is_recording = True
                    self.audio_buffer = []
                    self.silence_count = 0

                    # 录制当前帧
                    self.audio_buffer.append(audio_data)

                    # 继续录制直到静音
                    while self.is_recording:
                        audio_data = stream.read(CHUNK)

                        # 检查是否还有人声
                        if self.is_speech(audio_data):
                            self.audio_buffer.append(audio_data)
                            self.silence_count = 0
                        else:
                            self.silence_count += 1

                            # 如果连续几帧没有检测到人声，则停止录制
                            if self.silence_count >= self.MAX_SILENCE_FRAMES:
                                self.is_recording = False
                                self.save_recording()
                                print("检测到人声，开始录制...")
                            else:
                                self.audio_buffer.append(audio_data)
                else:
                    print(".", end="", flush=True)  # 显示等待状态

        except KeyboardInterrupt:
            print("\n录音已停止")
        finally:
            # 清理资源
            stream.stop_stream()
            stream.close()

    def save_recording(self):
        """
        保存录音到WAV文件
        """
        if not self.audio_buffer:
            return

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")

        # 保存为WAV文件
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.audio_buffer))

        print(f"录音已保存: {filename}")

        # 计算录音时长
        duration = len(b''.join(self.audio_buffer)) / RATE / 2
        print(f"录音时长: {duration:.2f}秒")

    def close(self):
        """
        关闭音频资源
        """
        self.audio.terminate()

def main():
    """
    主函数
    """
    print("VAD录音器")
    print("按 Ctrl+C 停止录音")

    recorder = VADRecorder()

    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        print("\n正在停止录音器...")
    finally:
        recorder.close()
        print("录音器已停止")

if __name__ == "__main__":
    main()