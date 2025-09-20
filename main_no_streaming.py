import multiprocessing as mp
import time
import threading
from audio_capture import AudioCapture
from command_executor import CommandExecutor
from speech_recognition import SpeechRecognition


# ==================== 主程序 ====================
def main():
    print("启动语音控制街霸系统...")

    # 创建进程间通信队列
    audio_queue = mp.Queue(maxsize=100)  # 音频数据队列
    command_queue = mp.Queue(maxsize=10)  # 指令队列
    stop_event = mp.Event()

    try:
        # 创建并启动音频采集进程
        audio_capture = AudioCapture(audio_queue, stop_event)
        audio_process = mp.Process(target=audio_capture.start)
        audio_process.start()

        # 创建并启动语音识别进程
        speech_recognition = SpeechRecognition(audio_queue, command_queue, stop_event)
        recognition_process = mp.Process(target=speech_recognition.start)
        recognition_process.start()

        # 创建并启动指令执行线程
        command_executor = CommandExecutor(command_queue, stop_event)
        command_thread = threading.Thread(target=command_executor.start)
        command_thread.start()

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
