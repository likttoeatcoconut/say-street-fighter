import pyaudio
import webrtcvad
import time
import numpy as np
import keyboard  # 需要安装: pip install keyboard
import funASR

# ==================== 配置参数 ====================
# 音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz对于语音识别足够
CHUNK = int(RATE * 0.02)  # 20ms块，平衡延迟和处理效率
VAD_AGGRESSIVENESS = 3  # VAD激进程度 (0-3)

# 语音指令映射
COMMAND_MAPPING = {
    "hadouken": ["down", "down", "right", "p"],
    "shoryuken": ["right", "down", "right", "p"],
    "sonicboom": ["back", "back", "p"],
    "波": ["s", ("s", "d"), "d", "o"],
    "升": ["d", ("s", "d"), "s", ("s", "d"),"d","i"],
    # 添加更多指令...
}

# ==================== 主程序 ====================
def main():
    print("启动语音控制街霸系统...")

    # 初始化组件
    p = pyaudio.PyAudio()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    model = funASR.FunASR()

    # 查找并打开音频设备
    device_index = p.get_default_input_device_info()["index"]
    print(f"使用音频设备: {device_index}")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )

    # 音频处理状态变量
    audio_buffer = b''
    silence_frames = 0
    MAX_SILENCE_FRAMES = 1  # 15*20ms=300ms静音判定结束
    is_recording = False

    print("系统就绪，开始监听...")

    try:
        while True:
            # 读取音频数据
            data = stream.read(CHUNK, exception_on_overflow=False)

            # VAD检测
            is_speech = False
            try:
                is_speech = vad.is_speech(data, RATE)
            except:
                pass  # 忽略VAD异常

            # 语音活动检测状态机
            if is_speech:
                silence_frames = 0
                is_recording = True
                audio_buffer += data
            elif is_recording:
                silence_frames += 1
                audio_buffer += data

                # 检测到足够长的静音，处理音频
                if silence_frames > MAX_SILENCE_FRAMES:
                    # 执行语音识别
                    result = model.generate(audio_buffer)
                    print("识别结果:", result)

                    # 映射到执行命令
                    if result and len(result) > 0:
                        text = result[0].get('text', '')
                        for key in COMMAND_MAPPING.keys():
                            if key in text:
                                print(f"执行指令: {key}")
                                execute_command(COMMAND_MAPPING[key])

                    # 重置状态
                    audio_buffer = b''
                    silence_frames = 0
                    is_recording = False

    except KeyboardInterrupt:
        print("正在停止系统...")
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("系统已停止")

def execute_command(keys):
    """执行按键命令"""
    for key in keys:
        if isinstance(key, tuple):
            # 处理同时按键的情况
            for k in key:
                keyboard.press(k)
            time.sleep(0.02)
            for k in key:
                keyboard.release(k)
        else:
            # 处理单个按键的情况
            keyboard.press(key)
            time.sleep(0.02)
            keyboard.release(key)

if __name__ == "__main__":
    main()