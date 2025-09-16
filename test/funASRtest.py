import time
import librosa
import torch
from funasr import AutoModel

model = AutoModel(
    model="model/speech_charctc_kws_phone-xiaoyun",
    keywords="波",
    output_dir="../outputs/debug",
    device='cpu'
)

test_wav = "outputs/debug/b1.wav"

# 手动加载音频数据
audio_data, sample_rate = librosa.load(test_wav, sr=16000)

# 在调用模型之前记录开始时间
start_time = time.time()

# 直接传入音频数据
res = model.generate(input=audio_data, cache={})

# 在调用模型之后记录结束时间
end_time = time.time()

# 计算耗时
elapsed_time = end_time - start_time
print(res)
print(f"Model inference took {elapsed_time} seconds.")
