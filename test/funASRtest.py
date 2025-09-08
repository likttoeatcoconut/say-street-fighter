import time

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun",
    keywords="发波",
    output_dir="../outputs/debug",
    device='cpu'
)

test_wav = "outputs/debug/audio_20250904_170302_859438.wav"

# 在调用模型之前记录开始时间
start_time = time.time()

res = model.generate(input=test_wav, cache={})

# 在调用模型之后记录结束时间
end_time = time.time()

# 计算耗时
elapsed_time = end_time - start_time
print(res)
print(f"Model inference took {elapsed_time} seconds.")