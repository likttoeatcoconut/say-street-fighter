import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
# 注意：这里 model 参数指向的是你训练的工作目录，而不是之前的模型ID
my_kws_model_path = 'model\speech_charctc_kws_phone-xiaoyun'  # 你的 work_dir


kwsbp_16k_pipline = pipeline(
    task=Tasks.keyword_spotting,
    model=my_kws_model_path,
    pipeline_name='kws-kwsbp',
    model_revision=None,
    local_model=True
)

kws_result = kwsbp_16k_pipline(audio_in='./outputs/debug/s1.wav')
print(kws_result)