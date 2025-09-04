# coding = utf-8

import os
from modelscope.utils.hub import read_config
from modelscope.utils.hub import snapshot_download
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

def main():
    enable_training = True
    enable_testing = True

    # s1 手动创建一个本地工作目录，然后配置到work_dir，用于保存所有训练过程产生的文件
    work_dir = './test_kws_training'

    # s2 获取小云模型库中的配置文件，包含训练参数信息，模型ID确保为'damo/speech_charctc_kws_phone-xiaoyun'
    model_id = 'iic/speech_charctc_kws_phone-xiaoyun'
    model_dir = snapshot_download(model_id)
    print('模型地址：',model_dir)
    configs = read_config(model_id)
    # update some configs
    configs.train.max_epochs = 10
    configs.preprocessor.batch_conf.batch_size = 256
    configs.train.dataloader.workers_per_gpu = 4
    configs.evaluation.dataloader.workers_per_gpu = 4

    config_file = os.path.join(work_dir, 'config.json')
    configs.dump(config_file)

    # s3 初始化一个近场唤醒训练器，trainer tag为'speech_kws_fsmn_char_ctc_nearfield'
    kwargs = dict(
        model=model_id,
        work_dir=work_dir,
        cfg_file=config_file,
        seed=666,
    )
    trainer = build_trainer(
        Trainers.speech_kws_fsmn_char_ctc_nearfield, default_args=kwargs)

    # s4 配置准备好的训练数据列表(kaldi风格)，音频列表分为train/cv，标注合为一个文件，然后启动训练。
    train_scp = './example_kws/train_wav.scp'
    cv_scp = './example_kws/cv_wav.scp'
    test_scp = './example_kws/test_wav.scp'
    trans_file = './example_kws/merge_trans.txt'

    train_checkpoint = ''
    test_checkpoint = ''

    if enable_training:
        kwargs = dict(
            train_data=train_scp,
            cv_data=cv_scp,
            trans_data=trans_file,
            checkpoint=train_checkpoint,
            tensorboard_dir='tensorboard',
            need_dump=True,
        )
        trainer.train(**kwargs)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size > 1 and rank != 0:
        enable_testing = False

    # s5 配置唤醒词，多个请使用英文‘,’分隔；配置测试目录和测试数据列表(kaldi风格)，然后启动测试，最终在测试目录生成测试结果文件——score.txt
    if enable_testing:
        keywords = '小云小云'
        test_dir = os.path.join(work_dir, 'test_dir')

        kwargs = dict(
            test_dir=test_dir,
            test_data=test_scp,
            trans_data=trans_file,
            average_num=10,
            gpu=0,
            keywords=keywords,
            batch_size=256,
        )
        trainer.evaluate(test_checkpoint, None, **kwargs)

if __name__ == '__main__':
    main()