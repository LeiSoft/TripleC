import tensorflow as tf
import argparse
import pickle
import numpy as np

from train import Trainer


if __name__ == '__main__':
    # model_folder = './models/cased_L-12_H-768_A-12'
    parser = argparse.ArgumentParser(description="training model")

    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="transfer model type, [bert, w2v, xlnet, mpnet]")
    parser.add_argument("--task_type", type=str, required=True, default='intent',
                        help="specify attributes for predict [3c-shared/intent;3c-shared/intent;scite;arl-arc]")
    parser.add_argument("--gpu_idx", type=int, required=False, default=3,
                        help="set the gpu")
    # whether need multi_label for input
    parser.add_argument("--multi_label", default=None,
                        help="None or List[str], other type of input will be thought to be None")

    args = parser.parse_args()
    assert args.model_type in ['bert', 'w2v', 'xlnet', 'mpnet', 'bert_hf'], + \
        "only support [bert, w2v, xlnet, mpnet, bert_hf]"

    trainer = Trainer(args)
    # test参数为False则不做测试集验证，只进行模型训练
    gpu_idx = args.gpu_idx if args.gpu_idx else 1
    with tf.device("/gpu:0"):
        if args.task_type == "scicite":
            '''
            task_num为参与训练的任务个数
            当交叉验证方法为fold时，k折交叉 => fold=k
            当方法为random时，fold为随机的次数，种子范围默认为range(fold)
            '''
            # trainer.train_sci_cross("./datasets/scicite/", task_num=2, cross='fold', fold=10, with_feature=True)
            # trainer.train_sci_cross("./datasets/scicite/", task_num=2,
            #                         cross='random', fold=1, t_v_size=2777, v_size=916, with_feature=True)

            # solid_rate 需要在test_rate参数指定test集的标签比例, v_size设置验证集占去除test集之后的所有数据的比例，多任务情况取主任务标签
            trainer.train_sci_cross("./datasets/scicite/", task_num=2, 
                cross='solid_rate', fold=1, test_rate=[259, 997, 605], v_size=0.15, with_feature=True)

        if args.task_type == "acl-arc":
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
            # with mirrored_strategy.scope():
            #     trainer.train_sci_cross("./datasets/acl-arc/", task_num=1, cross='fold', fold=5, v_size=125, with_feature=True)
            # 按照allenai的两篇论文的数据集数量切分 验证+测试=253 验证=114
            # trainer.train_sci_cross("./datasets/acl-arc/", task_num=1, cross='repeat', fold=5, with_feature=True)

            # trainer.train_sci_cross("./datasets/acl-arc/", task_num=1,
            #                         cross='random', fold=20, t_v_size=250, v_size=125, with_feature=True)
            # random: 完全随机，solid：固定测试集，solid_rate：固定测试集的label比例，样本随机选择
            # with mirrored_strategy.scope():
            #     trainer.train_scaffold('./datasets/acl-arc/', data_split_type='solid_rate')
            trainer.train_scaffold('./datasets/acl-arc/', data_split_type='solid_rate')

        if args.task_type in ["3c-shared/intent", "3c-shared/influence", "3c"]:
            # 3C任务多个任务数据集是分离的
            trainer.train_sci_cross("./datasets/3c-shared/intent/", task_num=2,
                                    cross='random', fold=10, t_v_size=1000, v_size=500, with_feature=True)
                                    
            # trainer.train_3c("./datasets/" + args.task_type + "/train.tsv",
            #                  multi_paths=['./datasets/3c-shared/influence/train.tsv'],
            #                  vali_size=0.1, test_size=None, task_num=1)
