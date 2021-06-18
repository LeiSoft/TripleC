from train import Trainer

import tensorflow as tf
import argparse
import pickle


def repeat_run(k):
    reports = []
    for i in range(k):
        reports.append(trainer.train_scicite("./datasets/acl-arc/", task_num=1))
    acc, p, r, f1 = [], [], [], []
    with open("./datasets/acl-arc/repeat_res.tsv", 'w', encoding='utf-8') as f:
        for report in reports:
            if isinstance(report, list):
                acc.append(report[0]['detail']['accuracy'])
                p.append(report[0]['detail']['macro avg']['precision'])
                r.append(report[0]['detail']['macro avg']['recall'])
                f1.append(report[0]['detail']['macro avg']['f1-score'])
            else:
                acc.append(report['detail']['accuracy'])
                p.append(report['detail']['macro avg']['precision'])
                r.append(report['detail']['macro avg']['recall'])
                f1.append(report['detail']['macro avg']['f1-score'])
        res = "k={}\n{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\n{}\t{}\t{}\t{}\n". \
            format(k, sum(acc) / k, sum(p) / k, sum(r) / k, sum(f1) / k,
                   max(acc), max(p), max(r), max(f1),
                   min(acc), min(p), min(r), min(f1))
        f.write(res)

    pickle.dump((acc, p, r, f1), open("./datasets/acl-arc/details.pkl", 'wb'))
    # if (sum(f1) / k) > 0.70:
    #     print(sum(f1)/k)
    #     exit(99)


if __name__ == '__main__':
    # model_folder = './models/cased_L-12_H-768_A-12'
    parser = argparse.ArgumentParser(description="training model")

    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='bert',
                        help="transfer model type, [bert, w2v, xlnet, mpnet]")
    parser.add_argument("--task_type", type=str, required=True, default='intent',
                        help="specify attributes for predict [3c-shared/intent;3c-shared/intent;scite;arl-arc]")
    # whether need multi_label for input
    parser.add_argument("--multi_label", default=None,
                        help="None or List[str], other type of input will be thought to be None")

    args = parser.parse_args()
    assert args.model_type in ['bert', 'w2v', 'xlnet', 'mpnet'], + \
        "only support [bert, w2v, xlnet, mpnet]"

    trainer = Trainer(args)
    # test参数为False则不做测试集验证，只进行模型训练
    with tf.device("/gpu:2"):
        if args.task_type == "scicite":
            '''
            task_num为参与训练的任务个数
            当交叉验证方法为fold时，k折交叉 => fold=k
            当方法为random时，fold为随机的次数，种子范围默认为range(fold)
            '''
            # trainer.train_scicite_cross("./datasets/scicite/", task_num=2, cross='fold', fold=10)
            # trainer.train_scicite_cross("./datasets/scicite/", task_num=2,
            #                             cross='random', fold=10, t_v_size=2777, v_size=916)
            trainer.train_scicite("./datasets/scicite/", task_num=1)

        if args.task_type == "acl-arc":
            repeat_run(5)
            # trainer.train_scicite("./datasets/acl-arc/", task_num=1)
            # trainer.train_scicite_cross("./datasets/acl-arc/", task_num=1, cross='fold', fold=5)
            # 按照allenai的两篇论文的数据集数量切分 验证+测试=253 验证=114
            # while True:
            #     trainer.train_scicite_cross("./datasets/acl-arc/", task_num=1,
            #                                 cross='random', fold=10, t_v_size=253, v_size=114)

        if args.task_type in ["3c-shared/intent", "3c-shared/influence"]:
            # 3C任务多个任务数据集是分离的
            trainer.train_3c("./datasets/" + args.task_type + "/train.tsv",
                             multi_paths=['./datasets/3c-shared/influence/train.tsv'],
                             vali_size=0.1, test_size=None, task_num=2)
