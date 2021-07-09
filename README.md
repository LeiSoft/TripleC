# MFF4TriC
### Multi-Feature Fusion method for Citation Context Classfication
@CCIR2021

基于[kashgari](https://github.com/BrikerMan/Kashgari) /  [transformers](https://huggingface.co/transformers/)  
Features：多特征融合，多分类任务（迁移学习），多种预训练语言模型适配

> requirements  
>- tensorflow 2.*
>- kashgari 2.0.1

> 更新日志（from beta v0.2）：  
> 2021.06.18 (β v0.2)  
>> 大版本 实现XLNet，MPNet微调，优化

> 项目结构：  
>- run.py 程序入口
>- utils.py 数据读取，输入数据构建
>- train.py 模型训练（高层封装）
>- preprocess.py 数据预处理
>- datasets 数据集
>- features
>>- extractor.py 从数据中抽取特征
>>- features_layers.py 特征融合层
>- models 模型输出、模型结构
>- pretrain.py 预训练语言模型
>> covert_pytorch_checkpoit_to_tf2.py 根据huggingface transformers改进的pytorch模型转tensorflow模型  
>> 预训练工作参见[我的博客](http://hikki.top/2021/03/29/%e5%a6%82%e4%bd%95%e8%ae%ad%e7%bb%83%e4%b8%80%e4%b8%aa%e7%ae%80%e5%8d%95%e7%9a%84bert%e8%af%ad%e8%a8%80%e6%a8%a1%e5%9e%8b%ef%bc%88%e6%94%af%e6%8c%81pytorchtf1-tf2/)  
>> pretrain_google [谷歌官方tf1.*的bert预训练](https://github.com/google-research/bert)
>- kashgari_local
>>对kashgari源码进行修改以适配多特征输入以及多任务功能
>- reference 杂项

### kashgari源码修改说明  
满足了向模型输入多个Input的需求  
完全适配多个标签的分类任务之间的迁移学习，增加了task_num参数，只需要指定任务数量即可  
增加了xlnet的embedding和MPNet的embedding功能  
**注意，如果需要选择huggingface模型的指定隐层需要指定config  "output_hidden_states": true**  
