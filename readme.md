## 环境配置说明
1. 安装d2l、torch、torchvision等库，可通过清华源 https://pypi.tuna.tsinghua.edu.cn/simple 安装
2. 安装相应的CUDA版本，使得模型可以在GPU上进行训练

## 脚本说明
1. dataset.py负责下载并解压aclIMDB数据集,提供加载数据集的函数load_data_imdb()，其主函数仅需执行一次。
2. bi_rnn.py和text_cnn.py定义了两个用于情感分类的神经网络模型
3. train.py负责模型训练，包括超参数设置和训练方法的选择
4. predict.py负责测试应用训练好的模型进行预测
5. deal_fig.py用于保存和加载训练迭代的走势图

## 脚本运行顺序
1. dataset.py
2. train.py
3. deal_fig.py
4. predict.py

## 可能需要的库源代码修改:
1. 在d2l.torch.TokenEmbedding._load_embedding中：<br>
&emsp;&emsp;在with open(os.path.join(data_dir, 'vec.txt'), 'r')语句的参数表中补上 encoding='utf-8'