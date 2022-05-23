### CRNN_Captcha

---

#### 1. 项目介绍

##### 1.1 背景

- 学校体育场地的预定接口自 2021 年添加了图形验证码，之前学长遗留下的代码不能使用了

- 临时用了 `tesseract-ocr` 现成的接口，效果不佳，勉强能用
- 高准确率的验证码识别模型便成了需求
- 上学期人工神经网络课程的大作业有个选题是用 `jittor` 复现 `CRNN`，看其他组做的效果不错，正确率能达到 90%+，于是就想着什么时候更新一下 “祖传” 抢场地代码
- 在这学期的某天，突然想起这件事，于是在 `CRNN` 的源代码基础上进行一定的修改，得到了效果不错的结果

##### 1.2 验证码类型

- 以下是需要处理的验证码图片（width：200 px，height：50 px）

  <img src="image\0_mh3I.jpg" alt="0_mh3I" />

  <img src="image\2_Llw7.jpg" alt="2_Llw7" />
  
  <img src="image\0_mh3I.jpg" alt="0_mh3I" />

- 字符集：阿拉伯数字、大小写字母（区分大小写）

- 恰好这学期自己写的一个网站用的也是这个， Java 下的 `Kaptcha` 

##### 1.3 模型

- CRNN 论文：["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2016), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717)

- CRNN 详解：[https://www.ycc.idv.tw/crnn-ctc.html](https://www.ycc.idv.tw/crnn-ctc.html)

- CRNN 模型：

  <img src="image\crnn_structure.png" alt="crnn_structure" style="zoom: 33%;" />



#### 2. 数据集

- 用 `Kaptcha` 设置好合适的参数，随机生成数据即可
- 训练集：1000000 张图片
- 验证集：100000 张图片
- 测试集：100000 张图片
- 将图片命名为 `[index]_[text].jpg`，例如 `123_Rf5k.jpg`
- 下载链接：https://cloud.tsinghua.edu.cn/d/701bb82138bc41a389b9/



#### 3. 如何使用

##### 3.1 设置

- 如果你想使用其他数据集，需要对 `VerifyDataset` 进行适当的修改

##### 3.2 训练

- `python train.py`
- 用 `RTX 1070Ti` 训几十分钟后就有很显著的成效
- 一开始的 `learning_rate` 为 `0.0005`，当训练到正确率为 95% 左右时，可以适当降低
- 最终模型训了大致 4000000 张图片的数据，在验证集有 99.99% 的正确率
- （因为不是作业，所以写的很随性，没有给出具体数据）

##### 3.3 测试

- `python evaluate.py`
- 最终模型在测试集上有 99.9% 的正确率
- （因为用的是同一个程序生成的数据，所以特征极为相似）

##### 3.4 识别

- `python predict.py`
- 一张图片的识别
- 写得挺随性，需要手动去修改一下文件中一些变量

































