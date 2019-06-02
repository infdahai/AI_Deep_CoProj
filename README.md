# AI_Deep_CoProj
description : cooperative deep learning project

# Target

We build Grapht-Text-Graph Project in this work.

# Environment

## 类脑平台
url: https://www.bitahub.com/beta/  
username: isaacc@mail.ustc.edu.cn  
password: W8h268

## Colab (recommend platform)
url: https://colab.research.google.com

# Related Sites

1. search paper : [arxiv](https://arxiv.org/list/cs/recent)
2. Text to Graph:  
https://nthu-datalab.github.io/ml/competitions/04_Reverse-Image-Caption/04_Reverse_Image_Caption.html
3. Graph to Text:  
潜在url:  
https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8
4. Compose

# Process

5.22 - now

  train code ( reverse image caption)  
  数据集下载link : http://rec.ustc.edu.cn/share/3dd85e90-7c6d-11e9-8455-af8d7307775b
  
  **Problems**
    case 1:
    数据集在类脑平台的上传和在Google colab的上传均未成功，因此目前在自己的电脑上跑
    case2：
    代码在我的电脑跑的时候在Iterate the data_iterator这部分会自动终止，编译时pycharm的报错：
    Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
    Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
    OP_REQUIRES failed at whole_file_read_ops.cc:114 : Not found: NewRandomAccessFile failed to Create/Open: /102flowers/image_08110.jpg : ϵͳ�Ҳ���ָ����·����; No such process
    在网上搜到的方法没什么效果，暂时卡在这，应该需要对代码细节仔细看才能解决。
    
  **hint**
    在本地下载好数据集的时候直接解压到已创建的pyhton项目上，此外可以用jupyter调试初期的一些问题，但注意jupyter不能解压文件
    
    配置本地环境：
    下载tensorflow cpu版（若有英伟达显卡可以下载gpu版），下载keras包，tensorlayer包（注意此包需要与tensorflow的版本对应，cpu版对应1.14左右版本，gpu版对应最新的2.1.0版），此外还需在下载nltk包后用nltk.download（）下载punc包。建议用anaconda的conda指令下载，pip太慢。
    
    其他的问题可自行解决
