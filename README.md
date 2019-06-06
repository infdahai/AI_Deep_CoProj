# AI_Deep_CoProj
description : cooperative deep learning project

# Target

We build Grapht-Text-Graph Project in this work.

# Environment

## 类脑平台
url: https://www.bitahub.com:3443/beta/login  
username: isaacc@mail.ustc.edu.cn  
password: W8h268

## Colab (recommend platform)
url: https://colab.research.google.com

# Related Sites

1. search paper : [arxiv](https://arxiv.org/list/cs/recent)

2. Text to Graph:  
https://nthu-datalab.github.io/ml/competitions/04_Reverse-Image-Caption/04_Reverse_Image_Caption.html

3. Graph to Text:  
  [潜在url](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)<br>
  [完整源码](https://github.com/hlamba28/Automatic-Image-Captioning)
  需要改动的地方是读取数据集的部分<br>
  ［✓］这部分已经在6/5完成 [参考](./Image2Caption.ipynb)
  
  

4. Compose

# Process

5.22 - 6.2

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
    
  **hint：**
    在本地下载好数据集的时候直接解压到已创建的pyhton项目上，此外可以用jupyter调试初期的一些问题，但注意jupyter不能解压文件
    
    配置本地环境：

    下载tensorflow cpu版（若有英伟达显卡可以下载gpu版），下载keras包，tensorlayer包（注意此包需要与tensorflow的版本对应，cpu版对应1.14左右版本，gpu版对应最新的2.1.0版），此外还需在下载nltk包后用nltk.download（）下载punc包。建议用anaconda的conda指令下载，pip太慢。
    
    其他的问题可自行解决
    
 
 6.2 - now
  
上传reverse文件夹下main.ipynb和.py文件，均可运行。如若出现 类似软件包undefined情况，可以重新执行最后一个block.
    
目前情况如下(执行最后一个块时):
    
 ```
    WARNING:tensorflow:From /home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/main.py:141: LSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.LSTMCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/main.py:147: dynamic_rnn (from tensorflow.python.ops.rnn) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell)`, which is equivalent to this API
WARNING:tensorflow:From /home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/tensorflow/contrib/layers/python/layers/layers.py:1624: flatten (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.flatten instead.
WARNING:tensorflow:From /home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/main.py:167: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.dense instead.
WARNING:tensorflow:From /home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/tensorflow/python/ops/array_grad.py:425: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch: [ 0/ 5] [   0/ 115] time: 0.8072s, d_loss: 1.370, g_loss: 0.740
Epoch: [ 0/ 5] [  50/ 115] time: 0.2090s, d_loss: 1.786, g_loss: 0.504
Epoch: [ 0/ 5] [ 100/ 115] time: 0.2009s, d_loss: 2.736, g_loss: 0.742
Epoch: [ 1/ 5] [   0/ 115] time: 0.1495s, d_loss: 2.005, g_loss: 0.099
Epoch: [ 1/ 5] [  50/ 115] time: 0.1483s, d_loss: 1.662, g_loss: 0.787
Epoch: [ 1/ 5] [ 100/ 115] time: 0.1513s, d_loss: 2.849, g_loss: 5.220
Epoch: [ 2/ 5] [   0/ 115] time: 0.1513s, d_loss: 2.107, g_loss: 0.691
Epoch: [ 2/ 5] [  50/ 115] time: 0.1508s, d_loss: 2.009, g_loss: 0.406
Epoch: [ 2/ 5] [ 100/ 115] time: 0.1538s, d_loss: 2.522, g_loss: 3.219
Epoch: [ 3/ 5] [   0/ 115] time: 0.1570s, d_loss: 5.046, g_loss: 1.719
Epoch: [ 3/ 5] [  50/ 115] time: 0.1553s, d_loss: 1.665, g_loss: 3.649
Epoch: [ 3/ 5] [ 100/ 115] time: 0.1475s, d_loss: 1.328, g_loss: 1.443
Epoch: [ 4/ 5] [   0/ 115] time: 0.1508s, d_loss: 1.910, g_loss: 1.140
Epoch: [ 4/ 5] [  50/ 115] time: 0.1498s, d_loss: 1.483, g_loss: 0.783
Epoch: [ 4/ 5] [ 100/ 115] time: 0.1483s, d_loss: 1.249, g_loss: 1.665
-----success saved checkpoint--------
Traceback (most recent call last):
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/IPython/core/interactiveshell.py", line 3296, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-3-549d33b68efc>", line 7, in <module>
    gan.training()
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/main.py", line 359, in training
    self._sample_visiualize(_epoch)
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/main.py", line 410, in _sample_visiualize
    save_images(img_gen, [ni, ni], self.sample_path + '/train_{:02d}.png'.format(epoch))
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/utils.py", line 73, in save_images
    return imsave(images, size, image_path)
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/utils.py", line 70, in imsave
    return scipy.misc.imsave(path, merge(images, size))
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/numpy/lib/utils.py", line 101, in newfunc
    return func(*args, **kwds)
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/scipy/misc/pilutil.py", line 219, in imsave
    im.save(name)
  File "/home/cluster/Downloads/datalabcup-reverse-image-caption-ver2/venv/lib/python3.5/site-packages/PIL/Image.py", line 1925, in save
    fp = builtins.open(filename, "w+b")
FileNotFoundError: [Errno 2] No such file or directory: './samples/train_04.png'

 ```

发现samples以及 inference文件夹位置不明，需要尽快了解。否则可能需要尽快换方案尝试。

  **hint** 
    在本机环境运行时，需要将代码中path 改为本机所在工程的根文件夹。  
    环境配置: 注意 tensorlayer == 1.11.0 (上2.0只支持tf2.0，这里统一确定该包版本)
            其他可自由参考            
```
              tensorboard                   1.13.1                
                tensorflow                    1.13.1                
                tensorflow-estimator          1.13.0                
                 tensorlayer                   1.11.0  
```
              
