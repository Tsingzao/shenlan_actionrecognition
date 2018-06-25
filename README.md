# shenlan_actionrecognition

1、从https://pan.baidu.com/s/1c3J0rhI#list/path=%2F 下载预处理好的视频数据，或者原始视频帧数据；更新：2018/6/25

2、如果下载的是处理后的视频h5数据，则直接进行下一步；否则，按照提供的VideoPreProcessingAndDataAugmentation.py文件对视频进行预处理以及转存成h5文件；

3、按照two_stream.py文件提供的代码框架，编写完成网络结构，并训练和测试数据；


NOTE：

可参考twostream.py文件编写two_stream.py代码，且C3D.py文件提供了课程中讲解的三维卷积网络结构。

caffe版C3D参考：https://github.com/facebook/C3D

课上出现的问题在于我们服务器cudnn5.1对于cuda8.0的版本不对应，修改之后就可跑起。
