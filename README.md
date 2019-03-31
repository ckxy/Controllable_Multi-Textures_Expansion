# Controllable multiple textures synthesis


### Requirements

This code is tested under Ubuntu 14.04 and 16.04. The total project can be well functioned under the following environment: 

* python-2.7 
* pytorch-0.3.0 with cuda correctly specified
* cuda-8.0 or 9.0
* other packages under python-2.7

### Acknowledgements

The code is based on project [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We sincerely thank for their great work.

# 可控的多重纹理扩展合成

本项目为本人的硕士学位论文的实现代码，该项目基于Zhou等人提出的非均匀纹理的扩展合成方法[Non-stationary texture synthesis using adversarial expansions](http://vcc.szu.edu.cn/research/2018/TexSyn)，主要创新点为：
1.引入了Inspiration层[Multi-style Generative Network for Real-time Transfer](https://github.com/zhanghang1989/MSG-Net)并改进了生成器网络的结构，使生成器网络的输入变为两张图片。用户可以利用多张纹理来训练网络，使得该网络具有生成多种纹理扩展图片和纹理迁移图片的能力，且用户可以根据选择不同的图片对来控制网络的合成结果。
2.引入了标签向量、分类误差和分类器网络，有效防止生成器网络发生模式崩溃。
