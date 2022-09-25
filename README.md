# algorithm-cpp
- algorithm-cpp projects
- 提供大量的算法C++学习案例程序
- 以算法的角度学习所必须的C++知识

# 内容包括
1. 基本的编译、g++、makefile
2. 文件格式的解析，解析zip、解析mnist，以及保存tensor让python读取
3. 基本语法，数据类型、指针、函数等
4. 用C++实现矩阵乘法
5. 用C++实现求解根号x，梯度下降法、牛顿法
6. 线性回归
7. 逻辑回归
8. 逻辑回归，基于矩阵法
9. 逻辑回归，基于高斯牛顿法
10. BP反向传播
11. 标量自动微分
12. 向量自动微分
13. 矩阵自动微分
14. 基于矩阵自动微分下的BP实现
15. 基于矩阵自动微分下的CNN实现
16. 一个旋转相册案例
17. 3d渲染之模型加载并显示
18. 3d渲染之模型加载并显示，配光照系统

# 使用方法-自行配置环境
1. 案例均使用makefile作为编译工具
    - 在其中以`${@CUDA_HOME}`此类带有@符号表示为特殊变量
    - 替换此类特殊变量为你系统真实环境，即可顺利使用
2. 大部分时候，配置完毕后，可以通过`make run`实现编译运行

# 使用方法-自动配置环境
1. 要求linux-ubuntu16.04以上系统
2. 安装python包，`pip install akiwi -U -i https://pypi.org/simple`
3. 配置快捷方式，`echo alias kiwi=\"python -m atrtpy\" >> ~/.bashrc`
4. 应用快捷方式：`source ~/.bashrc`
5. 配置key：`kiwi set-key sxaikiwik`
6. 下载案例`kiwi get-templ algorithm-cpp-02nn-14-autograd-matrix-bp`
7. 编译运行`make run`
8. 如果key不可用，请加微信 shouxie_ai 入群后索取

# Reference
- 交流QQ群：686852956
- 交流微信：shouxie_ai
- 课程录屏-pptx: https://pan.baidu.com/s/1_cL7C8PRxolHhM-eQanBhA?pwd=1237 
- 相对应其他的课程：https://github.com/shouxieai/learning-cuda-trt
- TensorRT的B站视频讲解：https://www.bilibili.com/video/BV1Xw411f7FW
- 官方的视频讲解：https://www.bilibili.com/video/BV15Y4y1W73E
- trtpy前期介绍文档：https://zhuanlan.zhihu.com/p/462980738
- 其他tensorRT课程（腾讯课堂）：https://ke.qq.com/course/4993141
