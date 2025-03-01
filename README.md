三种场景：
1. user-N: cifar10, user-F: cifar10
2. user-N: MNIST, user-F: cifar10
3. user-N: Europarl, user-F: cifar10
主要代码分别在cifar_cifar, mnist_cifar和cifar_europarl文件夹里面

以cifar_cifar为例：
[label](cifar_cifar/cifar_cifar_noma_main.py)是awgn信道下的训练代码，训练好的模型保存在models对应的文件夹下面
[label](cifar_cifar/cifar_cifar_awgn_test.py)是awgn信道下的测试代码
[label](cifar_cifar/cifar_cifar_test_awgn_quanbitmod.py)是hybrid方法(deepJSCC+传统调制+传统SIC)
[label](cifar_cifar/cifar_cifar_test_awgn_jpegldpc.py)是jpeg+LDPC+QAM+SIC方法，需要配合matlab.engine库使用

调制解调器模型训练：
[label](2usr_modem_model_train.py)
训练好的模型文件保存在models\pretrained_mod下面