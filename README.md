Pytorch implementation for paper "Non-Orthogonal Multiple Access Enhanced  Multi-User Semantic Communication"
three different scenarios：
1. user-N: cifar10, user-F: cifar10
2. user-N: MNIST, user-F: cifar10
3. user-N: Europarl, user-F: cifar10
where the corresponding codes are in the folder cifar_cifar/, mnist_cifar/ and cifar_europarl/
training and testing codes for Openimage and Cityscape dataset are also provided
we have provided pretrained image codec for Openimage dataset, please download them from https://drive.google.com/file/d/1JDW8bCOtgliRskgHkqDuHasARRx1kxJs/view?usp=share_link.

take the cifar_cifar scenario as an example：
(cifar_cifar/cifar_cifar_noma_main.py) is the training code under awgn channel
(cifar_cifar/cifar_cifar_awgn_test.py) is the testing code under awgn channel
(cifar_cifar/cifar_cifar_test_awgn_quanbitmod.py) is the code for hybrid method (DeepJSCC + modulation + SIC)
(cifar_cifar/cifar_cifar_test_awgn_jpegldpc.py) is the code for jpeg+LDPC+QAM+SIC, which requires the matlab.engine 

training code for the modem model is in
(2usr_modem_model_train.py)
we have also provided pretrained modem model in models\pretrained_mod
