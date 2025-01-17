import os
import sys

import torch
from torch import optim, nn
import torch.distributed as dist

from python.common_net import register_layer, register_weight_layer, get_layer_weight, get_layer_input, \
    get_layer_weight_grad, get_layer_output, get_layer_output_grad, get_layer_input_grad
from python.enclave_interfaces import GlobalTensor
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.batch_norm_1d import SecretBatchNorm1dLayer
from python.layers.conv2d import SecretConv2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.linear_base import SecretLinearLayerBase
from python.layers.matmul import SecretMatmulLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.linear_shares import init_communicate, warming_up_cuda, SecretNeuralNetwork, SgdOptimizer
from python.logger_utils import Logger
from python.quantize_net import NetQ
from python.test_linear_shares import argparser_distributed, marshal_process, load_cifar10, seed_torch
from python.timer_utils import NamedTimerInstance, VerboseLevel, NamedTimer
from python.torch_utils import compare_expected_actual

device_cuda = torch.device("cuda:0")

def compare_layer_member(layer: SecretLinearLayerBase, layer_name: str,
                         extract_func , member_name: str, save_path=None) -> None:
    print(member_name)
    layer.make_sure_cpu_is_latest(member_name)
    compare_expected_actual(extract_func(layer_name), layer.get_cpu(member_name), get_relative=True, verbose=True)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Directory ", save_path, " Created ")
        else:
            print("Directory ", save_path, " already exists")

        torch.save(extract_func(layer_name), os.path.join(save_path, member_name + "_expected"))
        torch.save(layer.get_cpu(member_name), os.path.join(save_path, member_name + "_actual"))


def compare_layer(layer: SecretLinearLayerBase, layer_name: str, save_path=None) -> None:
    print("comparing with layer in expected NN :", layer_name)
    compare_name_function = [("input", get_layer_input), ("output", get_layer_output),
                             ("DerOutput", get_layer_output_grad), ]
    if layer_name != "conv1":
        compare_name_function.append(("DerInput", get_layer_input_grad))
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)

def compare_weight_layer(layer: SecretLinearLayerBase, layer_name: str, save_path=None) -> None:
    compare_layer(layer, layer_name, save_path)
    compare_name_function = [("weight", get_layer_weight), ("DerWeight", get_layer_weight_grad) ]
    for member_name, extract_func in compare_name_function:
        compare_layer_member(layer, layer_name, extract_func, member_name, save_path=save_path)


def local_vgg9(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    n_img_channel = 3
    img_hw = 32
    n_classes = 10

    n_unit_fc1 = 512
    n_unit_fc2 = 512

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)

    def generate_conv_module(index, n_channel_conv, is_big=True):
        res = []
        if is_big:
            conv_local_1 = SecretConv2dLayer(sid, f"Conv{index}A", n_channel_conv, 3)
            norm_local_1 = SecretBatchNorm2dLayer(sid, f"Norm1{index}A")
            relu_local_1 = SecretReLULayer(sid, f"Relu{index}A")
            res += [conv_local_1, norm_local_1, relu_local_1]
        conv_local_2 = SecretConv2dLayer(sid, f"Conv{index}B", n_channel_conv, 3)
        norm_local_2 = SecretBatchNorm2dLayer(sid, f"Norm{index}B")
        relu_local_2 = SecretReLULayer(sid, f"Relu{index}B")
        pool_local_2 = SecretMaxpool2dLayer(sid, f"Pool{index}B", 2)
        res += [conv_local_2, norm_local_2, relu_local_2, pool_local_2]
        return res


    conv_module_1 = generate_conv_module(1, 64, is_big=False)
    conv_module_2 = generate_conv_module(2, 128, is_big=False)
    conv_module_3 = generate_conv_module(3, 256, is_big=True)
    conv_module_4 = generate_conv_module(4, 512, is_big=True)
    conv_module_5 = generate_conv_module(5, 512, is_big=True)
    all_conv_module = conv_module_1 + conv_module_2 + conv_module_3 + conv_module_4 + conv_module_5
    # all_conv_module = conv_module_1 + conv_module_2

    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, n_unit_fc1)
    # fc_norm1 = SecretBatchNorm1dLayer(sid, "FcNorm1")
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, n_unit_fc2)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, n_classes)
    output_layer = SecretOutputLayer(sid, "OutputLayer")

    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer] + all_conv_module + [flatten, fc1, fc_relu1, fc2, fc_relu2, fc3, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    secret_optim = SgdOptimizer(sid)
    secret_optim.set_eid(GlobalTensor.get_eid())
    secret_optim.set_layers(layers)

    input_layer.StoreInEnclave = False

    if is_compare:
        net = NetQ()
        net.to(device_cuda)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    conv1, norm1, relu1, pool1 = conv_module_1[0:4]
    conv2, norm2, relu2, pool2 = conv_module_2[0:4]
    conv3, norm3 = conv_module_3[0:2]
    conv4, norm4 = conv_module_3[3:5]
    conv5, norm5 = conv_module_4[0:2]
    conv6, norm6 = conv_module_4[3:5]
    conv7, norm7 = conv_module_5[0:2]
    conv8, norm8, relu8, pool8 = conv_module_5[3:7]


    validation_net = NetQ()
    validation_net.to(torch.device("cuda:0"))

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 200
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break


            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                def compute_expected_nn():
                    optimizer.zero_grad()
                    outputs = net(input_f.to(device_cuda))
                    loss = criterion(outputs, target_f.to(device_cuda))
                    loss.backward()
                    optimizer.step()
                    print("netQ loss:", loss)

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()

                with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()
        NamedTimer.end("TrainValidationEpoch")

def local_vgg16(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    n_img_channel = 3
    img_hw = 32
    n_classes = 10

    n_unit_fc1 = 512
    n_unit_fc2 = 512

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)

    def generate_conv_module(index, n_channel_conv, num_small=0):
        res = []
        for i in range(num_small):  # VGG-16 has 2 or 3 conv layers before a pool in the bigger blocks
            conv_local = SecretConv2dLayer(sid, f"Conv{index}_{i}", n_channel_conv, 3)
            relu_local = SecretReLULayer(sid, f"Relu{index}_{i}")
            res += [conv_local, relu_local]
                
        conv_local = SecretConv2dLayer(sid, f"Conv{index}", n_channel_conv, 3)
        relu_local = SecretReLULayer(sid, f"Relu{index}")
        pool_local = SecretMaxpool2dLayer(sid, f"Pool{index}", 2)
        res += [conv_local, relu_local, pool_local]
        index += 1
        return res, index

    index = 1
    conv_module_1, index = generate_conv_module(index, 64, 1)
    conv_module_2, index = generate_conv_module(index, 128, 1)
    conv_module_3, index = generate_conv_module(index, 256, 2)
    conv_module_4, index = generate_conv_module(index, 512, 2)
    conv_module_5, index = generate_conv_module(index, 512, 2)

    all_conv_modules = conv_module_1 + conv_module_2 + conv_module_3 + conv_module_4 + conv_module_5

    # Fully connected layers adjusted for CIFAR-10 classification
    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, 4096)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, 4096)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, n_classes)  # Output layer for CIFAR-10

    output_layer = SecretOutputLayer(sid, "OutputLayer")


    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer] + all_conv_modules + [flatten, fc1, fc_relu1, fc2, fc_relu2, fc3, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()
                break
        with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()

        NamedTimer.end("TrainValidationEpoch")

def local_alexnet(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    n_img_channel = 3
    img_hw = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=32)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)
    all_conv_modules = []
    
    
    
    # index = 1
    # conv_local = SecretConv2dLayer(sid, f"Conv{index}", 96, 11, stride=4, padding = 10)
    # norm_local = SecretBatchNorm2dLayer(sid, f"Norm{index}")
    # relu_local = SecretReLULayer(sid, f"Relu{index}")
    # pool_local = SecretMaxpool2dLayer(sid, f"Pool{index}", 3, maxpoolpadding = 1, row_stride = 2, col_stride = 2)
    # all_conv_modules += [conv_local, norm_local, relu_local, pool_local]
    
    # index = 2
    # conv_local = SecretConv2dLayer(sid, f"Conv{index}", 256, 5)
    # norm_local = SecretBatchNorm2dLayer(sid, f"Norm{index}")
    # relu_local = SecretReLULayer(sid, f"Relu{index}")
    # pool_local = SecretMaxpool2dLayer(sid, f"Pool{index}", 3, maxpoolpadding = 1, row_stride = 2, col_stride = 2)
    # all_conv_modules += [conv_local, norm_local, relu_local, pool_local]
    
    
    
    # Adjusting the convolutional layers for AlexNet
    # TODO: Solve this.
    conv1 = SecretConv2dLayer(sid, "Conv1", 128, 11, stride=4, padding = 10)
    norm1 = SecretBatchNorm2dLayer(sid, "Norm1")
    relu1 = SecretReLULayer(sid, "Relu1")
    pool1 = SecretMaxpool2dLayer(sid, "Pool1", 3, maxpoolpadding = 1, row_stride = 2, col_stride = 2)

    conv2 = SecretConv2dLayer(sid, "Conv2", 256, 5)
    norm2 = SecretBatchNorm2dLayer(sid, "Norm2")
    relu2 = SecretReLULayer(sid, "Relu2")
    pool2 = SecretMaxpool2dLayer(sid, "Pool2", 3, maxpoolpadding = 1, row_stride = 2, col_stride = 2)

    conv3 = SecretConv2dLayer(sid, "Conv3", 384, 3)
    relu3 = SecretReLULayer(sid, "Relu3")

    conv4 = SecretConv2dLayer(sid, "Conv4", 384, 3)
    relu4 = SecretReLULayer(sid, "Relu4")

    conv5 = SecretConv2dLayer(sid, "Conv5", 256, 3)
    relu5 = SecretReLULayer(sid, "Relu5")

    flatten = SecretFlattenLayer(sid, "FlattenLayer")

    # Adjusting the fully connected layers for AlexNet
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, 256)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")


    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, 256)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")

    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, n_classes)
    fc_relu3 = SecretReLULayer(sid, "FcRelu3")

    output_layer = SecretOutputLayer(sid, "OutputLayer")

    # Assembling the layers for AlexNet
    layers = [input_layer, conv1, norm1, relu1, pool1, conv2, norm2, relu2, pool2, conv3, relu3, conv4, 
              relu4, conv5, relu5, flatten, fc1, fc_relu1, fc2, fc_relu2, fc3, fc_relu3, output_layer]
    # layers = [input_layer, conv1, norm1, flatten, fc3, fc_relu3, output_layer]
    # layers = [input_layer] + all_conv_modules + [flatten, fc3,fc_relu3, output_layer]

    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                print("Forward")
                secret_nn.forward()
            break

        # NamedTimer.end("TrainValidationEpoch")

def local_secureml(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    diminput = 784
    x_shape = [batch_size, diminput]
    n_img_channel = 3
    img_hw = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)


    # Fully connected layers adjusted for CIFAR-10 classification
    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc1 = SecretMatmulLayer(sid, "FC1", batch_size, 128)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, 128)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, 10)  # Output layer for CIFAR-10
    fc_relu3 = SecretReLULayer(sid, "FcRelu3")
    output_layer = SecretOutputLayer(sid, "OutputLayer")


    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer, flatten, fc1, fc_relu1, fc2, fc_relu2, fc3, fc_relu3, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                # input("Press Enter to continue...")
                secret_nn.forward()
                break
        print("dist.GLOBAL_SEND, dist.GLOBAL_RECV")
        print(dist.GLOBAL_SEND, dist.GLOBAL_RECV)
        with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()

        NamedTimer.end("TrainValidationEpoch")

def local_sarda(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    diminput = 784
    x_shape = [batch_size, diminput]
    n_img_channel = 1
    img_hw = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)


    # Fully connected layers adjusted for CIFAR-10 classification
    
    conv_local = SecretConv2dLayer(sid, f"Conv1", 5, 2,padding=2)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc2 = SecretMatmulLayer(sid, "FC2", batch_size, 100)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, 10)  # Output layer for CIFAR-10
    fc_relu3 = SecretReLULayer(sid, "FcRelu3")
    output_layer = SecretOutputLayer(sid, "OutputLayer")


    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer, conv_local, fc_relu1,flatten, fc2, fc_relu2, fc3, fc_relu3, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()
                break
        with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()

        NamedTimer.end("TrainValidationEpoch")
        
def local_minionn(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    diminput = 784
    x_shape = [batch_size, diminput]
    n_img_channel = 1
    img_hw = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)


    # Fully connected layers adjusted for CIFAR-10 classification
    
    conv_local1 = SecretConv2dLayer(sid, f"Conv1", 16, 5,padding=4)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    pool1 = SecretMaxpool2dLayer(sid, "Pool1", 2, maxpoolpadding = 0, row_stride = 2, col_stride = 2)
    conv_local2 = SecretConv2dLayer(sid, f"Conv2", 16, 5,padding=4)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    pool2 = SecretMaxpool2dLayer(sid, "Pool2", 2, maxpoolpadding = 0, row_stride = 2, col_stride = 2)
    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, 100)
    fc_relu3 = SecretReLULayer(sid, "FcRelu3")
    fc4 = SecretMatmulLayer(sid, "FC4", batch_size, 10)  # Output layer for CIFAR-10
    fc_relu4 = SecretReLULayer(sid, "FcRelu4")
    output_layer = SecretOutputLayer(sid, "OutputLayer")


    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer, conv_local1, fc_relu1,pool1, conv_local2, fc_relu2,pool2, flatten, fc3, fc_relu3, fc4, fc_relu4, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()
                break
        with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()

        NamedTimer.end("TrainValidationEpoch")
        
def local_lenet(sid, master_addr, master_port, is_compare=False):
    init_communicate(sid, master_addr, master_port)
    warming_up_cuda()

    batch_size = 128
    diminput = 784
    x_shape = [batch_size, diminput]
    n_img_channel = 1
    img_hw = 32
    n_classes = 10

    x_shape = [batch_size, n_img_channel, img_hw, img_hw]

    

    trainloader, testloader = load_cifar10(batch_size, test_batch_size=128)

    GlobalTensor.init()

    input_layer = SecretInputLayer(sid, "InputLayer", x_shape)


    # Fully connected layers adjusted for CIFAR-10 classification
    
    conv_local1 = SecretConv2dLayer(sid, f"Conv1", 20, 5,padding=4)
    fc_relu1 = SecretReLULayer(sid, "FcRelu1")
    pool1 = SecretMaxpool2dLayer(sid, "Pool1", 2, maxpoolpadding = 0, row_stride = 2, col_stride = 2)
    conv_local2 = SecretConv2dLayer(sid, f"Conv2", 50, 5,padding=4)
    fc_relu2 = SecretReLULayer(sid, "FcRelu2")
    pool2 = SecretMaxpool2dLayer(sid, "Pool2", 2, maxpoolpadding = 0, row_stride = 2, col_stride = 2)
    flatten = SecretFlattenLayer(sid, "FlattenLayer")
    fc3 = SecretMatmulLayer(sid, "FC3", batch_size, 500)
    fc_relu3 = SecretReLULayer(sid, "FcRelu3")
    fc4 = SecretMatmulLayer(sid, "FC4", batch_size, 10)  # Output layer for CIFAR-10
    fc_relu4 = SecretReLULayer(sid, "FcRelu4")
    output_layer = SecretOutputLayer(sid, "OutputLayer")


    # layers = [input_layer] + all_conv_module + [flatten, fc1, fc_norm1, fc_relu1, fc2, output_layer]
    layers = [input_layer, conv_local1, fc_relu1,pool1, conv_local2, fc_relu2,pool2, flatten, fc3, fc_relu3, fc4, fc_relu4, output_layer]
    secret_nn = SecretNeuralNetwork(sid, "SecretNeuralNetwork")
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(layers)

    input_layer.StoreInEnclave = False

    NamedTimer.set_verbose_level(VerboseLevel.RUN)

    train_counter = 0
    running_loss = 0

    NumEpoch = 1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    for epoch in range(NumEpoch):  # loop over the dataset multiple times
        # NamedTimer.start("TrainValidationEpoch", verbose_level=VerboseLevel.RUN)
        for input_f, target_f in trainloader:
            run_batch_size = input_f.size()[0]
            if run_batch_size != batch_size:
                break

            train_counter += 1
            with NamedTimerInstance("TrainWithBatch", VerboseLevel.RUN):

                if sid != 2:
                    input_layer.set_input(input_f)
                    output_layer.load_target(target_f)

                dist.barrier()
                secret_nn.forward()
                break
        with NamedTimerInstance(f"Sid: {sid} Free cuda cache"):
                    torch.cuda.empty_cache()

        # NamedTimer.end("TrainValidationEpoch")


from scapy.all import sniff

def packet_analysis(packet):
    # You can add filtering logic here to process only certain packets
    print(packet.summary())


if __name__ == "__main__":
    input_sid, MasterAddr, MasterPort, test = argparser_distributed()

    sys.stdout = Logger()
    print("====== New Tests ======")
    print("input_sid, MasterAddr, MasterPort", input_sid, MasterAddr, MasterPort)

    seed_torch(123)
    
    marshal_process(input_sid, MasterAddr, MasterPort, local_vgg16, [])
    
    # marshal_process(input_sid, MasterAddr, MasterPort, local_sarda, [])
    # marshal_process(input_sid, MasterAddr, MasterPort, local_minionn, [])
    # marshal_process(input_sid, MasterAddr, MasterPort, local_lenet, [])
    # marshal_process(input_sid, MasterAddr, MasterPort, local_alexnet, [])
    # marshal_process(input_sid, MasterAddr, MasterPort, local_vgg16, [])
    # input("Press Enter to continue...")

