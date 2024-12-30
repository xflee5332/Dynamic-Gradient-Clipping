import torch
import sys

from utils_ import models_

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 模型名称
model_name = "resnet18"
assert model_name in [
    # "alexnet",
    "vgg13", "vgg19",
    # "googlenet",
    "densnet121", "densnet169",
    "resnet18", "resnet34", "resnet101",
    "wideresnet50", "wideresnet101"
]

# 使用数据集名称
dataset_name = "CIFAR10"
assert dataset_name in ['MNIST', 'FASHION-MNIST', 'CIFAR10', 'CIFAR100', "GTSRB", "CLTL"]

# 后门攻击trigger类型
trigger_type = "blendTrigger"
assert trigger_type in ['gridTrigger', 'blendTrigger', 'trojanTrigger', "warpingTrigger",
                        'squareTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger']

# 后门攻击label类型
target_type = "all2one"
assert target_type in ['all2one', 'all2all']
# acc:0.8336,asr:0.0000
per_class_benign_samples_usage = 500
per_backdoor_samples_usage = 5
# 训练后门模型时，后门攻击的label
target_label = 1  # under all-to-one attack

# 模型训练与测试
train_batch_size = 320
train_print_freq = 32
# train_print_freq = sys.maxsize

test_batch_size = 512
test_print_freq = sys.maxsize

learning_rate = 0.0001
# 后门训练集样本污染率(在训练后门模型时使用)
inject_rate = 0.1
# 训练轮次(如果模型在进行Fuzz测试之前未经过训练的话，会先训练待测试模型)
train_epoch = 7

# 分类样本的类别总数
classes_num = 10
if dataset_name == "CLTL":
    classes_num = 7
    per_class_benign_samples_usage = 400
elif dataset_name == "GTSRB":
    classes_num = 43
    per_class_benign_samples_usage = 110
elif dataset_name == "CIFAR100":
    classes_num = 100
    train_epoch = 15
    # per_class_benign_samples_usage = 50

repair_epoch = 10
samples_root_file = "/home/xfLee/ModelRepair/samples_" + f"/{trigger_type}/{target_type}/{dataset_name}"
weights_file_name = "weights_/" + f"{model_name}-{dataset_name}-{target_type}-{trigger_type}.pth"

# print("-" * 120)
# print(f"target-model: {model_name} \t dataset: {dataset_name} \t target type: {target_type} \t trigger type: {trigger_type}")
# print("-" * 120)

if model_name == "alexnet":
    Model = models_.AlexNet
elif model_name == "vgg13":
    Model = models_.VGG13
elif model_name == "vgg19":
    Model = models_.VGG19
elif model_name == "googlenet":
    Model = models_.GoogleNet
elif model_name == "densnet121":
    Model = models_.DenseNet121
elif model_name == "densnet169":
    Model = models_.DenseNet169
elif model_name == "resnet18":
    Model = models_.ResNet18
elif model_name == "resnet101":
    Model = models_.ResNet101
elif model_name == "wideresnet50":
    Model = models_.WideResNet50
elif model_name == "wideresnet101":
    Model = models_.WideResNet101
