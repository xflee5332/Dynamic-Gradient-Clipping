import os
import shutil
import warnings

from torch.nn import CrossEntropyLoss, BatchNorm2d
from torch.optim import Adam

import param
import utils_
from utils_ import data_loader as dl
from utils_.TAT import *

warnings.filterwarnings("ignore")
# 后门模型
target_model = None

train_set, train_tf = None, None
test_set, test_tf = None, None

# 测试集
acc_loader = None
asr_loader1 = None

# 修复集及其状态
repair_loader = None


def init():
    global target_model, train_set, train_tf, test_set, test_tf, acc_loader, asr_loader1, repair_loader
    # 后门模型
    target_model = param.Model(param.classes_num, True)

    criterion = CrossEntropyLoss()
    optimizer = Adam(params=target_model.parameters(), lr=param.learning_rate)

    train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
    backdoor_train_loader = dl.get_backdoor_loader(train_set, train_tf, True, param.train_batch_size, param.inject_rate)

    epoch = 1
    while epoch <= param.train_epoch:
        train_model(backdoor_train_loader, target_model, optimizer, criterion, epoch, param.device, param.train_print_freq)
        epoch += 1
    # 保存模型参数
    torch.save(target_model.state_dict(), param.weights_file_name)

    test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

    # 测试集
    acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
    asr_loader = dl.get_backdoor_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

    _, acc = test_model(acc_loader, target_model, criterion, param.device, param.test_print_freq)
    _, asr = test_model(asr_loader, target_model, criterion, param.device, param.test_print_freq)
    print(f"ACC:{acc:.4f} \t ASR:{asr:.4f}")

    # 修复集及其状态
    repair_set = dl.get_repair_dataset(train_set, param.per_class_benign_samples_usage, param.per_backdoor_samples_usage, test_tf, sample_fusing=True)
    repair_set_state = []
    print(f"repair set size:{len(repair_set)}")
    repair_loader = DataLoader(dataset=repair_set, batch_size=512, shuffle=False, drop_last=False)
    return acc, asr


def train_mode():
    for module in target_model.modules():
        if isinstance(module, BatchNorm2d):
            module.eval()
        else:
            module.train()


def train_():
    global target_model
    # 模型迁移至device
    target_model.to(param.device)
    optimizer = Adam(params=target_model.parameters(), lr=param.learning_rate)
    criterion = CrossEntropyLoss()
    # 状态分类
    # classify_state(True)
    # 训练模型
    acc = 0
    asr = 0

    # 训练模型
    for epoch in range(param.repair_epoch):
        # 使模型进入训练模式
        train_model(repair_loader, target_model, optimizer, criterion, epoch, param.device, param.train_print_freq)
        # --------
        # 测试模型
        # --------
        # 每训练完一轮，对模型进行一次测试
        _, acc = test_model(acc_loader, target_model, criterion, param.device, param.test_print_freq)
        # 每训练完一轮，对模型进行投毒数据测试
        _, asr = test_model(asr_loader1, target_model, criterion, param.device, param.test_print_freq)
        print(f"acc:{acc:.4f},asr:{asr:.4f}\n")
    return acc, asr


if __name__ == '__main__':
    with open("./log-repairSF.txt", 'a') as log_file:
        # for model_name in ["vgg13", "vgg19", "resnet18", "resnet34", "wideresnet50", "densnet121"]:
        #     for dataset_name in ['CIFAR10']:  # , 'MNIST', 'FASHION-MNIST', "GTSRB", "CLTL"
        for dataset_name, model_name in [('MNIST', "vgg13")]:  # ('CIFAR10', "resnet18"), ("GTSRB", "resnet18"), ('CLTL', "vgg13"),
            for trigger_type in ['trojanTrigger', 'gridTrigger', 'blendTrigger', "warpingTrigger"]:
                for target_type in ['all2one']:  # 'all2all'
                    param.model_name = model_name
                    param.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
                    param.dataset_name = dataset_name
                    param.trigger_type = trigger_type
                    param.target_type = target_type

                    param.classes_num = 10

                    if dataset_name == "CLTL":
                        param.classes_num = 7
                        param.per_class_benign_samples_usage = 400
                    elif dataset_name == "GTSRB":
                        param.classes_num = 43
                        param.per_class_benign_samples_usage = 110
                    elif dataset_name == "CIFAR100":
                        param.classes_num = 100
                        param.train_epoch = 15
                        param.per_class_benign_samples_usage = 50

                    param.samples_root_file = "/home/xfLee/ModelRepair/samples_" + f"/{trigger_type}/{target_type}/{dataset_name}"
                    param.weights_file_name = "./weights_" + f"/{model_name}-{dataset_name}-{target_type}-{trigger_type}.pth"

                    print("-" * 120)
                    print(f"target-model: {model_name} \t dataset: {dataset_name} \t target type: {target_type} \t trigger type: {trigger_type} \t trigger label: {param.target_label}")
                    print("-" * 120)

                    log_file.write("-" * 120 + "\n")
                    log_file.write(f"target-model: {model_name} \t dataset: {dataset_name} \t target type: {target_type} \t trigger type: {trigger_type}" + "\n")
                    log_file.write("-" * 120 + "\n")
                    log_file.flush()

                    if model_name == "alexnet":
                        param.Model = utils_.models_.AlexNet
                    elif model_name == "vgg13":
                        param.Model = utils_.models_.VGG13
                    elif model_name == "vgg19":
                        param.Model = utils_.models_.VGG19
                    elif model_name == "googlenet":
                        param.Model = utils_.models_.GoogleNet
                    elif model_name == "densnet121":
                        param.Model = utils_.models_.DenseNet121
                    elif model_name == "densnet169":
                        param.Model = utils_.models_.DenseNet169
                    elif model_name == "resnet18":
                        param.Model = utils_.models_.ResNet18
                    elif model_name == "resnet101":
                        param.Model = utils_.models_.ResNet101
                    elif model_name == "wideresnet50":
                        param.Model = utils_.models_.WideResNet50
                    elif model_name == "wideresnet101":
                        param.Model = utils_.models_.WideResNet101

                    print("remove cache backdoor image:", end="")
                    if os.path.exists(param.samples_root_file):
                        shutil.rmtree(param.samples_root_file)
                    print("OK")

                    print("train backdoor model: ")
                    acc, asr = init()
                    log_file.write(f"before => acc:{acc:.4f}, asr:{asr:.4f}" + "\n")
                    log_file.flush()

                    print("repair backdoor model:")
                    acc, asr = train_()
                    log_file.write(f"DGC => acc:{acc:.4f}, asr:{asr:.4f}" + "\n")
                    log_file.flush()
                    print("\n")
