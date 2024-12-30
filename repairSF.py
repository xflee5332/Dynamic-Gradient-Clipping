import os
import shutil

from torch.nn import CrossEntropyLoss, BatchNorm2d, Dropout
from torch.optim import Adam

import param
from attack import trainBackdoorModel
from utils_ import data_loader as dl
from utils_.TAT import *
from utils_.TAT import train_model

print("remove cache backdoor image:")
if os.path.exists(param.samples_root_file):
    shutil.rmtree("./samples_")

print("train backdoor model:")
# 后门模型
target_model = trainBackdoorModel()

print("repair backdoor model:")
criterion = CrossEntropyLoss()
optimizer = Adam(params=target_model.parameters(), lr=param.learning_rate)

train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

# 测试集
acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
asr_loader1 = dl.get_backdoor_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

# 修复集及其状态
repair_set = dl.get_repair_dataset(train_set, param.per_class_benign_samples_usage, param.per_backdoor_samples_usage, test_tf, sample_fusing=True)

print(f"repair set size:{len(repair_set)}")
repair_loader = DataLoader(dataset=repair_set, batch_size=512, shuffle=False, drop_last=False)


def train_mode():
    for module in target_model.modules():
        if isinstance(module, BatchNorm2d) or isinstance(module, Dropout):
            module.eval()
        else:
            module.train()


def train_():
    global repair_loader
    # 模型迁移至device
    target_model.to(param.device)

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


if __name__ == '__main__':
    train_()
