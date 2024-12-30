import os
import shutil
import warnings

import numpy as np
from torch.nn import CrossEntropyLoss, BatchNorm2d
from torch.optim import Adam

import param
import utils_
from clipping import NextClipping
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
asr_loader2 = None

# 修复集及其状态
repair_set = None
repair_set_state = []
repair_loader = None
state0, state1, state2 = 0, 0, 0
alpha = 0.3
beta = 0.7


# =======================================
# 训练后门模型：Blend 和 Trojan 攻击方式
# =======================================

def init():
    global target_model, train_set, train_tf, test_set, test_tf, acc_loader, asr_loader1, asr_loader2, repair_set, repair_set_state, repair_loader
    # 后门模型
    target_model = param.Model(param.classes_num, True)

    criterion = CrossEntropyLoss()
    optimizer = Adam(params=target_model.parameters(), lr=param.learning_rate)

    train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
    test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

    acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

    # =================================
    param.trigger_type = "warpingTrigger"
    backdoor_train_loader = dl.get_backdoor_loader(train_set, train_tf, True, param.train_batch_size, param.inject_rate)

    epoch = 1
    while epoch <= param.train_epoch:
        train_model(backdoor_train_loader, target_model, optimizer, criterion, epoch, param.device, param.train_print_freq)
        epoch += 1

    # 测试集
    asr_loader2 = dl.get_backdoor_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

    _, acc = test_model(acc_loader, target_model, criterion, param.device, param.test_print_freq)
    _, asr = test_model(asr_loader2, target_model, criterion, param.device, param.test_print_freq)
    print(f"ACC:{acc:.4f} \t ASR:{asr:.4f}")

    # =================================
    param.trigger_type = "gridTrigger"
    backdoor_train_loader = dl.get_backdoor_loader(train_set, train_tf, True, param.train_batch_size, param.inject_rate)

    epoch = 1
    while epoch <= param.train_epoch:
        train_model(backdoor_train_loader, target_model, optimizer, criterion, epoch, param.device, param.train_print_freq)
        epoch += 1

    # 测试集
    asr_loader1 = dl.get_backdoor_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

    _, acc = test_model(acc_loader, target_model, criterion, param.device, param.test_print_freq)
    _, asr = test_model(asr_loader1, target_model, criterion, param.device, param.test_print_freq)
    print(f"ACC:{acc:.4f} \t ASR:{asr:.4f}")

    # =================================
    # 修复集及其状态
    repair_set = dl.get_repair_dataset(train_set, param.per_class_benign_samples_usage, param.per_backdoor_samples_usage, test_tf)
    repair_set_state = []
    print(f"repair set size:{len(repair_set)}")
    repair_loader = DataLoader(dataset=repair_set, batch_size=512, shuffle=False, drop_last=False)
    return acc, asr


def classify_state(reset: bool):
    """对修复集中的样本进行状态分类"""
    global state0, state1, state2
    target_model.eval()
    target_model.to(param.device)
    # 不对计算追踪，不计算梯度，不更新参数
    with torch.no_grad():
        predict = []
        for idx, (img, target) in enumerate(repair_loader):
            img = img.to(param.device)
            results = target_model(img)
            predict_ = torch.sigmoid(results)
            predict.extend(predict_.view(-1, param.classes_num).cpu().numpy())

        shut_predict = np.empty((len(predict), len(predict[0])), dtype=float)
        for i in range(len(predict)):
            for j in range(len(predict[i])):
                if alpha < predict[i][j] < beta:
                    shut_predict[i][j] = alpha
                else:
                    shut_predict[i][j] = predict[i][j]

        one_hot_label_arr = []
        one_hot_target_arr = []
        for _, label in repair_set:
            one_hot_label = [0] * param.classes_num
            one_hot_label[label] = 1

            one_hot_target = [0] * param.classes_num
            if param.target_type == 'all2one':
                one_hot_target[param.target_label] = 1
            elif param.target_type == 'all2all':
                one_hot_target[(label + 1) % param.classes_num] = 1
            one_hot_label_arr.append(one_hot_label)
            one_hot_target_arr.append(one_hot_target)
        shut_predict_t = torch.tensor(shut_predict) - torch.tensor(one_hot_label_arr)
        shut_predict_b = torch.tensor(shut_predict) - torch.tensor(one_hot_target_arr)
        shut_predict_amb = torch.tensor(shut_predict) - (torch.tensor(one_hot_label_arr) + torch.tensor(one_hot_target_arr))

        judge_loss_t = torch.norm(shut_predict_t, p=2, dim=1)
        judge_loss_b = torch.norm(shut_predict_b, p=2, dim=1)
        judge_loss_amb = torch.norm(shut_predict_amb, p=2, dim=1)

        # -------
        # 状态分类
        # -------
        state0, state1, state2 = 0, 0, 0
        if reset:
            repair_set_state.clear()
        for i in range(len(judge_loss_amb)):
            if judge_loss_t[i] <= judge_loss_b[i] and judge_loss_t[i] <= judge_loss_amb[i]:
                if reset:
                    repair_set_state.append((0, i))
                state0 += 1
            elif judge_loss_b[i] <= judge_loss_t[i] and judge_loss_b[i] <= judge_loss_amb[i]:
                if reset:
                    repair_set_state.append((2, i))
                state2 += 1
            else:
                if reset:
                    repair_set_state.append((1, i))
                state1 += 1
        print(f"states samples state0:{state0}, state1:{state1}, state2:{state2}")


def train_mode():
    for module in target_model.modules():
        if isinstance(module, BatchNorm2d):
            module.eval()
        else:
            module.train()


def repair_train():
    global repair_set, state0, state1, state2, target_model
    # 模型迁移至device
    target_model.to(param.device)
    optimizer = Adam(params=target_model.parameters(), lr=param.learning_rate)
    criterion = CrossEntropyLoss()
    # 状态分类
    # classify_state(True)
    # 训练模型
    # classify_state(True)
    # rmc = NextClipping(state0, state1, state2)
    for epoch in range(param.repair_epoch):
        # 状态分类
        classify_state(True)
        rmc = NextClipping()
        penalty_parameter = torch.tensor(rmc.get_clip_parameter())
        print(penalty_parameter)
        # 使模型进入训练模式
        train_mode()
        # target_model.train()
        # 打乱样本顺序
        np.random.shuffle(repair_set_state)
        # 计算训练一轮所需要的步数
        total_step = len(repair_set) // param.train_batch_size

        for step in range(total_step):
            loss_avg = 0
            start = step * param.train_batch_size
            end = min(len(repair_set), (step + 1) * param.train_batch_size)
            # 梯度惩罚需要使用的数据
            penalty_img = [
                [],
                [],
                []
            ]
            penalty_img_label = [
                [],
                [],
                []
            ]
            for i in range(start, end):
                state, idx = repair_set_state[i]
                penalty_img[state].append(repair_set[idx][0])
                penalty_img_label[state].append(repair_set[idx][1])

            # 训练模型（带有梯度惩罚）
            for i in range(len(penalty_img)):
                if len(penalty_img[i]) != 0:
                    # 将list图片转换为pytorch Tensor的数据类型
                    penalty_img_ = torch.stack(penalty_img[i]).to(param.device)
                    penalty_img_.requires_grad = True
                    penalty_prediction = target_model(penalty_img_)

                    loss = criterion(penalty_prediction, torch.tensor(penalty_img_label[i]).to(param.device))
                    loss_avg += loss.cpu().item()
                    # if i == 2:
                    #     loss -= criterion(penalty_prediction, torch.tensor([param.target_label] * len(penalty_img_label[i])).to(param.device))

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(target_model.parameters(), penalty_parameter[i])
                    optimizer.step()
            if step % 5 == 4:
                print(
                    f"{len(penalty_img[0]):03d},{len(penalty_img[1]):03d},{len(penalty_img[2]):03d}    "
                    f"epoch: {epoch + 1:03d}    "
                    f"step: [{step + 1:03d} / {total_step:03d}]   "
                    f"loss: {loss_avg / step :.4f}    "
                )
        # --------
        # 测试模型
        # --------
        # 每训练完一轮，对模型进行一次测试
        _, acc = test_model(acc_loader, target_model, criterion, param.device, param.test_print_freq)
        # 每训练完一轮，对模型进行投毒数据测试
        _, asr1 = test_model(asr_loader1, target_model, criterion, param.device, param.test_print_freq)
        # 每训练完一轮，对模型进行投毒数据测试
        _, asr2 = test_model(asr_loader2, target_model, criterion, param.device, param.test_print_freq)
        print(f"acc:{acc:.4f}, asr1:{asr1:.4f}, asr2:{asr2:.4f}")


if __name__ == '__main__':
    print("train backdoor model: ")
    init()

    print("repair backdoor model:")
    repair_train()
