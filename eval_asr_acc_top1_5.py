"""投毒模型的训练与测试"""
import os

import param
import utils_.data_loader as dl
from param import Model
from utils_.TAT import *


def eval_model_asr_acc_top1_5():
    backdoor_model = Model(param.classes_num, False)

    if os.path.exists(param.weights_file_name):
        backdoor_model.load_state_dict(torch.load(param.weights_file_name, map_location=torch.device('cpu')))
    else:
        raise RuntimeError("should trojan before repairing !!!")

    test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

    asr_loader = dl.get_backdoor_loader(test_set, test_tf, True, param.test_batch_size, 1.0, shuffle=False)
    acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
    print(len(asr_loader) == len(acc_loader))

    # todo 测试asr中的acc(TOP1、TOP5)
    backdoor_model.eval()
    backdoor_model.to(param.device)

    total_acc_top0 = 0.
    total_acc_top1 = 0.
    total_acc_top2 = 0.
    total_asr = 0
    batch_num = len(asr_loader)
    topk = (1, 3, 5)

    for asr_batch, acc_batch in zip(asr_loader, acc_loader):
        asr_batch_img, asr_batch_label = asr_batch
        acc_batch_img, acc_batch_label = acc_batch

        asr_batch_img = asr_batch_img.to(param.device)

        asr_batch_label = asr_batch_label.to(param.device)
        acc_batch_label = acc_batch_label.to(param.device)

        with torch.no_grad():
            outputs = backdoor_model(asr_batch_img)
        predict = outputs.max(dim=-1)[-1]
        asr = predict.eq(asr_batch_label).float().mean()

        maxk = max(topk)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(acc_batch_label.view(1, -1).expand_as(pred))
        batch_size = asr_batch_img.size(0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        total_asr += asr
        total_acc_top0 += res[0].item()
        total_acc_top1 += res[1].item()
        total_acc_top2 += res[2].item()

    print(f"--------->Test  "
          f"ASR:{total_asr / batch_num:.4f}  "
          f"ACC-TOP{topk[0]}:{total_acc_top0 / batch_num:.4f}  "
          f"ACC-TOP{topk[1]}:{total_acc_top1 / batch_num:.4f}  "
          f"ACC-TOP{topk[2]}:{total_acc_top2 / batch_num:.4f}")


# 测试
if __name__ == '__main__':
    eval_model_asr_acc_top1_5()
