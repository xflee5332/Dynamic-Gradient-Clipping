"""投毒模型的训练与测试"""
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import param
import utils_.data_loader as dl
from param import Model
from utils_.TAT import *


def trainBackdoorModel(with_eval=True):
    backdoor_model = Model(param.classes_num, True)
    optimizer = Adam(params=backdoor_model.parameters(), lr=param.learning_rate)
    criterion = CrossEntropyLoss()

    test_set, test_tf = dl.get_dataset_and_transformers(param.dataset_name, False)

    if with_eval:
        asr_loader = dl.get_backdoor_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
        acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)

    train_set, train_tf = dl.get_dataset_and_transformers(param.dataset_name, True)
    backdoor_train_loader = dl.get_backdoor_loader(train_set, train_tf, True, param.train_batch_size, param.inject_rate)

    # if os.path.exists(param.weights_file_name):
    #     backdoor_model.load_state_dict(torch.load(param.weights_file_name, map_location=torch.device('cpu')))
    #     asr_loader = dl.get_backdoor_loader(test_set, test_tf, True, param.test_batch_size, 1.0)
    #     acc_loader = dl.get_initial_loader(test_set, test_tf, False, param.test_batch_size, 1.0)
    #     _, acc = test_model(acc_loader, backdoor_model, criterion, param.device, param.test_print_freq)
    #     _, asr = test_model(asr_loader, backdoor_model, criterion, param.device, param.test_print_freq)
    #     print(f"ACC:{acc:.4f} \t ASR:{asr:.4f}")
    #     return

    epoch = 1
    while epoch <= param.train_epoch:
        train_model(backdoor_train_loader, backdoor_model, optimizer, criterion, epoch, param.device, param.train_print_freq)
        if with_eval:
            _, acc = test_model(acc_loader, backdoor_model, criterion, param.device, param.test_print_freq)
            _, asr = test_model(asr_loader, backdoor_model, criterion, param.device, param.test_print_freq)
            print(f"ACC:{acc:.4f} \t ASR:{asr:.4f}")
        epoch += 1
    # 保存模型参数
    torch.save(backdoor_model.state_dict(), param.weights_file_name)
    return backdoor_model


# 测试
if __name__ == '__main__':
    trainBackdoorModel()
