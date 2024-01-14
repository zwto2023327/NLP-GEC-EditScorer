import os
import re
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm
import copy
import random


def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"


def get_batch_metrics(pred_labels, labels, mask=None, ignore_labels=None, metric_func=None, note_use=False,
                      threshold=0.5):
    answer = defaultdict(int)
    error_list = copy.deepcopy(pred_labels)
    for r, (curr_pred_labels, curr_labels) in enumerate(zip(pred_labels, labels)):
        if mask is not None:
            curr_labels = [x for x, flag in zip(curr_labels, mask[r]) if flag]
        elif ignore_labels is not None:
            curr_labels = [label for label in curr_labels if label not in ignore_labels]
        # assert len(curr_pred_labels) == len(curr_labels), f"{len(curr_pred_labels)}-{len(curr_labels)}"
        for key, value in metric_func(curr_labels, curr_pred_labels, threshold=threshold, note_use=note_use).items():
            if key != "error_list":
                answer[key] += value
            else:
                error_list[r] = value
    if note_use:
        return {"answer": answer, "error_list": error_list}
    return answer


# 返回结果中包含错误索引list——error_list:1表示正确，0表示错误
def update_metrics(metrics, batch_output, batch, mask=None,
                   answer_field="labels", y_field="y", extract_func=None,
                   metric_func=None, aggregate_func=None, note_use=False, threshold=0.5):
    n_batches = metrics["n_batches"]
    for key, value in batch_output.items():
        if "loss" in key:
            # todo n_batches同步修改
            metrics[key] = (metrics.get(key, 0.0) * n_batches + value.item()) / (n_batches + 1)
    metrics["n_batches"] += 1
    if extract_func is not None:
        y_pred, y_true = extract_func(batch_output, batch)
    else:
        y_pred, y_true = batch_output[answer_field], batch[y_field].cpu().tolist()
    batch_metrics = get_batch_metrics(y_pred, y_true, mask=mask, ignore_labels=None, metric_func=metric_func,
                                      note_use=note_use, threshold=threshold)
    if note_use:
        for key, value in batch_metrics["answer"].items():
            metrics[key] = metrics.get(key, 0) + value
    else:
        for key, value in batch_metrics.items():
            metrics[key] = metrics.get(key, 0) + value
    # print(metrics)
    aggregate_func(metrics)
    return batch_metrics


def replace_index_note(index_map, index_list, elem):
    if elem.shape[0] == 0 or elem.shape[1] != 2:
        return elem
    delete = []
    index = -1
    # 将不存在的索引删除
    for e in elem:
        index = index + 1
        if e[0] not in index_list or e[1] not in index_list:
            delete.append(index)
    elem = np.delete(elem, delete, axis=0)
    # 将现有索引替换
    for key in index_map:
        elem[elem == key] = index_map[key]
    return elem


# 根据标签采取不同的数据过滤处理机制和按照比例的随机机制
# todo 增加reverse机制
def get_batch_reverse_note(enable, correctlist, notelist, batch_note, note_now, model, correct_n=1, reverse_n=1, mode="train"):
    # index_list永远是一维
    index_list = []
    index_map = {}
    index_add = 0
    sum = 0
    default = []
    default_i = 0
    default_index = []
    reverse_index = []
    offset = [0]
    last_index_add = -1
    defaultcorrect = 0
    for index in range(len(notelist)):
        di = index - sum
        if di == batch_note["default"][default_i]:
            sum = sum + di + 1
            default_value = index_add - last_index_add - 1
            if default_value != 0:
                if index in correctlist:
                    defaultcorrect = defaultcorrect + 1
                default_index.append(default_i)
                index_map[index] = index_add
                default.append(default_value)
                last_index_add = index_add
                index_add = index_add + 1
                offset.append(index_add)
                index_list.append(index)
            default_i = default_i + 1
            continue
        if index in correctlist and mode == "train":
            continue
        index_map[index] = index_add
        index_add = index_add + 1
        index_list.append(index)
    batch_note['input_ids'] = torch.index_select(batch_note['input_ids'], 0, torch.tensor(index_list).to(model.device))
    batch_note['label'] = torch.index_select(batch_note['label'], 0, torch.tensor(index_list).to(model.device))
    '''for index in reverse_index:
        if batch_note['label'][index] == 0 and enable:
            batch_note['label'][index] = 1
        elif batch_note['label'][index] == 1 and enable:
            batch_note['label'][index] = 0'''
    batch_note['start'] = [batch_note['start'][i] for i in index_list]
    batch_note['end'] = [batch_note['end'][i] for i in index_list]
    batch_note['origin_start'] = [batch_note['origin_start'][i] for i in index_list]
    batch_note['origin_end'] = [batch_note['origin_end'][i] for i in index_list]
    batch_note['default'] = default
    batch_note['reverse_index'] = reverse_index
    batch_note['indexes'] = [batch_note['indexes'][i] for i in default_index]
    batch_note['hard_pairs'] = replace_index_note(index_map, index_list, batch_note['hard_pairs'])
    batch_note['soft_pairs'] = replace_index_note(index_map, index_list, batch_note['soft_pairs'])
    batch_note['no_change_pairs'] = replace_index_note(index_map, index_list, batch_note['no_change_pairs'])
    batch_note['offset'] = offset
    return {"batch_note": batch_note, "index_list": index_list,"defaultcorrect":defaultcorrect}


def get_batch_note(notelist, batch_note, note_now, model, correct_n=1, mode="train"):
    # index_list永远是一维
    index_list = []
    index_map = {}
    index_add = 0
    sum = 0
    default = []
    default_i = 0
    default_index = []
    offset = [0]
    last_index_add = -1
    for index in range(len(notelist)):
        di = index - sum
        if di == batch_note["default"][default_i]:
            sum = sum + di + 1
            default_value = index_add - last_index_add - 1
            if default_value != 0:
                default_index.append(default_i)
                index_map[index] = index_add
                default.append(default_value)
                last_index_add = index_add
                index_add = index_add + 1
                offset.append(index_add)
                index_list.append(index)
            default_i = default_i + 1
            continue
        index_map[index] = index_add
        index_add = index_add + 1
        index_list.append(index)
    batch_note['input_ids'] = torch.index_select(batch_note['input_ids'], 0, torch.tensor(index_list).to(model.device))
    batch_note['label'] = torch.index_select(batch_note['label'], 0, torch.tensor(index_list).to(model.device))
    batch_note['start'] = [batch_note['start'][i] for i in index_list]
    batch_note['end'] = [batch_note['end'][i] for i in index_list]
    batch_note['origin_start'] = [batch_note['origin_start'][i] for i in index_list]
    batch_note['origin_end'] = [batch_note['origin_end'][i] for i in index_list]
    batch_note['default'] = default
    batch_note['indexes'] = [batch_note['indexes'][i] for i in default_index]
    batch_note['hard_pairs'] = replace_index_note(index_map, index_list, batch_note['hard_pairs'])
    batch_note['soft_pairs'] = replace_index_note(index_map, index_list, batch_note['soft_pairs'])
    batch_note['no_change_pairs'] = replace_index_note(index_map, index_list, batch_note['no_change_pairs'])
    batch_note['offset'] = offset
    return {"batch_note": batch_note, "index_list": index_list}


class ModelTrainer:
    # todo 多GPU 训练策略 有无bug
    def __init__(self, epochs=1, initial_epoch=0,
                 checkpoint_dir=None, checkpoint_name="checkpoint.pt", save_all_checkpoints=False,
                 eval_steps=None, evaluate_after=False, validate_metric="accuracy", less_is_better=False):
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            self.checkpoint_path = None
        self.save_all_checkpoints = save_all_checkpoints
        self.eval_steps = eval_steps
        self.evaluate_after = evaluate_after
        self.validate_metric = validate_metric
        self.less_is_better = less_is_better
        self.notelist = {}
        self.lasttrainacc = 0
        self.lastvalacc = 0
        self.reverse = 0
        self.reverse_enable = False
        self.correctlist = {}
        self.errorlist = {}
        self.correct_num = 0
        self.all_num = 0
        self.defaultcorrect = 0
        self.testvalidate = 0

        # todo 多GPU支持 根据note_now操作数据和筛选标签 默认是all
    def do_epoch(self, model, dataloader, mode="validate", epoch=0, eval_steps=None,
                 answer_field="labels", y_field="y",
                 extract_func=None, metric_func=None, aggregate_func=None, display_func=None,
                 ncols=200, dynamic_ncols=False, count_mode="batch", total=None,
                 check_field="input_ids", check_dim=1, max_length=512, **kwargs):
        metrics = {"n_batches": 0, "loss": 0.0}
        func = model.train_on_batch if mode == "train" else model.validate_on_batch
        if count_mode == "batch":
            total = getattr(dataloader, "__len__", None)
        progress_bar = tqdm(total=total, leave=True, ncols=ncols, dynamic_ncols=dynamic_ncols)
        if self.reverse == 1:
            self.reverse_enable = True
        progress_bar.set_description(f"{mode}, epoch={(epoch + 1) if mode == 'train' else epoch}")
        evaluation_step = 0
        with progress_bar:
            for batch in dataloader:
                batch_len = len(batch['indexes'])
                prev_evaluation_step = evaluation_step
                if (mode == "train" and check_field in batch and batch[check_field].shape[check_dim] > max_length):
                    batch_metrics = dict()
                else:
                    batch_answers, mask = batch[y_field], batch.get("mask")
                    if mask is not None:
                        mask = mask.bool()
                    try:
                        if progress_bar.n <= -1 and mode == "train":
                            batch_metrics = dict()
                        else:
                            index_list = {}
                            noteindex = ""
                            for i in range(len(batch["indexes"])):
                                noteindex = noteindex + str(batch["indexes"][i]) + str(batch["start"][i]) + str(
                                    len(batch["label"]))
                            if (len(batch["default"]) > 1):
                                a = 1
                            else:
                                a = 0
                            if kwargs["note_use"] and (mode == "train" or self.testvalidate == 1)and noteindex in self.notelist:
                                batch_note = batch.copy()
                                # 根据不同阶段和权重参数值筛选batch并赋值
                                correct_n = 1
                                reverse_n = 1
                                # todo 加入reverse机制：假标签机制，导致过拟合的数据correct和error置换 + 回退机制
                                if noteindex in self.correctlist:
                                    batch_list = get_batch_reverse_note(self.reverse_enable,
                                                                        self.correctlist[noteindex],
                                                                        self.notelist[noteindex], batch_note,
                                                                        kwargs["note_now"], model=model,
                                                                        correct_n=correct_n, reverse_n=reverse_n,mode=mode)
                                    self.defaultcorrect = self.defaultcorrect + batch_list["defaultcorrect"]
                                    #self.actreverse[noteindex] = batch_list["batch_note"]["reverse_index"].copy()
                                else:
                                    if kwargs["note_now"] == "all":
                                        correct_n = 1
                                    batch_list = get_batch_note(self.notelist[noteindex], batch_note,
                                                                kwargs["note_now"], model=model,
                                                                correct_n=correct_n,mode=mode)
                                batch = batch_list["batch_note"]
                                index_list = batch_list["index_list"]
                                if len(batch["default"]) == 0 or len(index_list) == 0:
                                    continue
                            batch_output = func(batch, mask=mask)

                            # metrics: {'loss': 0.0, 'n_batches': 0}
                            # batch_output: {'bce_loss': tensor(0.3124, device='cuda:0', grad_fn=<DivBackward0>), 'probs': tensor([0.6415, 0.1042, 0.0189, 0.3386], device='cuda:0', grad_fn=<SigmoidBackward0>), 'loss': tensor(0.3124, device='cuda:0', grad_fn=<DivBackward0>), 'soft_loss': tensor(0.2496, device='cuda:0', grad_fn=<MeanBackward0>), 'hard_loss': tensor(0., device='cuda:0'), 'no_change_loss': tensor(0.4224, device='cuda:0', grad_fn=<MeanBackward0>)}
                            # y_field:'label'
                            if (kwargs["note_use"] and mode == "train") or self.testvalidate == 1:
                                stage_n = "note_" + kwargs["note_now"] + "_n"
                                batch_metrics = update_metrics(
                                    metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                    extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func,
                                    note_use=True, threshold=kwargs[stage_n]
                                )
                                # todo 直接修改无法生效到下一个epoch 且epoch每次不可打乱顺序，不然会失效：已实现方法：self参数存储 缺点是占内存，查找慢  方法1.1：读写文件（解决占内存问题） 方法1.2：参数引用可传递（生成器不好保存改动） 方法1.3：每次重新加载dataloader（费时，随机）
                                if noteindex in self.notelist and epoch != 0:
                                    fir = len(batch_metrics["error_list"])
                                    num = 0
                                    # 记录正确率变化的部分
                                    # todo note机制与reverse机制耦合到了一起
                                    for i in range(fir):
                                        sec = len(batch_metrics["error_list"][i])
                                        if sec != 0:
                                            for index in range(sec):
                                                if self.correctflag > 0:
                                                    if noteindex in self.correctlist and index_list[num] in self.correctlist[noteindex]:
                                                        if batch_metrics["error_list"][i][index] == 1 :
                                                            self.correct_num = self.correct_num + 1
                                                        self.all_num = self.all_num + 1
                                                else:
                                                    if index_list[num] in self.errorlist[noteindex] and batch_metrics["error_list"][i][index] == 1 and (num + 1) not in batch["offset"]:
                                                        if noteindex in self.correctlist:
                                                            self.correctlist[noteindex][index_list[num]] = 1
                                                        else:
                                                            self.correctlist[noteindex]={}
                                                            self.correctlist[noteindex][index_list[num]] = 1
                                                self.notelist[noteindex][index_list[num]] = batch_metrics["error_list"][i][index]
                                                num = num + 1
                                    if self.correctflag == 0:
                                        num = 0
                                        self.errorlist[noteindex] = {}
                                        for i in range(fir):
                                            sec = len(batch_metrics["error_list"][i])
                                            if sec != 0:
                                                for index in range(sec):
                                                    if batch_metrics["error_list"][i][index] == 0:
                                                        self.errorlist[noteindex][index_list[num]] = 1
                                                    num = num + 1
                                else:
                                    self.notelist[noteindex] = []
                                    self.errorlist[noteindex] = {}
                                    fir = len(batch_metrics["error_list"])
                                    num = 0
                                    for i in range(fir):
                                        sec = len(batch_metrics["error_list"][i])
                                        if sec != 0:
                                            for index in range(sec):
                                                self.notelist[noteindex].append(batch_metrics["error_list"][i][index])
                                                if batch_metrics["error_list"][i][index] == 0 :
                                                    self.errorlist[noteindex][num] = 1
                                                num = num + 1
                                if count_mode == "sample":
                                    batch_size = batch_len
                            else:
                                batch_metrics = update_metrics(
                                    metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                    extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func
                                )
                                if count_mode == "sample":
                                    batch_size = batch_metrics[
                                        "seq_total"] if "seq_total" in batch_metrics else batch_len

                    except ValueError:
                        continue
                postfix = display_func(metrics)

                progress_bar.update(batch_size if count_mode == "sample" else 1)
                postfix["lr"] = f"{model.scheduler.get_last_lr()[0]:.2e}"
                progress_bar.set_postfix(postfix)
                # todo eval策略修改
                if mode == "train" and eval_steps is not None:
                    evaluation_step = progress_bar.n // eval_steps
                    if evaluation_step != prev_evaluation_step:
                        self.eval_func(model, epoch=f"{epoch}_{progress_bar.n}")
        return metrics
    #flag为1表示激励，为0表示惩罚
    def change_reverse(self, dataloader, flag = 1):
        for batch in dataloader:
            noteindex = ""
            for i in range(len(batch["indexes"])):
                noteindex = noteindex + str(batch["indexes"][i]) + str(batch["start"][i]) + str(
                    len(batch["label"]))
            if noteindex in self.actreverse:
                for index in self.actreverse[noteindex]:
                    if flag == 1 and index in self.errorlist[noteindex]:
                        self.errorlist[noteindex][index] = self.errorlist[noteindex][index] * 0.5
                    elif flag == 0 and index in self.errorlist[noteindex]:
                        self.errorlist[noteindex][index] = self.errorlist[noteindex][index] * 2

    def train(self, model, train_data, dev_data=None, total=None, dev_total=None, count_mode="sample", **kwargs):
        self.notebook_args = {key: value for key, value in kwargs.items() if key[:5] == "note_"}
        self.best_score = np.inf if self.less_is_better else -np.inf
        eval_steps = self.eval_steps if dev_data is not None else None
        self.eval_func = partial(
            self.evaluate_and_save_model, dev_data=dev_data, total=dev_total, count_mode=count_mode, **kwargs
        )
        self.new_lr = kwargs["new_lr"]
        # 添加变换机制
        if kwargs["note_use"]:
            note_num = 0
            self.note_list = kwargs["list"]
            stages = []
            for i in range(0, len(self.note_list)):
                stage = "note_" + self.note_list[i] + "_num"
                num = kwargs[stage]
                for j in range(0, num):
                    stages.append(self.note_list[i])
                note_num = note_num + num
        file = open("/home/amax/data/wzx/error_epoch_th.txt", "a")
        self.correctflag = 0
        self.correctlist = {}
        for epoch in range(self.initial_epoch, self.epochs):
            self.reverse = 1
            self.reverse_enable = True
            # 判断epoch所在阶段
            kwargs["note_now"] = stages[epoch % note_num]
            # 传递阶段值
            train_metrics = self.do_epoch(
                model, train_data, mode="train", epoch=epoch, total=total,
                eval_steps=eval_steps, count_mode=count_mode, **kwargs
            )
            self.correctflag = len(self.correctlist)
            dev_metrics = self.eval_func(model, epoch=epoch + 1)
            if self.correctflag > 0:
                self.all_num = 0
                self.correct_num = 0
                self.defaultcorrect = 0
                self.testvalidate = 1
                train_metrics = self.do_epoch(
                    model, train_data, mode="validate", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
                self.testvalidate = 0
                a = self.correct_num/self.all_num
                file.write("The number is {:.2f}\n".format(a))
                file.write("The all number is {:.2f}\n".format(self.all_num))
                file.write("The default number is {:.2f}\n".format(self.defaultcorrect))
                file.flush()
        file.close()
        if dev_data is not None and self.evaluate_after:
            if self.checkpoint_path is not None and not self.save_all_checkpoints:
                model.load_state_dict(torch.load(self.checkpoint_path))
            self.do_epoch(model, dev_data, mode="validate", epoch="evaluate",
                          total=dev_total, count_mode=count_mode, **kwargs)
        return

    def is_better_score(self, epoch_score, best_score):
        if epoch_score is None:
            return False
        return (self.less_is_better == (epoch_score <= best_score))

    def evaluate_and_save_model(self, model, dev_data, epoch=None, total=None, **kwargs):
        if dev_data is not None:
            dev_metrics = self.do_epoch(model, dev_data, mode="validate", epoch=epoch, total=total, **kwargs)
            epoch_score = dev_metrics.get(self.validate_metric)
            to_save_checkpoint = self.save_all_checkpoints
            if self.is_better_score(epoch_score, self.best_score):
                to_save_checkpoint, self.best_score = True, epoch_score
        else:
            dev_metrics, to_save_checkpoint = None, True
        if to_save_checkpoint and self.checkpoint_path is not None:
            path_to_save = (attach_index(self.checkpoint_path, epoch, "\.pt") if self.save_all_checkpoints
                            else self.checkpoint_path)
            torch.save(model.state_dict(), path_to_save)
        return dev_metrics