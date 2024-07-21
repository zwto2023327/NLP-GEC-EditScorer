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
import gc

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
    TP_list = copy.deepcopy(pred_labels)
    FN_list = copy.deepcopy(pred_labels)
    FP_list = copy.deepcopy(pred_labels)
    for r, (curr_pred_labels, curr_labels) in enumerate(zip(pred_labels, labels)):
        if mask is not None:
            curr_labels = [x for x, flag in zip(curr_labels, mask[r]) if flag]
        elif ignore_labels is not None:
            curr_labels = [label for label in curr_labels if label not in ignore_labels]
        # assert len(curr_pred_labels) == len(curr_labels), f"{len(curr_pred_labels)}-{len(curr_labels)}"
        for key, value in metric_func(curr_labels, curr_pred_labels, threshold=threshold, note_use=note_use).items():
            if key != "error_list" and key != "TP_list" and key != "FN_list" and key != "FP_list":
                answer[key] += value
            else:
                if key == "error_list":
                    error_list[r] = value
                elif key == "TP_list":
                    TP_list[r] = value
                elif key == "FN_list":
                    FN_list[r] = value
                elif key == "FP_list":
                    FP_list[r] = value
    return {"answer": answer, "error_list": error_list, "TP_list": TP_list, "FN_list": FN_list, "FP_list": FP_list}


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
        for key, value in batch_metrics["answer"].items():
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
        if index in correctlist and mode == "train":
            continue
        index_map[index] = index_add
        index_add = index_add + 1
        index_list.append(index)
    batch_note['input_ids'] = torch.index_select(batch_note['input_ids'], 0, torch.tensor(index_list).to(model.device))
    batch_note['words'] = [batch_note['words'][i] for i in index_list]
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
    batch_note['flag'] = [batch_note['flag'][i] for i in index_list]
    batch_note['target'] = [batch_note['target'][i] for i in index_list]
    batch_note['default'] = default
    batch_note['reverse_index'] = reverse_index
    batch_note['indexes'] = [batch_note['indexes'][i] for i in default_index]
    batch_note['hard_pairs'] = replace_index_note(index_map, index_list, batch_note['hard_pairs'])
    batch_note['soft_pairs'] = replace_index_note(index_map, index_list, batch_note['soft_pairs'])
    batch_note['no_change_pairs'] = replace_index_note(index_map, index_list, batch_note['no_change_pairs'])
    batch_note['offset'] = offset
    return {"batch_note": batch_note, "index_list": index_list}


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
    batch_note['words'] = [batch_note['words'][i] for i in index_list]
    batch_note['label'] = torch.index_select(batch_note['label'], 0, torch.tensor(index_list).to(model.device))
    batch_note['start'] = [batch_note['start'][i] for i in index_list]
    batch_note['end'] = [batch_note['end'][i] for i in index_list]
    batch_note['origin_start'] = [batch_note['origin_start'][i] for i in index_list]
    batch_note['origin_end'] = [batch_note['origin_end'][i] for i in index_list]
    batch_note['flag'] = [batch_note['flag'][i] for i in index_list]
    batch_note['target'] = [batch_note['target'][i] for i in index_list]
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
        self.initmodelflag = True
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
        self.all_num = 0
        self.testvalidate = 0
        self.newflag = False
        self.correctflag = 0
        self.correctlist = {}
        self.min_lr = -1
        self.max_lr = -1
        self.replace_num = 0
        self.delete_num = 0
        self.insert_num = 0
        self.null_num = 0
        self.replace_correct = 0
        self.delete_correct = 0
        self.insert_correct = 0
        self.null_correct = 0
        self.correct_replace_num = 0
        self.correct_delete_num = 0
        self.correct_insert_num = 0
        self.correct_replace_correct = 0
        self.correct_delete_correct = 0
        self.correct_insert_correct = 0

        # todo 多GPU支持 根据note_now操作数据和筛选标签 默认是all memory_base=["error_list", "TP_list", "FN_list", "FP_list"]
    def do_epoch(self, model, dataloader, mode="validate", epoch=0, eval_steps=None,
                 answer_field="labels", y_field="y",
                 extract_func=None, metric_func=None, aggregate_func=None, display_func=None,
                 ncols=200, dynamic_ncols=False, count_mode="batch", total=None,
                 check_field="input_ids", check_dim=1, max_length=512, memory_base=["error_list", "TP_list", "FN_list", "FP_list"], **kwargs):
        metrics = {"n_batches": 0, "loss": 0.0}
        self.replace_num = 0
        self.delete_num = 0
        self.insert_num = 0
        self.null_num = 0
        self.replace_correct = 0
        self.delete_correct = 0
        self.insert_correct = 0
        self.null_correct = 0
        self.correct_num = {}
        for metric in memory_base:
            self.correct_num[metric] = 0
        self.all_num = 0
        if self.testvalidate == 1:
            self.correct_replace_num = 0
            self.correct_delete_num = 0
            self.correct_insert_num = 0
            self.correct_replace_correct = 0
            self.correct_delete_correct = 0
            self.correct_insert_correct = 0
        func = model.train_on_batch if mode == "train" else model.validate_on_batch
        if mode == "validate":
            if self.testvalidate == 1:
                path_to_load = attach_index(self.txt_path, epoch, "\.txt")
                if os.path.exists(path_to_load):
                    os.remove(path_to_load)
                epochfile = open(path_to_load, "a")
                path_to_load = attach_index(self.traintxt_path, epoch, "\.txt")
                if os.path.exists(path_to_load):
                    os.remove(path_to_load)
                epochfilet = open(path_to_load, "a")
            else:
                path_to_load = attach_index(self.valtxt_path, epoch, "\.txt")
                if os.path.exists(path_to_load):
                    os.remove(path_to_load)
                epochfilet = open(path_to_load, "a")

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
                            if kwargs["note_use"] and (mode == "train" or self.testvalidate == 1) and noteindex in self.notelist:
                                batch_note = batch.copy()
                                # 根据不同阶段和权重参数值筛选batch并赋值
                                correct_n = 1
                                reverse_n = 1
                                # todo 加入reverse机制：假标签机制，导致过拟合的数据correct和error置换 + 回退机制
                                if noteindex in self.correctlist and "error_list" in self.correctlist[noteindex]:
                                    batch_list = get_batch_reverse_note(self.reverse_enable,
                                                                        self.correctlist[noteindex]["error_list"],
                                                                        self.notelist[noteindex][memory_base[0]], batch_note,
                                                                        kwargs["note_now"], model=model,
                                                                        correct_n=correct_n, reverse_n=reverse_n,mode=mode)

                                    #self.actreverse[noteindex] = batch_list["batch_note"]["reverse_index"].copy()
                                else:
                                    if kwargs["note_now"] == "all":
                                        correct_n = 1
                                    batch_list = get_batch_note(self.notelist[noteindex][memory_base[0]], batch_note,
                                                                kwargs["note_now"], model=model,
                                                                correct_n=correct_n,mode=mode)
                                batch = batch_list["batch_note"]
                                index_list = batch_list["index_list"]
                                if len(batch["default"]) == 0 or len(index_list) == 0:
                                    continue
                            if self.newflag == True:
                                batch_output = func(batch, mask=mask, new_lr=self.new_lr, new_flag=self.newflag)
                                self.newflag = False
                            else:
                                batch_output = func(batch, mask=mask)
                            # metrics: {'loss': 0.0, 'n_batches': 0}
                            # batch_output: {'bce_loss': tensor(0.3124, dev     ice='cuda:0', grad_fn=<DivBackward0>), 'probs': tensor([0.6415, 0.1042, 0.0189, 0.3386], device='cuda:0', grad_fn=<SigmoidBackward0>), 'loss': tensor(0.3124, device='cuda:0', grad_fn=<DivBackward0>), 'soft_loss': tensor(0.2496, device='cuda:0', grad_fn=<MeanBackward0>), 'hard_loss': tensor(0., device='cuda:0'), 'no_change_loss': tensor(0.4224, device='cuda:0', grad_fn=<MeanBackward0>)}
                            # y_field:'label'
                            if (kwargs["note_use"] and mode == "train") or self.testvalidate == 1:
                                stage_n = "note_" + kwargs["note_now"] + "_n"
                                batch_metrics = update_metrics(
                                    metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                    extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func,
                                    note_use=True, threshold=kwargs[stage_n]
                                )
                                # todo 直接修改无法生效到下一个epoch 且epoch每次不可打乱顺序，不然会失效：已实现方法：self参数存储 缺点是占内存，查找慢  方法1.1：读写文件（解决占内存问题） 方法1.2：参数引用可传递（生成器不好保存改动） 方法1.3：每次重新加载dataloader（费时，随机）
                                if noteindex in self.notelist and self.initmodelflag == False:
                                    for metric in memory_base:
                                        fir = len(batch_metrics[metric])
                                        num = 0
                                        # 记录正确率变化的部分
                                        # todo note机制与reverse机制耦合到了一起
                                        for i in range(fir):
                                            sec = len(batch_metrics[metric][i])
                                            if sec != 0:
                                                for index in range(sec):
                                                    if self.notelist[noteindex][metric][index_list[num]] == 0 and batch_metrics[metric][i][index] == 1 and (num + 1) not in batch["offset"] and self.testvalidate == 1:
                                                        epochfilet.write("***")
                                                        epochfilet.write(batch['target'][num])
                                                    if self.correctflag > 0 :
                                                        if self.testvalidate == 1:
                                                            #ra = random.randint(0, 100) / 100
                                                            if noteindex in self.correctlist and metric in self.correctlist[noteindex] and index_list[num] in self.correctlist[noteindex][metric] :#and ra <= 0.1:
                                                                if batch_metrics[metric][i][index] == 1 :
                                                                    self.correct_num[metric] = self.correct_num[metric] + 1#todo correct_num适应
                                                                if metric == "error_list":
                                                                    self.all_num = self.all_num + 1
                                                                    if (num + 1) not in batch["offset"]:
                                                                        if batch['flag'][num] == 0:
                                                                            if batch['target'][num] == '':
                                                                                self.correct_delete_num = self.correct_delete_num + 1
                                                                                if batch_metrics[metric][i][index] == 1:
                                                                                    self.correct_delete_correct = self.correct_delete_correct + 1
                                                                                    if self.notelist[noteindex][metric][index_list[num]] == 0:
                                                                                        epochfile.write("***delete***")
                                                                                        for word in batch["words"][index_list[num]]:
                                                                                            epochfile.write(word)
                                                                                            epochfile.write(" ")
                                                                                        epochfile.write("***delete***\n")
                                                                            else:
                                                                                self.correct_replace_num = self.correct_replace_num + 1
                                                                                if batch_metrics[metric][i][index] == 1:
                                                                                    self.correct_replace_correct = self.correct_replace_correct + 1
                                                                                    if self.notelist[noteindex][metric][index_list[num]] == 0:
                                                                                        epochfile.write("***replace***")
                                                                                        for word in batch["words"][index_list[num]]:
                                                                                            epochfile.write(word)
                                                                                            epochfile.write(" ")
                                                                                        epochfile.write("***replace***\n")
                                                                        if batch['flag'][num] > 0:
                                                                            self.correct_insert_num = self.correct_insert_num + 1
                                                                            if batch_metrics[metric][i][index] == 1:
                                                                                self.correct_insert_correct = self.correct_insert_correct + 1
                                                                                if self.notelist[noteindex][metric][
                                                                                    index_list[num]] == 0:
                                                                                    epochfile.write("***insert***")
                                                                                    for word in batch["words"][index_list[num]]:
                                                                                        epochfile.write(word)
                                                                                        epochfile.write(" ")
                                                                                    epochfile.write("***insert***\n")
                                                                    else:
                                                                        epochfile.write("***default***")
                                                                        for word in batch["words"][index_list[num]]:
                                                                            epochfile.write(word)
                                                                            epochfile.write(" ")
                                                                        epochfile.write("***default***\n")

                                                    else:
                                                        if self.notelist[noteindex][metric][index_list[num]] == 0 and batch_metrics[metric][i][index] == 1 and (num + 1) not in batch["offset"]:
                                                            if noteindex not in self.correctlist:
                                                                self.correctlist[noteindex] = {}
                                                            if metric not in self.correctlist[noteindex]:
                                                                self.correctlist[noteindex][metric] = {}
                                                            self.correctlist[noteindex][metric][index_list[num]] = 1
                                                    self.notelist[noteindex][metric][index_list[num]] = batch_metrics[metric][i][index]
                                                    #todo 要不要取消积累机制
                                                    num = num + 1
                                else:
                                    self.notelist[noteindex] = {}
                                    for metric in memory_base:
                                        self.notelist[noteindex][metric] = []
                                        fir = len(batch_metrics[metric])
                                        num = 0
                                        for i in range(fir):
                                            sec = len(batch_metrics[metric][i])
                                            if sec != 0:
                                                for index in range(sec):
                                                    self.notelist[noteindex][metric].append(batch_metrics[metric][i][index])
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
                if noteindex not in self.notelist and mode == "validate":
                    self.notelist[noteindex] = {}
                    for metric in memory_base:
                        self.notelist[noteindex][metric] = []
                        fir = len(batch_metrics[metric])
                        num = 0
                        for i in range(fir):
                            sec = len(batch_metrics[metric][i])
                            if sec != 0:
                                for index in range(sec):
                                    self.notelist[noteindex][metric].append(batch_metrics[metric][i][index])
                                    num = num + 1
                #统计总的词法特征
                if mode == "validate" or self.testvalidate == 1:
                    for metric in memory_base:
                        if metric == "error_list":
                            fir = len(batch_metrics[metric])
                            num = 0
                            for i in range(fir):
                                sec = len(batch_metrics[metric][i])
                                if sec != 0:
                                    for index in range(sec):
                                        if (num + 1) in batch["offset"]:
                                            self.null_num = self.null_num + 1
                                            if batch_metrics[metric][i][index] == 1:
                                                self.null_correct = self.null_correct + 1
                                            epochfilet.write("***default***")
                                            for word in batch["words"][num]:
                                                epochfilet.write(word)
                                                epochfilet.write(" ")
                                            epochfilet.write("***default***\n")
                                        else:
                                            if batch['flag'][num] == 0:
                                                if batch['target'][num] == '':
                                                    self.delete_num = self.delete_num + 1
                                                    if batch_metrics[metric][i][index] == 1:
                                                        self.delete_correct = self.delete_correct + 1
                                                        if self.notelist[noteindex][metric][
                                                            num] == 0:
                                                            epochfilet.write("***delete***")
                                                            for word in batch["words"][num]:
                                                                epochfilet.write(word)
                                                                epochfilet.write(" ")
                                                            epochfilet.write("***delete***\n")
                                                else:
                                                    self.replace_num = self.replace_num + 1
                                                    if batch_metrics[metric][i][index] == 1:
                                                        self.replace_correct = self.replace_correct + 1
                                                        if self.notelist[noteindex][metric][num] == 0:
                                                            epochfilet.write("***replace***")
                                                            for word in batch["words"][num]:
                                                                epochfilet.write(word)
                                                                epochfilet.write(" ")
                                                            epochfilet.write("***replace***\n")
                                            if batch['flag'][num] > 0:
                                                self.insert_num = self.insert_num + 1
                                                if batch_metrics[metric][i][index] == 1:
                                                    self.insert_correct = self.insert_correct + 1
                                                    if self.notelist[noteindex][metric][
                                                        num] == 0:
                                                        epochfilet.write("***insert***")
                                                        for word in batch["words"][num]:
                                                            epochfilet.write(word)
                                                            epochfilet.write(" ")
                                                        epochfilet.write("***insert***\n")
                                        self.notelist[noteindex][metric][num] = batch_metrics[metric][i][index]
                                        num = num + 1
                progress_bar.update(batch_size if count_mode == "sample" else 1)
                postfix["lr"] = f"{model.scheduler.get_last_lr()[0]:.2e}"
                progress_bar.set_postfix(postfix)
                # todo eval策略修改
                if mode == "train" and eval_steps is not None:
                    evaluation_step = progress_bar.n // eval_steps
                    if evaluation_step != prev_evaluation_step:
                        self.eval_func(model, epoch=f"{epoch}_{progress_bar.n}")
        if mode == "validate":
            if self.testvalidate == 1:
                epochfile.flush()
                epochfilet.flush()
        return metrics
    #flag为1表示激励，为0表示惩罚
    '''def change_reverse(self, dataloader, flag = 1):
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
                        self.errorlist[noteindex][index] = self.errorlist[noteindex][index] * 2'''

    def train(self, model, train_data, dev_data=None, total=None, dev_total=None, count_mode="sample", memory_base=["error_list", "TP_list", "FN_list", "FP_list"], **kwargs):
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
        file_path = "/home/boot/wzx/error_epoch_th.txt"
        if os.path.exists(file_path):
            os.remove(file_path)
        file = open(file_path, "a")
        self.txt_path = "/home/boot/wzx/epoch_txt_th/"
        self.valtxt_path = "/home/boot/wzx/epoch_val_txt_th/"
        self.traintxt_path = "/home/boot/wzx/epoch_train_txt_th/"
        if not os.path.exists(self.txt_path):
            os.mkdir(self.txt_path)
        if not os.path.exists(self.valtxt_path):
            os.mkdir(self.valtxt_path)
        if not os.path.exists(self.traintxt_path):
            os.mkdir(self.traintxt_path)
        dev_metrics = self.eval_func(model, epoch=self.initial_epoch)
        self.initmodel = True
        for epoch in range(self.initial_epoch, self.epochs):
            self.reverse = 1
            self.reverse_enable = True
            # 判断epoch所在阶段
            kwargs["note_now"] = stages[epoch % note_num]
            if self.correctflag == 0 and self.initmodel == True and self.newflag == True:
                path_to_load = attach_index(self.checkpoint_path, self.initial_epoch, "\.pt")
                model.load_state_dict(torch.load(path_to_load), False)
                del self.notelist
                gc.collect()
                self.notelist = {}
                file.write("===================init=======================\n")
                file.flush()
                self.initmodelflag = True
                # 传递阶段值
                train_metrics = self.do_epoch(
                    model, train_data, mode="train", epoch=self.initial_epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )

            else:
                # 传递阶段值
                train_metrics = self.do_epoch(
                    model, train_data, mode="train", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
            '''TP = sum(int(edit["is_correct"]) for elem in chain.from_iterable(extracted) for edit in elem)
                        pred = sum(len(elem) for elem in chain.from_iterable(extracted))
                        total = sum(int(edit.start >= 0) for elem in source_data for edit in elem["edits"])
                        to_save = {
                            "alpha": alpha, "threshold": threshold, "alpha_threshold": alpha_threshold,
                            "max_edits": n_max,  "TP": TP, "FP": pred-TP, "FN": total-TP,
                            "P": round(100 * TP / max(pred, 1), 2), "R": round(100 * TP / max(total, 1), 2),
                            "F": round(100 * TP / max(0.2 * total + 0.8 * pred, 1), 2)
                        }'''
            self.initmodelflag = False
            #todo 位置细节
            if len(self.correctlist) > 0 and self.correctflag == 0:
                self.correctflag = 1
            dev_metrics = self.eval_func(model, epoch=epoch + 1)
            self.trainacc = train_metrics["accuracy"]
            self.valacc = dev_metrics["accuracy"]
            if self.replace_num != 0:
                self.val_replace = self.replace_correct/self.replace_num
            else:
                self.val_replace = 0
            if self.delete_num != 0:
                self.val_delete = self.delete_correct/self.delete_num
            else:
                self.val_delete = 0
            if self.insert_num != 0:
                self.val_insert = self.insert_correct/self.insert_num
            else:
                self.val_insert = 0
            if self.null_num != 0:
                self.val_null = self.null_correct/self.null_num
            else:
                self.val_null = 0
            if self.correctflag == 1:
                self.testvalidate = 1
                train_metrics = self.do_epoch(
                    model, train_data, mode="validate", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
                if self.replace_num != 0:
                    self.train_replace = self.replace_correct / self.replace_num
                else:
                    self.train_replace = 0
                if self.delete_num != 0:
                    self.train_delete = self.delete_correct / self.delete_num
                else:
                    self.train_delete = 0
                if self.insert_num != 0:
                    self.train_insert = self.insert_correct / self.insert_num
                else:
                    self.train_insert = 0
                if self.null_num != 0:
                    self.train_null = self.null_correct / self.null_num
                else:
                    self.train_null = 0
                if self.correct_replace_num != 0:
                    self.correct_train_replace = self.correct_replace_correct / self.correct_replace_num
                else:
                    self.correct_train_replace = 0
                if self.correct_delete_num != 0:
                    self.correct_train_delete = self.correct_delete_correct / self.correct_delete_num
                else:
                    self.correct_train_delete = 0
                if self.correct_insert_num != 0:
                    self.correct_train_insert = self.correct_insert_correct / self.correct_insert_num
                else:
                    self.correct_train_insert = 0
                self.testvalidate = 0
                if self.all_num == 0:
                    file.write("The all_num is zero,skip now\n")
                    del self.correctlist
                    gc.collect()
                    self.correctlist = {}
                    self.correctflag = 0
                    file.flush()
                    continue
                self.correctflag = 1
                file.write("The epoch is {:.1f}\n".format(epoch))
                file.write("The trainacc is {:.3f}\n".format(self.trainacc))
                file.write("The valacc is {:.3f}\n".format(self.valacc))
                for metric in memory_base:
                    if metric == "error_list":
                        self.lastmero = self.correct_num[metric] / self.all_num
                        self.stagenum = self.lastmero
                        file.write("The number is {:.4f}\n".format(self.lastmero))
                    elif metric == "TP_list":
                        file.write("The TP_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_TP is {:.3f}\n".format(dev_metrics['TP']))
                        file.write("The train_TP is {:.3f}\n".format(train_metrics['TP']))
                    elif metric == "FN_list":
                        file.write("The FN_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_FN is {:.3f}\n".format(dev_metrics['FN']))
                        file.write("The train_FN is {:.3f}\n".format(train_metrics['FN']))
                    elif metric == "FP_list":
                        file.write("The FP_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_FP is {:.3f}\n".format(dev_metrics['FP']))
                        file.write("The train_FP is {:.3f}\n".format(train_metrics['FP']))
                if "TP_list" in memory_base and "FN_list" in memory_base and "FP_list" in memory_base:
                    file.write("The val_F is {:.3f}\n".format(dev_metrics['F']))
                    file.write("The val_R is {:.3f}\n".format(dev_metrics['recall']))
                    file.write("The val_P is {:.3f}\n".format(dev_metrics['precision']))
                    file.write("The train_F is {:.3f}\n".format(train_metrics['F']))
                    file.write("The train_R is {:.3f}\n".format(train_metrics['recall']))
                    file.write("The train_P is {:.3f}\n".format(train_metrics['precision']))
                    F_num = round(100 * self.correct_num["TP_list"] / max(
                        0.2 * (self.correct_num["TP_list"] + self.correct_num["FN_list"]) + 0.8 * (self.correct_num["TP_list"] + self.correct_num["FP_list"]), 1), 2)
                    file.write("The F_number is {:.3f}\n".format(F_num))
                '''file.write("The replace is {:.3f}\n".format(self.val_replace))
                file.write("The insert is {:.3f}\n".format(self.val_insert))
                file.write("The null is {:.3f}\n".format(self.val_null))
                file.write("The delete is {:.3f}\n".format(self.val_delete))
                file.write("The train_replace is {:.3f}\n".format(self.train_replace))
                file.write("The train_insert is {:.3f}\n".format(self.train_insert))
                file.write("The train_null is {:.3f}\n".format(self.train_null))
                file.write("The train_delete is {:.3f}\n".format(self.train_delete))
                file.write("The train_replace_num is {:.3f}\n".format(self.replace_num))
                file.write("The train_insert_num is {:.3f}\n".format(self.insert_num))
                file.write("The train_null_num is {:.3f}\n".format(self.null_num))
                file.write("The train_delete_num is {:.3f}\n".format(self.delete_num))
                file.write("The correct_replace is {:.3f}\n".format(self.correct_train_replace))
                file.write("The correct_insert is {:.3f}\n".format(self.correct_train_insert))
                file.write("The correct_delete is {:.3f}\n".format(self.correct_train_delete))
                file.write("The correct_replace num is {:.1f}\n".format(self.correct_replace_num))
                file.write("The correct_insert num is {:.1f}\n".format(self.correct_insert_num))
                file.write("The correct_delete num is {:.1f}\n".format(self.correct_delete_num))'''
                file.write("The lr is {:.4e}\n".format(self.new_lr))
                file.write("\n")
                file.flush()
            elif self.correctflag == 2:
                lr = self.new_lr
                self.testvalidate = 1
                train_metrics = self.do_epoch(
                    model, train_data, mode="validate", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
                if self.replace_num != 0:
                    self.train_replace = self.replace_correct / self.replace_num
                else:
                    self.train_replace = 0
                if self.delete_num != 0:
                    self.train_delete = self.delete_correct / self.delete_num
                else:
                    self.train_delete = 0
                if self.insert_num != 0:
                    self.train_insert = self.insert_correct / self.insert_num
                else:
                    self.train_insert = 0
                if self.null_num != 0:
                    self.train_null = self.null_correct / self.null_num
                else:
                    self.train_null = 0
                if self.correct_replace_num != 0:
                    self.correct_train_replace = self.correct_replace_correct / self.correct_replace_num
                else:
                    self.correct_train_replace = 0
                if self.correct_delete_num != 0:
                    self.correct_train_delete = self.correct_delete_correct / self.correct_delete_num
                else:
                    self.correct_train_delete = 0
                if self.correct_insert_num != 0:
                    self.correct_train_insert = self.correct_insert_correct / self.correct_insert_num
                else:
                    self.correct_train_insert = 0
                self.testvalidate = 0
                if self.all_num == 0:
                    file.write("The all_num is zero,skip now\n")
                    del self.correctlist
                    gc.collect()
                    self.correctlist = {}
                    self.correctflag = 0
                    file.flush()
                    continue
                for metric in memory_base:
                    if metric == "error_list":
                        self.lastmero = self.correct_num[metric] / self.all_num
                        self.stagenum = self.lastmero
                        file.write("The number is {:.4f}\n".format(self.lastmero))
                    elif metric == "TP_list":
                        file.write("The TP_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_TP is {:.3f}\n".format(dev_metrics['TP']))
                        file.write("The train_TP is {:.3f}\n".format(train_metrics['TP']))
                    elif metric == "FN_list":
                        file.write("The FN_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_FN is {:.3f}\n".format(dev_metrics['FN']))
                        file.write("The train_FN is {:.3f}\n".format(train_metrics['FN']))
                    elif metric == "FP_list":
                        file.write("The FP_list is {:.3f}\n".format(self.correct_num[metric]))
                        file.write("The val_FP is {:.3f}\n".format(dev_metrics['FP']))
                        file.write("The train_FP is {:.3f}\n".format(train_metrics['FP']))
                self.renum = 0
                self.nochangenum = 0
                self.flagnum = 0#0表示上升，1表示下降
                if self.nowmero > self.lastmero + 0.001:
                    if self.flagnum == 0:
                        self.renum = self.renum + 1
                    else:
                        self.renum = 0
                    self.flagnum = 0
                elif self.nowmero < self.lastmero - 0.001:
                    if self.flagnum == 1:
                        self.renum = self.renum + 1
                    else:
                        self.renum = 0
                    self.flagnum = 1
                if self.initmodel == True:
                    if (self.nowmero > self.lastmero + 0.005) or (self.renum > 1 and self.nowmero > self.lastmero):
                        self.min_lr = self.new_lr
                        # todo 有波动的情况 要不要退化
                        if self.max_lr > -1 and self.max_lr > self.new_lr:
                            self.new_lr = self.new_lr + (1 / 2) * (self.max_lr - self.new_lr)
                        else:
                            self.new_lr = self.new_lr * 10
                        self.newflag = True
                        self.renum = 0
                        self.correctflag = 0
                        self.nochangenum = 0
                    elif (self.nowmero < self.lastmero - 0.005) or (self.renum > 1 and self.nowmero < self.lastmero):
                        # todo 有波动的情况 要不要退化
                        self.max_lr = self.new_lr
                        if self.min_lr > -1 and self.min_lr < self.new_lr:
                            self.new_lr = self.new_lr - (1 / 2) * (self.new_lr - self.min_lr)
                        else:
                            self.new_lr = self.new_lr * (1 / 10)
                        self.newflag = True
                        self.renum = 0
                        self.correctflag = 0
                        self.nochangenum = 0
                    else:
                        #todo 三种情况：过大 过小 正好
                        self.correctflag = 2
                        self.initmodel = False
                        self.nochangenum = self.nochangenum + 1
                else:
                    if (self.nowmero > self.stagenum + 0.01) or (self.renum > 1 and self.nowmero > self.lastmero):
                        self.new_lr = self.new_lr*1.2
                        self.newflag = True
                        self.renum = 0
                        self.correctflag = 0
                        self.nochangenum = 0
                    elif (self.nowmero < self.stagenum - 0.01) or (self.renum > 1 and self.nowmero < self.lastmero):
                        # todo 有波动的情况 要不要退化
                        self.new_lr = self.new_lr*0.8
                        self.newflag = True
                        self.renum = 0
                        self.correctflag = 0
                        self.nochangenum = 0
                    else:
                        #todo 三种情况：过大 过小 正好
                        self.correctflag = 2
                        self.nochangenum = self.nochangenum + 1
                if self.nochangenum > 5:
                    self.new_lr = self.new_lr * 0.9
                    self.newflag = True
                    self.renum = 0
                    self.correctflag = 0
                    self.nochangenum = 0
                if self.newflag == True:
                    del self.correctlist
                    del self.notelist
                    gc.collect()
                    self.initmodelflag = True
                    self.notelist = {}
                    self.correctlist = {}
                file.write("The epoch is {:.1f}\n".format(epoch))
                file.write("The trainacc is {:.3f}\n".format(self.trainacc))
                file.write("The valacc is {:.3f}\n".format(self.valacc))

                if "TP_list" in memory_base and "FN_list" in memory_base and "FP_list" in memory_base:
                    file.write("The val_F is {:.3f}\n".format(dev_metrics['F']))
                    file.write("The val_R is {:.3f}\n".format(dev_metrics['recall']))
                    file.write("The val_P is {:.3f}\n".format(dev_metrics['precision']))
                    file.write("The train_F is {:.3f}\n".format(train_metrics['F']))
                    file.write("The train_R is {:.3f}\n".format(train_metrics['recall']))
                    file.write("The train_P is {:.3f}\n".format(train_metrics['precision']))
                    F_num = round(100 * self.correct_num["TP_list"] / max(
                        0.2 * (self.correct_num["TP_list"] + self.correct_num["FN_list"]) + 0.8 * (
                                    self.correct_num["TP_list"] + self.correct_num["FP_list"]), 1), 2)
                    file.write("The F_number is {:.3f}\n".format(F_num))
                '''file.write("The replace is {:.3f}\n".format(self.val_replace))
                file.write("The insert is {:.3f}\n".format(self.val_insert))
                file.write("The null is {:.3f}\n".format(self.val_null))
                file.write("The delete is {:.3f}\n".format(self.val_delete))
                file.write("The train_replace is {:.3f}\n".format(self.train_replace))
                file.write("The train_insert is {:.3f}\n".format(self.train_insert))
                file.write("The train_null is {:.3f}\n".format(self.train_null))
                file.write("The train_delete is {:.3f}\n".format(self.train_delete))
                file.write("The correct_replace is {:.3f}\n".format(self.correct_train_replace))
                file.write("The correct_insert is {:.3f}\n".format(self.correct_train_insert))
                file.write("The correct_delete is {:.3f}\n".format(self.correct_train_delete))
                file.write("The correct_replace num is {:.1f}\n".format(self.correct_replace_num))
                file.write("The correct_insert num is {:.1f}\n".format(self.correct_insert_num))
                file.write("The correct_delete num is {:.1f}\n".format(self.correct_delete_num))'''
                file.write("The lr is {:.4e}\n".format(self.new_lr))
                file.write("\n")
                file.flush()
            else:
                file.write("The epoch is {:.4f}\n".format(epoch))
                file.write("The xtrainacc is {:.3f}\n".format(self.trainacc))
                file.write("The xvalacc is {:.3f}\n".format(self.valacc))
                file.write("The xreplace is {:.3f}\n".format(self.val_replace))
                file.write("The xinsert is {:.3f}\n".format(self.val_insert))
                file.write("The xnull is {:.3f}\n".format(self.val_null))
                file.write("The xdelete is {:.3f}\n".format(self.val_delete))
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