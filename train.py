from time import time
import orjson
import json
from argparse import ArgumentParser
from functools import partial

import sys
import os

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from numpyencoder import NumpyEncoder

from utils.data_utils import load_ranking_dataset
from common.training import get_batch_metrics, ModelTrainer
from common.metrics import aggregate_binary_sequence_metrics, display_metrics
from ranking.metrics import extract_labels, item_score_func, evaluate_predictions
from ranking.data import prepare_dataset, prepare_dataloader, output_predictions
from ranking.model import VariantScoringModel, VariantScoringModelWithAdditionalLayers, \
    VariantScoringModelWithCrossAttention, predict_with_model


argument_parser = ArgumentParser()
argument_parser.add_argument("-t", "--train_file", nargs="+", default="/home/amax/data/wzx/VSR/NLP-GEC/data/bea_reranking/gector_variants/bea.train.variants")
argument_parser.add_argument("--processed_dataset", action="store_true")
argument_parser.add_argument("--only_generated", action="store_true")
argument_parser.add_argument("--only_with_positive", action="store_true")
argument_parser.add_argument("--save_processed_train", default=None)
argument_parser.add_argument("--save_processed_dev", default=None)
argument_parser.add_argument("--only_process_dataset", action="store_true")
argument_parser.add_argument("--language", default=None)
argument_parser.add_argument("--max_length", default=200, type=int)
argument_parser.add_argument("-T", "--test_file", default="/home/amax/data/wzx/VSR/NLP-GEC/data/bea_reranking/gector_variants/bea.dev.variants")
argument_parser.add_argument("-n", "--max_sents", default=None, type=int)
argument_parser.add_argument("-m", "--model", default="/home/amax/data/wzx/VSR/NLP-GEC/NLP-GEC-EditScorer/roberta-base")
argument_parser.add_argument("-L", "--load", default="/home/amax/data/wzx/VSR/NLP-GEC/NLP-GEC-EditScorer/checkpoints/checkpoints/pie_bea-gector/checkpoint_2.pt")
#argument_parser.add_argument("-L", "--load", default="/home/amax/data/wzx/VSR/NLP-GEC/NLP-GEC-EditScorer/checkpoint_26.pt")
argument_parser.add_argument("-a", "--attention_layers", default=0, type=int)
argument_parser.add_argument("-C", "--cross_attention", action="store_true")
argument_parser.add_argument("-r", "--residual", action="store_true")
argument_parser.add_argument("-U", dest="use_position", action="store_false")
argument_parser.add_argument("--position_mode", default="first", choices=['first', 'last', 'mean'])
argument_parser.add_argument("-O", "--use_origin", action="store_true")
argument_parser.add_argument("--alpha_contrastive", default=0.0, type=float)
argument_parser.add_argument("--average_loss_for_sample", dest="average_loss_for_batch", action="store_false")
argument_parser.add_argument("-I", "--concat_mode", default=None, type=str, choices=['infersent'])
argument_parser.add_argument("-M", "--mlp_hidden", default=768, type=lambda x:eval(f"[{x}]"))
argument_parser.add_argument("--mlp_dropout", default=0.0, type=float)
argument_parser.add_argument("--loss_by_class", action="store_true")
argument_parser.add_argument("--init_with_last_layer", action="store_true")
argument_parser.add_argument("-P", "--alpha_pos", default=1.0, type=float)
argument_parser.add_argument("-S", "--alpha_soft", default=0.0, type=float)
argument_parser.add_argument("-H", "--alpha_hard", default=0.0, type=float)
argument_parser.add_argument("-N", "--alpha_no_change", default=0.0, type=float)
argument_parser.add_argument("-c", "--checkpoint_dir", default="/home/amax/data/wzx/VSR/NLP-GEC/NLP-GEC-EditScorer/checkpoints/")
argument_parser.add_argument("--save_all_checkpoints", action="store_true")
argument_parser.add_argument("-b", "--batch_size", default=64, type=int)
argument_parser.add_argument("-e", "--epochs", default=200, type=int)
argument_parser.add_argument("--initial_epoch", default=0, type=int)
argument_parser.add_argument("--eval_every_n_steps", dest="eval_steps", default=None, type=int)
argument_parser.add_argument("-E", "--recall_estimate", default=0.4, type=float)
argument_parser.add_argument("--lr", default=2e-6, type=float)
argument_parser.add_argument("--attention_lr", default=None, type=float)
argument_parser.add_argument("--clip", default=None, type=float)
argument_parser.add_argument("--batches_per_update", default=1, type=int)
argument_parser.add_argument("--scheduler", default="constant", choices=["constant", "constant_with_warmup"])
argument_parser.add_argument("--warmup", default=0, type=int)
argument_parser.add_argument("-s", "--seed", default=117, type=int)
argument_parser.add_argument("--threshold", default=0.5, type=float)
argument_parser.add_argument("-o", "--outfile", default=None)
argument_parser.add_argument("--min_diff", default=None, type=float)
#错题本训练参数 note_correct_n表示correct阶段采取的阈值 note_correct表示correct阶段正确样本占比 note_error表示error阶段正确样本占比
argument_parser.add_argument("--note_correct_n", default=0.5, type=float)
argument_parser.add_argument("--note_error_n", default=0.5, type=float)
argument_parser.add_argument("--note_all_n", default=0.5, type=float)
argument_parser.add_argument("--note_keep_n", default=0.5, type=float)
argument_parser.add_argument("--note_correct_num", default=1, type=int)
argument_parser.add_argument("--note_error_num", default=3, type=int)
argument_parser.add_argument("--note_all_num", default=10, type=int)
argument_parser.add_argument("--note_keep_num", default=0, type=int)
argument_parser.add_argument("--note_correct", default=1, type=float)
argument_parser.add_argument("--note_error", default=0, type=float)
argument_parser.add_argument("--note_all", default=1, type=float)
argument_parser.add_argument("--note_all_reverse", default=1, type=float)
argument_parser.add_argument("--note_use", default=True, type=bool)

NOTEBOOK_KEYS = ["note_use", "note_correct_n", "note_error_n", "note_all_n", "note_keep_n", "note_correct", "note_error", "note_all", "note_all_reverse", "note_correct_num", "note_error_num", "note_all_num", "note_keep_num"]
MODEL_KEYS = ["position_mode", "loss_by_class", "alpha_pos", "alpha_soft", "alpha_hard", "alpha_no_change",
              "alpha_contrastive", "average_loss_for_batch", "use_origin", "concat_mode", "mlp_dropout"]
SAVE_KEYS = ["mlp_hidden", "epochs"]
OPTIMIZER_KEYS = ["lr", "clip", "scheduler", "warmup", "batches_per_update"]

if __name__ == "__main__":
    args = argument_parser.parse_args()
    args.alpha_hard = max(args.alpha_hard, args.alpha_soft)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.use_origin:
        args.position_mode = "mean"
    dataset_args = {
        "wrap_empty_edits": args.use_origin, "use_sep_for_default": args.use_origin,
        "language": args.language, "min_diff": args.min_diff
    }
    print("Loading data...")
    if args.processed_dataset:
        if isinstance(args.train_file, (list, tuple)):
            args.train_file = args.train_file[0]#train_file如果是一个集合只取第一个
        #todo 如果有超过训练集的数据，需要添加
        train_dataset, train_data = load_ranking_dataset(args.train_file)
        dev_dataset, dev_data = load_ranking_dataset(args.test_file)
    else:
        #todo 如果有超过训练集的数据，需要添加
        train_dataset, train_data = prepare_dataset(
            args.train_file, model=args.model, only_generated=args.only_generated,
            only_with_positive=args.only_with_positive, n=args.max_sents, **dataset_args
        )
        dev_dataset, dev_data = prepare_dataset(args.test_file, model=args.model, n=args.max_sents, **dataset_args)
        t1 = time()
        if args.save_processed_train:
            with open(args.save_processed_train, "wb") as fout:
                # json.dump([train_dataset, train_data], fout, cls=NumpyEncoder)
                fout.write(orjson.dumps([train_dataset, train_data], option=orjson.OPT_SERIALIZE_NUMPY))
        if args.save_processed_dev:
            with open(args.save_processed_dev, "wb") as fout:
                # json.dump([dev_dataset, dev_data], fout, cls=NumpyEncoder)
                fout.write(orjson.dumps([dev_dataset, dev_data], option=orjson.OPT_SERIALIZE_NUMPY))
        t2 = time()
        print(f"Total saving time {(t2-t1):.2f}")
        if args.only_process_dataset:
            sys.exit()
    print("Train dataset length before length filtering", len(train_dataset))

    short_indexes = [i for i, elem in enumerate(train_dataset)
                     if elem["data"] is not None and len(elem["data"][0]["input_ids"]) <= args.max_length]
    train_dataset = [train_dataset[i] for i in short_indexes]
    train_data = [train_data[i] for i in short_indexes]
    print("Train dataset length after length filtering", len(train_dataset))
    # assert all(len(elem["data"][0]["input_ids"]) <= args.max_length for elem in dev_dataset)
    print("Initializing the model...")
    #提取参数
    model_args = {key: getattr(args, key) for key in MODEL_KEYS}
    optimizer_args = {key: getattr(args, key) for key in OPTIMIZER_KEYS}
    notebook_args = {key: getattr(args, key) for key in NOTEBOOK_KEYS}
    if args.attention_layers > 0:
        model_args["n_attention_layers"] = args.attention_layers
        model_args["residual"] = args.residual
        model_args["init_with_last_layer"] = args.init_with_last_layer
        if args.attention_lr is not None:
            optimizer_args["optimizer_attention_lr"] = args.attention_lr
        cls = VariantScoringModelWithAdditionalLayers
    elif args.cross_attention:
        model_args["residual"] = args.residual
        model_args["cross_attention"] = True
        cls = VariantScoringModelWithCrossAttention
    else:
        cls = VariantScoringModel
    if args.note_use:
        #阶段顺序
        notebook_args["list"] = ["all","all"]#todo 支持"keep"
    #todo 多GPU
    notebook_args["new_lr"] = optimizer_args["lr"]
    model = cls(model=args.model, mlp_hidden=args.mlp_hidden, device="cuda",
                use_position=args.use_position, **model_args, **optimizer_args, **notebook_args)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load), False)
    torch.manual_seed(args.seed)
    #todo 无需修改数据 多GPU
    train_dataloader = prepare_dataloader(train_dataset, batch_size=args.batch_size, device=model.device)
    dev_dataloader = prepare_dataloader(dev_dataset, batch_size=args.batch_size, device=model.device)
    if args.recall_estimate is not None:
        metrics_to_display = ["recall_estimate", "F_estimate"]
        percent_metrics = ["recall_estimate", "F_estimate"]
        validate_metric = "F_estimate"
    else:
        metrics_to_display, percent_metrics, validate_metric = None, None, "F"
    metric_args = {
        "y_field": "label",
        "extract_func": extract_labels, "metric_func": item_score_func,
        "aggregate_func": partial(aggregate_binary_sequence_metrics, alpha=0.5,
                                  recall_estimate=args.recall_estimate),
        "display_func": partial(
            display_metrics, metrics_to_display=metrics_to_display, percent_metrics=percent_metrics,
            only_main_loss=bool(args.alpha_contrastive == 0.0)
        )
    }
    #todo 增加无需修改类别
    progress_bar_args = {
        "total": len(train_dataset), "dev_total": len(dev_dataset), "count_mode": "sample"
    }
    if args.checkpoint_dir is not None:
        config = model_args.copy()
        config.update({key: getattr(args, key) for key in SAVE_KEYS})
        config["cls"] = cls.__name__
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "config.json"), "w", encoding="utf8") as fout:
            json.dump(config, fout, indent=2)
    model_trainer = ModelTrainer(
        epochs=args.epochs, initial_epoch=args.initial_epoch, eval_steps=args.eval_steps,
        checkpoint_dir=args.checkpoint_dir, save_all_checkpoints=args.save_all_checkpoints,
        validate_metric=validate_metric, evaluate_after=True
    )
    model_trainer.train(model, train_dataloader, dev_dataloader, **metric_args, **progress_bar_args, **notebook_args)
    # train_model(model, train_dataloader, dev_dataloader, epochs=args.epochs, initial_epoch=args.initial_epoch,
    #             checkpoint_dir=args.checkpoint_dir, save_all_checkpoints=args.save_all_checkpoints,
    #             validate_metric=validate_metric, evaluate_after=True,
    #             **metric_args, **progress_bar_args)
    predictions = predict_with_model(model, dev_dataset, batch_size=args.batch_size, threshold=args.threshold)
    if args.checkpoint_dir is not None:
        fmetrics = open(os.path.join(args.checkpoint_dir, "metrics.jsonl"), "w", encoding="utf8")
    else:
        fmetrics = None
    #todo 阈值+训练策略
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = evaluate_predictions(predictions, dev_dataset, from_labels=False, threshold=threshold)
        print(threshold, metrics)
        if fmetrics is not None:
            metrics["threshold"] = threshold
            json.dump(metrics, fmetrics)
    if fmetrics is not None:
        fmetrics.close()
    if args.outfile is not None:
        output_predictions(predictions, dev_data, file=args.outfile)
    if args.checkpoint_dir is not None:
        outfile = os.path.join(args.checkpoint_dir, "output.out")
        output_predictions(predictions, dev_data, file=outfile)
    # for i in range(50):
    #     for j, batch in enumerate(train_dataloader):
    #         loss = model.train_on_batch(batch)
    #         y_pred, y_true = extract_labels(loss, batch)
    #         stats = dict(get_batch_metrics(y_pred, y_true, metric_func=item_score_func))
    #         print(i, j, end=" ")
    #         for key, value in loss.items():
    #             if "loss" in key:
    #                 print(f"{key}={value:.2f}", end=" ")
    #         # print("")
    #         # print(*([round(x, 2) for elem in y_pred for x in elem]))
    #         # print(batch["label"].tolist())
    #         for key, value in stats.items():
    #             print(f"{key}={value}", end=" ")
    #         print("")