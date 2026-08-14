import sys
import logging
import copy
import torch
import time
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.efficiency import (
    MetricsLogger,
    PhaseTimer,
    dir_size_mb,
    inference_flops_per_sample,
    profile_flops,
)
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    os.makedirs("tosca", exist_ok=True)
    # exist_ok=True: concurrent array tasks (5 seeds of the same dataset)
    # share this directory and race to create it if checked-then-created.
    os.makedirs(logs_name, exist_ok=True)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    metrics_log = MetricsLogger(logfilename + "_metrics.json", args)

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        # Opt-in (config flag `profile_train_flops`, off by default): measure
        # the REAL forward+backward FLOPs of this task's actual training
        # call via torch.profiler, instead of the analytic 3x-forward
        # estimate computed later in aggregate_results.py. Real profiling
        # naturally captures partial-freeze nuances (e.g. only backbone.tosca
        # + fc need gradients on tasks > 0) without special-casing. Off by
        # default -- meant for short, reduced-epoch dedicated runs.
        train_flops_measured = None
        if args.get("profile_train_flops", False):
            with PhaseTimer() as train_phase:
                train_flops_measured = profile_flops(
                    model.incremental_train, data_manager
                )
        else:
            with PhaseTimer() as train_phase:
                model.incremental_train(data_manager)
        logging.info(
            "Task {} wall-clock training time: {:.2f}s".format(
                task, train_phase.seconds
            )
        )
        with PhaseTimer() as eval_phase:
            eval_result = model.eval_task()
        eval_wall_seconds = eval_phase.seconds
        cnn_accy, nme_accy = eval_result[0], eval_result[1]
        eval_metrics = eval_result[2] if len(eval_result) > 2 else None
        infer_flops = inference_flops_per_sample(model, model.test_loader.dataset)
        metrics_log.log_task(
            task,
            train_seconds=train_phase.seconds,
            train_peak_mem_mb=train_phase.peak_mb,
            eval_seconds=eval_phase.seconds,
            eval_peak_mem_mb=eval_phase.peak_mb,
            eval_ms_per_sample=(
                1000.0 * eval_phase.seconds / max(len(model.test_loader.dataset), 1)
            ),
            inference_flops_per_sample=infer_flops,
            total_params=(
                count_parameters(model._network)
                + getattr(model, "ridge_extra_param_count", lambda: 0)()
            ),
            trainable_params=(
                count_parameters(model._network, True)
                + getattr(model, "expert_head_trainable_param_count", lambda: 0)()
            ),
            train_samples=len(model.train_loader.dataset),
            train_flops_measured=train_flops_measured,
            profiled_epochs=(
                args.get("epochs") if args.get("profile_train_flops", False) else None
            ),
            checkpoint_mb=dir_size_mb(model._ckpt_dir()),
            timing_breakdown=model._latest_task_timing,
            routing_flops=(eval_metrics or {}).get("routing_flops"),
            cnn_top1=cnn_accy["top1"],
        )
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
            if eval_metrics is not None:
                _log_eval_metrics(task, eval_metrics, eval_wall_seconds)
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            if eval_metrics is not None:
                _log_eval_metrics(task, eval_metrics, eval_wall_seconds)

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _log_eval_metrics(task, eval_metrics, eval_wall_seconds):
    logging.info("Task {} eval time: {:.2f}s".format(task, eval_wall_seconds))

    if "eval_seconds" in eval_metrics:
        logging.info(
            "Eval time: {:.2f}s ({:.3f} ms/sample)".format(
                eval_metrics["eval_seconds"],
                eval_metrics.get("ms_per_sample", 0.0),
            )
        )

    routing_flops = eval_metrics.get("routing_flops")
    if routing_flops is not None:
        logging.info(
            "Routing FLOPs => per_sample: {}, per_batch@{}: {}, num_tasks: {}".format(
                routing_flops["per_sample"],
                routing_flops["batch_size"],
                routing_flops["per_batch"],
                routing_flops["num_tasks"],
            )
        )
