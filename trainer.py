import sys
import logging
import copy
import torch
import time
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
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
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

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
    gate_curve = {"top1": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        task_wall_start = time.perf_counter()
        model.incremental_train(data_manager)
        task_train_only_seconds = time.perf_counter() - task_wall_start
        logging.info(
            "Task {} wall-clock training time: {:.2f}s".format(
                task, task_train_only_seconds
            )
        )
        eval_wall_start = time.perf_counter()
        eval_result = model.eval_task()
        eval_wall_seconds = time.perf_counter() - eval_wall_start
        cnn_accy, nme_accy = eval_result[0], eval_result[1]
        gate_accy = eval_result[2] if len(eval_result) > 2 else None
        routing_comparison = (
            model.eval_routing_comparison()
            if hasattr(model, "eval_routing_comparison")
            else None
        )
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            if routing_comparison is not None:
                logging.info(
                    "CNN (ENTROPY ROUTING): {}".format(
                        routing_comparison["entropy"]["grouped"]
                    )
                )
                logging.info(
                    "CNN (GATE ROUTING): {}".format(
                        routing_comparison["gate"]["grouped"]
                    )
                )
                logging.info(
                    "CNN (ORACLE ROUTING): {}".format(
                        routing_comparison["oracle"]["grouped"]
                    )
                )
                entropy_top1 = routing_comparison["entropy"]["top1"]
                gate_top1 = routing_comparison["gate"]["top1"]
                oracle_top1 = routing_comparison["oracle"]["top1"]
                logging.info("=" * 70)
                logging.info("FINAL CNN COMPARISON after Task {}".format(task))
                logging.info("Entropy routing top1: {:.2f}".format(entropy_top1))
                logging.info("Gate routing top1   : {:.2f}".format(gate_top1))
                logging.info("Oracle routing top1 : {:.2f}".format(oracle_top1))
                logging.info("Delta (gate-entropy): {:+.2f}".format(gate_top1 - entropy_top1))
                logging.info("Delta (oracle-gate) : {:+.2f}".format(oracle_top1 - gate_top1))
                logging.info("=" * 70)
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
            if gate_accy is not None:
                gate_curve["top1"].append(gate_accy["top1"])
                logging.info("Gate top1 curve: {}".format(gate_curve["top1"]))
                logging.info("Average Accuracy (Gate): {}".format(sum(gate_curve["top1"]) / len(gate_curve["top1"])))
                _log_gate_metrics(task, gate_accy, eval_wall_seconds)
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            if routing_comparison is not None:
                logging.info(
                    "CNN (ENTROPY ROUTING): {}".format(
                        routing_comparison["entropy"]["grouped"]
                    )
                )
                logging.info(
                    "CNN (GATE ROUTING): {}".format(
                        routing_comparison["gate"]["grouped"]
                    )
                )
                logging.info(
                    "CNN (ORACLE ROUTING): {}".format(
                        routing_comparison["oracle"]["grouped"]
                    )
                )
                entropy_top1 = routing_comparison["entropy"]["top1"]
                gate_top1 = routing_comparison["gate"]["top1"]
                oracle_top1 = routing_comparison["oracle"]["top1"]
                logging.info("=" * 70)
                logging.info("FINAL CNN COMPARISON after Task {}".format(task))
                logging.info("Entropy routing top1: {:.2f}".format(entropy_top1))
                logging.info("Gate routing top1   : {:.2f}".format(gate_top1))
                logging.info("Oracle routing top1 : {:.2f}".format(oracle_top1))
                logging.info("Delta (gate-entropy): {:+.2f}".format(gate_top1 - entropy_top1))
                logging.info("Delta (oracle-gate) : {:+.2f}".format(oracle_top1 - gate_top1))
                logging.info("=" * 70)

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            if gate_accy is not None:
                gate_curve["top1"].append(gate_accy["top1"])
                logging.info("Gate top1 curve: {}".format(gate_curve["top1"]))
                logging.info("Average Accuracy (Gate): {}".format(sum(gate_curve["top1"]) / len(gate_curve["top1"])))
                _log_gate_metrics(task, gate_accy, eval_wall_seconds)

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


def _log_gate_metrics(task, gate_accy, eval_wall_seconds):
    logging.info("Task {} eval time: {:.2f}s".format(task, eval_wall_seconds))

    if "eval_seconds" in gate_accy:
        logging.info(
            "Gate routing eval time: {:.2f}s ({:.3f} ms/sample)".format(
                gate_accy["eval_seconds"],
                gate_accy.get("ms_per_sample", 0.0),
            )
        )

    routing_flops = gate_accy.get("routing_flops")
    if routing_flops is not None:
        logging.info(
            "Gate routing FLOPs => per_sample: {}, per_batch@{}: {}, num_tasks: {}".format(
                routing_flops["per_sample"],
                routing_flops["batch_size"],
                routing_flops["per_batch"],
                routing_flops["num_tasks"],
            )
        )

    per_task = gate_accy.get("per_task")
    if per_task:
        logging.info("Gate routing task-by-task after Task {}".format(task))
        for task_idx, stats in per_task.items():
            logging.info(
                "  True Task {} => correct {}/{} ({:.2f}%), predicted_as_task {} times".format(
                    task_idx,
                    stats["correct"],
                    stats["total"],
                    stats["accuracy"],
                    stats["predicted"],
                )
            )
