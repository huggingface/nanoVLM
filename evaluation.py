# Evaluation script for lmms-eval, taken from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/__main__.py

import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

import hashlib
import logging
from pathlib import Path
from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    handle_non_serializable,
    make_table,
    simple_parse_args_string,
)

from eval.lmms_eval_wrapper import NanoVLMWrapper


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        logging.warning(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=32,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default='results/',
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = '0'
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument("--process_with_media", action="store_true", help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.")
    args = parser.parse_args()
    return args


def evaluate(
    config: str = "",
    model: str = "hf",
    tasks: str = None,
    model_args: str = "",
    num_fewshot: int = None,
    batch_size: str = "32",
    max_batch_size: int = None,
    device: str = "cuda",
    output_path: str = "results/",
    limit: float = None,
    use_cache: str = None,
    cache_requests: str = None,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = False,
    wandb_log_samples: bool = False,
    log_samples_suffix: str = "model_outputs",
    system_instruction: str = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    show_config: bool = False,
    include_path: str = None,
    gen_kwargs: str = "",
    verbosity: str = "INFO",
    wandb_args: str = "",
    timezone: str = "Asia/Singapore",
    hf_hub_log_args: str = "",
    predict_only: bool = False,
    seed: Union[str, list] = "0,1234,1234,1234",
    trust_remote_code: bool = False,
    process_with_media: bool = False,
):
    """
    A function to run lmms-eval evaluation.

    :param model:
        Name of model e.g. `hf`
    :param model_args:
        String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`
    :param tasks:
        To get full list of tasks, use the command lmms-eval --tasks list
    :param num_fewshot:
        Number of examples in few-shot context
    :param batch_size:
        Batch size. Acceptable values are 'auto', 'auto:N' or N, where N is an integer.
    :param max_batch_size:
        Maximal batch size to try with --batch_size auto
    :param device:
        Device to use (e.g. cuda, cuda:0, cpu)
    :param output_path:
        The path to the output file where the result metrics will be saved.
    :param limit:
        Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.
    :param use_cache:
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests:
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param check_integrity:
        Whether to run the relevant part of the test suite for the tasks
    :param write_out:
        Prints the prompt for the first few documents.
    :param log_samples:
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param wandb_log_samples:
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases
    :param log_samples_suffix:
        Specify a suffix for the log_samples file name.
    :param system_instruction:
        System instruction to be used in the prompt
    :param apply_chat_template:
        If True, applies the chat template to the prompt
    :param fewshot_as_multiturn:
        If True, uses the fewshot as a multi-turn conversation
    :param show_config:
        If True, shows the the full config of all tasks at the end of the evaluation.
    :param include_path:
        Additional path to include if there are external tasks to include.
    :param gen_kwargs:
        String arguments for model generation on greedy_until tasks, e.g. `temperature=0,top_k=0,top_p=0`
    :param verbosity:
        Log error when tasks are not registered.
    :param wandb_args:
        Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval`
    :param timezone:
        Timezone for datetime string, e.g. Asia/Singapore, America/New_York.
    :param hf_hub_log_args:
        Comma separated string arguments passed to Hugging Face Hub's log function.
    :param predict_only:
        Use with `log_samples`. Only model outputs will be saved and metrics will not be evaluated.
    :param seed:
        Set seed for python's random, numpy, torch, and fewshot sampling.
    :param trust_remote_code:
        Sets trust_remote_code to True to execute code to create HF Datasets from the Hub.
    :param process_with_media:
        Whether you will process you dataset with audio, image.
    :return:
        A list of results for each evaluation run.
    """
    args_dict = locals()

    if isinstance(args_dict["seed"], str):
        default_seed_string = "0,1234,1234,1234"
        args_dict["seed"] = _int_or_none_list_arg_type(3, 4, default_seed_string, args_dict["seed"])

    args = argparse.Namespace(**args_dict)

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args
        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator only if not already in a distributed context
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        if accelerator.is_main_process:
            is_main_process = True
        else:
            is_main_process = False

    for item_args in args_list:
        try:
            results, samples = cli_evaluate_single(item_args)
            results_list.append(results)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and item_args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if item_args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        except Exception as e:
            if item_args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)

    for item_args, results in zip(args_list, results_list):
        if results is not None:
            print(f"{item_args.model} ({item_args.model_args}), gen_kwargs: ({item_args.gen_kwargs}), limit: {item_args.limit}, num_fewshot: {item_args.num_fewshot}, " f"batch_size: {item_args.batch_size}")
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()

    return results_list


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_eval_args()

    # Check if no arguments were passed after parsing
    if len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.                          │")
        print("│ `python evaluation.py  --model lusxvr/nanoVLM-450M --tasks mmstar`            │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    evaluate(**vars(args))


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # update the evaluation tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    eval_logger.info(f"Evaluation tracker args: {evaluation_tracker_args}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format(f"\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_with_num":
        log_message = (
            "\n" + "=" * 70 + "\n" + "\n\tYou are trying to check all the numbers in each task." + "\n\tThis action will download the complete dataset." + "\n\tIf the results are not clear initially, call this again." + "\n\n" + "=" * 70
        )
        eval_logger.info(log_message)
        for task_name in sorted(task_manager.list_all_tasks()):
            try:
                task_dict = get_task_dict([task_name], model_name="llava")
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                eval_logger.info(f"\nTask : {task_obj.config.task}\n - #num : {len(task_obj.test_docs()) if task_obj.has_test_docs() else len(task_obj.validation_docs())}")
            except Exception as e:
                eval_logger.debug(f"\nTask : {task_name} fail to load \n Exception : \n {e}")
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lmms-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lmms-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    wrapped_model = NanoVLMWrapper(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = evaluator.simple_evaluate(
        model=wrapped_model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend='torchrun' if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 'accelerate',
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()