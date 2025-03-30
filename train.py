#import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

import random

import os
from tqdm import tqdm
import time
import argparse

from data import get_arithmetic_dataset
from lstm import LSTMLM
from gpt import GPT
from trainer import train as train_model
from checkpointing import get_all_checkpoints_per_trials
from plotter import plot_loss_accs

########################################################################################
########################################################################################

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################################################################
#     Utility functions library
########################################################################################
def convert_tensors(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()  # detach() 避免梯度问题
    elif isinstance(data, list):
        return [convert_tensors(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_tensors(v) for k, v in data.items()}
    else:
        return data
    
def load_trial_metrics(checkpoint_dir: str, exp_name: str) -> dict:
    """
    载入单次实验的训练统计信息 (all_metrics),
    即 trainer.py 里保存的 f"{checkpoint_dir}/{exp_name}.pth" 文件。
    """
    path = os.path.join(checkpoint_dir, f"{exp_name}.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file found: {path}")
    data = torch.load(path, map_location="cpu")
    return data

def mean_std_str(values):
    """将一组数字计算均值±标准差，并格式化输出"""
    arr = np.array(values, dtype=np.float32)
    return f"{arr.mean():.6f} ± {arr.std():.6f}"


def get_extrema_performance_steps(all_metrics: dict):
    """
    返回一组最值，包括:
      - Ltrain       (训练集最小loss)
      - tf(Ltrain)   (对应的训练步数)
      - Ltest        (测试/验证集最小loss)
      - tf(Ltest)    (对应的训练步数)
      - Atrain       (训练集最大accuracy)
      - tf(Atrain)   (对应的训练步数)
      - Atest        (测试/验证集最大accuracy)
      - tf(Atest)    (对应的训练步数)

    如果找不到对应数据(列表为空), 该指标将使用默认值 (inf 或 0.0), 对应step记为 -1。

    all_metrics 里应当包含:
      all_metrics["train"]["loss"]       -> list of floats
      all_metrics["train"]["accuracy"]   -> list of floats
      all_metrics["test"]["loss"]        -> list of floats
      all_metrics["test"]["accuracy"]    -> list of floats
      all_metrics["all_steps"]           -> list of ints (同样长度)
    """

    # 1) 取出曲线
    train_loss = all_metrics["train"]["loss"]       # list
    train_acc  = all_metrics["train"]["accuracy"]   # list
    test_loss  = all_metrics["test"]["loss"]        # list
    test_acc   = all_metrics["test"]["accuracy"]    # list
    steps      = all_metrics["all_steps"]           # list

    # 2) 找训练loss最小
    Ltrain = float("inf")
    tf_Ltrain = -1
    if len(train_loss) > 0:
        Ltrain = min(train_loss)
        i_Ltrain = train_loss.index(Ltrain)  # 最小值所在索引
        tf_Ltrain = steps[i_Ltrain]          # 对应 step

    # 3) 找测试loss最小
    Ltest = float("inf")
    tf_Ltest = -1
    if len(test_loss) > 0:
        Ltest = min(test_loss)
        i_Ltest = test_loss.index(Ltest)
        tf_Ltest = steps[i_Ltest]

    # 4) 找训练acc最大
    Atrain = 0.0
    tf_Atrain = -1
    if len(train_acc) > 0:
        Atrain = max(train_acc)
        i_Atrain = train_acc.index(Atrain)
        tf_Atrain = steps[i_Atrain]

    # 5) 找测试acc最大
    Atest = 0.0
    tf_Atest = -1
    if len(test_acc) > 0:
        Atest = max(test_acc)
        i_Atest = test_acc.index(Atest)
        tf_Atest = steps[i_Atest]

    # 6) 返回结果
    return {
        "Ltrain": Ltrain,        "tf(Ltrain)": tf_Ltrain,
        "Ltest":  Ltest,         "tf(Ltest)": tf_Ltest,
        "Atrain": Atrain,        "tf(Atrain)": tf_Atrain,
        "Atest":  Atest,         "tf(Atest)": tf_Atest,
    }


########################################################################################
########################################################################################

class DummyScheduler:
    """
    Dummy LR Scheduler that supports standard methods like state_dict, load_state_dict, etc.,
    but does nothing to the optimizer or learning rates.
    """
    def __init__(self, optimizer, *args, **kwargs):
        """
        Initialize the DummyScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer (required to match the API, not used).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optimizer = optimizer
        self._state = {}

    def step(self, *args, **kwargs):
        """
        Dummy step function that does nothing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.

        Returns:
            dict: A dictionary representing the scheduler's state.
        """
        return self._state

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state from a dictionary.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self._state.update(state_dict)

    def get_last_lr(self):
        """
        Get the last computed learning rate(s).

        Returns:
            list: A list of the last learning rates.
        """
        return [group['lr'] for group in self.optimizer.param_groups]


########################################################################################
########################################################################################

def train(args):
    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    # Create a directory to save the experiment results
    checkpoint_path = os.path.join(args.log_dir, str(args.exp_id))
    i=0
    while os.path.exists(checkpoint_path):
        i+=1
        checkpoint_path = os.path.join(args.log_dir, str(i))
    os.makedirs(checkpoint_path, exist_ok=True)

    ## Print parameters
    if args.verbose :
        print("=="*60)
        for k, v in vars(args).items() :
            print(k, ":", v)
        print("=="*60)

    # Data
    (train_dataset, valid_dataset), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(
        args.p, args.p, args.operator, args.r_train, args.operation_orders, is_symmetric=False, shuffle=True, seed=args.seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=min(args.train_batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
    )

    train_dataloader_for_eval = DataLoader(
        train_dataset,
        batch_size=min(args.eval_batch_size, len(train_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=min(args.eval_batch_size, len(valid_dataset)),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Model
    vocabulary_size = len(tokenizer)
    if args.model == "lstm":
        model = LSTMLM(
            vocabulary_size = vocabulary_size, 
            embedding_size = args.embedding_size, 
            hidden_size = args.hidden_size, 
            num_layers = args.num_layers, 
            dropout = args.dropout,
            padding_index = padding_index,
            bias_lstm = True,
            bias_classifier = args.bias_classifier,
            share_embeddings = args.share_embeddings
        )
    elif args.model == "gpt":
        model = GPT(
            num_heads = args.num_heads, 
            num_layers = args.num_layers,
            embedding_size = args.embedding_size,
            vocabulary_size = vocabulary_size,
            sequence_length = MAX_LENGTH,
            multiplier = 4,
            dropout = args.dropout,
            non_linearity = "gelu",
            padding_index = padding_index,
            bias_attention = True,
            bias_classifier = args.bias_classifier,
            share_embeddings = args.share_embeddings
        )
    else:
        raise ValueError("Unknown model {0}".format(args.model))

    #print(model)
    model = model.to(args.device)

    if args.verbose : 
        print("Model :", model, "\n")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model trainable parameters : {n_params}")

    # Optimizer
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # ==========================
    # TODO: Write your code here
    # ==========================
    # Learning rate scheduler
    scheduler = DummyScheduler(optimizer) # Dummy scheduler that does nothing
    # ==========================
    # ==========================

    # Train    
    all_metrics = train_model(
        model, train_dataloader, train_dataloader_for_eval, valid_dataloader, optimizer, scheduler,
        args.device, 
        args.exp_name, checkpoint_path, 
        n_steps=args.n_steps,
        eval_first=args.eval_first,
        eval_period=args.eval_period,
        print_step=args.print_step,
        save_model_step=args.save_model_step,
        save_statistic_step=args.save_statistic_step,
        verbose=args.verbose
    )
    
    all_metrics = convert_tensors(all_metrics)
    # Plot
    plot_loss_accs(
        all_metrics, multiple_runs=False, log_x=False, log_y=False,
        fileName=args.exp_name, filePath=checkpoint_path, show=False)

    return all_metrics, checkpoint_path


########################################################################################
########################################################################################

def train_m_models(args, M: int = None, seeds: list = None):
    """
    1) 按照给定的 seeds 或者 M 次数，跑多次实验 (train(args))；
    2) 对每次实验的统计信息，提取最低/最高指标并打印；
    3) 将所有实验合并到 all_metrics 后，用 plot_loss_accs(..., multiple_runs=True) 做多次实验的平均/方差可视化；
    4) 最后对多种子下的极值指标做均值±标准差并打印。
    """
    from copy import deepcopy

    # 为了不修改原 args，我们先复制一下
    base_args = deepcopy(args)

    # 如果没指定 seeds，就用 [base_args.seed, base_args.seed+1, ..., base_args.seed+(M-1)]
    assert M is not None or seeds is not None, "Either M or seeds should be provided."
    if seeds is not None:
        M = len(seeds)
    else:
        seeds = [base_args.seed + i for i in range(M)]

    # 存放所有实验的 checkpoint 文件夹
    all_checkpoint_paths = []

    # 为了最终能在多种子之间做均值±std，我们收集想要的指标
    # 这里示例收集 8 个：Ltrain, Lval, Atrain, Aval 及对应step
    metric_keys = [
        "Ltrain", "tf(Ltrain)",
        "Ltest",   "tf(Ltest)",
        "Atrain", "tf(Atrain)",
        "Atest",   "tf(Atest)"
    ]
    # 这里用一个 dict of lists 来存每个种子的指标
    results_across_seeds = {k: [] for k in metric_keys}

    ############################################################################
    # 1) 逐个 seed 跑实验，并立即打印该 seed 的最小/最大指标
    ############################################################################
    for idx, s in enumerate(seeds):
        print(f"\n=== Train model {idx+1}/{M}, seed={s} ===")

        base_args.exp_id = idx
        base_args.seed = s

        # 调用你已有的 train(...) 函数，返回单次实验的 all_metrics, checkpoint_path
        all_metrics_single, checkpoint_path = train(base_args)
        all_checkpoint_paths.append(checkpoint_path)

        # 载入 <exp_name>.pth 文件，提取最低/最高指标
        trial_metrics = load_trial_metrics(checkpoint_path, base_args.exp_name)
        ext = get_extrema_performance_steps(trial_metrics)
        # ext 里一般包含: 
        # "Ltrain", "tf(Ltrain)", "Lval", "tf(Lval)",
        # "Atrain", "tf(Atrain)", "Aval", "tf(Aval)"

        # 逐个放到 results_across_seeds
        for k in metric_keys:
            if k in ext:
                results_across_seeds[k].append(ext[k])
            else:
                # 如果没有该键，可能是因为 get_extrema_performance_steps 没写对应逻辑
                # 可视情况而定
                results_across_seeds[k].append(np.nan)
        
        # # 立刻打印本次种子的极值情况
        # # 下面只是简单演示你最关心的4个值
        # Ltrain_min = ext.get("Ltrain", float('inf'))
        # Lval_min   = ext.get("Lval",   float('inf'))
        # Atrain_max = ext.get("Atrain", 0.0)
        # Aval_max   = ext.get("Aval",   0.0)

        # print(f"Seed={s}: Ltrain(min)={Ltrain_min:.4f}, Lval(min)={Lval_min:.4f}, "
        #       f"Atrain(max)={Atrain_max:.4f}, Aval(max)={Aval_max:.4f}")
    
    ############################################################################
    # 2) 多次实验合并 => 画对比/平均曲线
    ############################################################################
    # 这一步会将所有 checkpoint 路径下的 exp_name.pth 文件合并为 all_metrics_multi
    all_models_per_trials, all_metrics_multi = get_all_checkpoints_per_trials(
        all_checkpoint_paths,
        base_args.exp_name,
        just_files=True,
        verbose=base_args.verbose
    )
    
    all_metrics_multi = convert_tensors(all_metrics_multi)

    # 画图
    plot_loss_accs(
        all_metrics_multi,
        multiple_runs=True,             # 多次实验
        log_x=False,
        log_y=False,
        fileName=f'{base_args.exp_name}_M={M}',
        filePath=base_args.log_dir,
        show=False
    )

    ############################################################################
    # 3) 对多种子的极值指标做均值±标准差并打印
    ############################################################################
    print(f"\n=== Summary of extrema performance across seeds {args.model}===")
    for k in metric_keys:
        vals = results_across_seeds[k]
        # 如果全是 nan 就跳过
        if all(np.isnan(vals)):
            continue
        # 打印均值±std
        msg = mean_std_str(vals)
        print(f"{k}: {msg}")

    return all_models_per_trials, all_metrics_multi, all_checkpoint_paths


########################################################################################
########################################################################################

class Arguments:
    # Data
    p: int = 31 # Prime number
    operator : str = "+" # ["+", "-", "*", "/"]
    r_train : float = .5
    operation_orders : int = 2 # 2, 3 or [2, 3]
    train_batch_size: int = 512
    eval_batch_size: int = 2**12
    num_workers: int = 0

    # Model
    model: str = 'lstm' # [lstm, gpt]
    num_heads: int = 4
    num_layers: int = 2
    embedding_size: int = 2**7
    hidden_size: int = 2**7
    dropout : float = 0.0
    share_embeddings : bool = False
    bias_classifier : bool = True

    # Optimization
    optimizer: str = 'adamw'  # [sgd, momentum, adam, adamw]
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-0

    # Training
    n_steps: int = 10**4 * 1 + 1
    eval_first: int = 10**2 * 1
    eval_period: int = 10**2 * 1
    print_step: int = 10**2 * 1
    save_model_step: int = 10**3
    save_statistic_step: int = 10**3

    # Experiment & Miscellaneous
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id: int = 0
    exp_name: str = "4.3_LSTM"
    log_dir: str = './log'
    seed: int = 42    
    verbose: bool = True

########################################################################################
########################################################################################

if __name__ == "__main__":
    args = Arguments()
    print("=="*60)
    #all_metrics, checkpoint_path = train(args)

    args.n_steps = 10**3 * 1 + 1
    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0,42])
    print("=="*60)
    print("Experiment finished.")