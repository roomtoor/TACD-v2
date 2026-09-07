# utils/train_utils.py
import os
import torch
from datetime import datetime

# =============== 日志与保存 ===============
class Logger:
    def __init__(self, log_dir="./logs", exp_name="exp"):
        os.makedirs(log_dir, exist_ok=True)
        t = datetime.now().strftime("%m%d_%H%M")
        self.fpath = os.path.join(log_dir, f"{exp_name}_{t}.log")
        print(f"[Logger] Log file: {self.fpath}")
        with open(self.fpath, "w") as f:
            f.write(f"=== Log start: {datetime.now()} ===\n")

    def write(self, s: str, also_print=True):
        if also_print:
            print(s)
        with open(self.fpath, "a") as f:
            f.write(s + "\n")

# =============== Checkpoint ===============
def save_checkpoint(
    model,
    optimizer,
    epoch,
    ckpt_dir="./checkpoints",
    exp_name="exp",
    metadata=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{exp_name}_ep{epoch}.pth")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, save_path)
    print(f"[Checkpoint] Saved: {save_path}")

# =============== 学习率调度 ===============
def cosine_lr_schedule(optimizer, base_lr, cur_epoch, max_epoch):
    lr = 0.5 * base_lr * (1 + torch.cos(torch.tensor(cur_epoch / max_epoch * 3.1415926)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr.item()
