# utils/meter.py
class AverageMeter:
    """
    计算和存储平均值、当前值，用于loss/acc跟踪
    """
    def __init__(self, name, fmt=":6.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg:{avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
