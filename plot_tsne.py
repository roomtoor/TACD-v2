# plot_tsne.py
from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib.pyplot as plt


def load_names(path: str):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def pca_reduce(X, dim=50, seed=0):
    X = X.astype(np.float32)
    X = X - X.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    dim = min(dim, U.shape[1])
    Z = (U[:, :dim] * S[:dim]).astype(np.float32)
    return Z


def tsne_embed(X, seed=0, perplexity=30):
    from sklearn.manifold import TSNE
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1500,
        verbose=1
    ).fit_transform(X)


def _resolve_labels(data, ids, args):
    ids = ids.astype(np.int64)
    if args.color == "domain":
        uniq = np.unique(ids)
        labels = [f"{args.domain_prefix}{int(i)}" for i in uniq]
    else:
        if args.label_from_keep_ids and ("keep_ids" in data):
            uniq = np.array(data["keep_ids"], dtype=np.int64)
        else:
            uniq = np.unique(ids)

        if args.names:
            class_names = load_names(args.names)
            labels = [class_names[i] if i < len(class_names) else str(int(i)) for i in uniq]
        else:
            labels = [str(int(i)) for i in uniq]

    uniq_list = uniq.tolist()
    id2k = {int(u): k for k, u in enumerate(uniq_list)}
    cind = np.array([id2k[int(v)] for v in ids], dtype=np.int64)
    return uniq, labels, cind


def _square_limits(emb_list, pad=0.06):
    # 用所有(一张或两张) embedding 的联合范围，强制成正方形坐标系
    X = np.concatenate([e[:, 0] for e in emb_list], axis=0)
    Y = np.concatenate([e[:, 1] for e in emb_list], axis=0)
    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())

    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    # pad
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy

    # square: 以中心为基准，取更大的边长
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    return (cx - half, cx + half, cy - half, cy + half)


def _set_square_axes(ax, lim):
    xmin, xmax, ymin, ymax = lim
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", "box")
    # 让每个子图的“盒子”也尽量是正方形（matplotlib>=3.3）
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])


def _annotate_with_repulsion(ax, emb, y, uniq, labels, lim,
                             min_dist_frac=0.08, iters=250):
    """
    简单但好用：先在簇中心放标签，然后做“排斥迭代”，避免互相重叠+不出边界
    在数据坐标系做（不依赖额外库）。
    """
    xmin, xmax, ymin, ymax = lim
    dx = xmax - xmin
    dy = ymax - ymin
    min_d = min_dist_frac * max(dx, dy)

    # 初始位置：簇中心 + 一个小偏移（避免箭头为0）
    centers = []
    names = []
    for u, name in zip(uniq.tolist(), labels):
        m = (y == u)
        if int(m.sum()) < 5:
            continue
        cx, cy = emb[m].mean(axis=0)
        centers.append([cx, cy])
        names.append(name)

    if not centers:
        return

    P = np.array(centers, dtype=np.float32)
    # 初始偏移（沿着从整体中心向外的方向）
    global_c = P.mean(axis=0)
    v = P - global_c
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    v = v / n
    P = P + v * (0.04 * max(dx, dy))

    # 迭代排斥
    for _ in range(iters):
        moved = False
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                dvec = P[i] - P[j]
                dist = float(np.sqrt((dvec * dvec).sum()) + 1e-6)
                if dist < min_d:
                    push = (min_d - dist) * 0.35
                    step = (dvec / dist) * push
                    P[i] += step
                    P[j] -= step
                    moved = True

        # 边界夹紧（留一点 margin）
        mx = 0.03 * dx
        my = 0.03 * dy
        P[:, 0] = np.clip(P[:, 0], xmin + mx, xmax - mx)
        P[:, 1] = np.clip(P[:, 1], ymin + my, ymax - my)

        if not moved:
            break

    # 画标签
    for (tx, ty), (cx, cy), name in zip(P, centers, names):
        ax.annotate(
            name,
            xy=(cx, cy),
            xytext=(float(tx), float(ty)),
            textcoords="data",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1.2),
            arrowprops=dict(arrowstyle="-", lw=1.0),
        )


def _plot_one(ax, data, emb, args, lim, title=None):
    y = data["y"].astype(np.int64)
    d = data["d"].astype(np.int64)

    ids = d if args.color == "domain" else y
    uniq, labels, cind = _resolve_labels(data, ids, args)

    ax.scatter(
        emb[:, 0], emb[:, 1],
        c=cind,
        s=args.point_size,
        alpha=args.alpha,
        cmap=args.cmap,
        linewidths=0
    )

    _set_square_axes(ax, lim)
    if title:
        ax.set_title(title, fontsize=14)

    if args.annotate and args.color == "class":
        _annotate_with_repulsion(
            ax=ax,
            emb=emb,
            y=y,
            uniq=uniq,
            labels=labels,
            lim=lim,
            min_dist_frac=args.label_sep,
            iters=args.label_iters
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--input2", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--color", type=str, choices=["class", "domain"], default="class")
    ap.add_argument("--names", type=str, default=None)
    ap.add_argument("--cmap", type=str, default="Set1")
    ap.add_argument("--point_size", type=float, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)  # 默认不透明
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--pca_dim", type=int, default=50)

    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--label_from_keep_ids", action="store_true")

    ap.add_argument("--title1", type=str, default=None)
    ap.add_argument("--title2", type=str, default=None)

    # 关键：强制正方形子图，所以双图时用 2:1 的画布（每个子图正方形）
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--label_sep", type=float, default=0.10, help="越大标签越不容易重叠(但更飘)")
    ap.add_argument("--label_iters", type=int, default=250)
    ap.add_argument("--domain_prefix", type=str, default="Domain ")

    args = ap.parse_args()

    def _load_and_embed(path: str):
        data = np.load(path, allow_pickle=True)
        feat = data["feat"]
        print(f"[Load] {path} feat={feat.shape}, y={data['y'].shape}, d={data['d'].shape}")
        Z = pca_reduce(feat, dim=args.pca_dim, seed=args.seed)
        emb = tsne_embed(Z, seed=args.seed, perplexity=args.perplexity)
        return data, emb

    data1, emb1 = _load_and_embed(args.input)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.input2:
        data2, emb2 = _load_and_embed(args.input2)
        lim = _square_limits([emb1, emb2], pad=args.pad)

        # 双图：每个子图正方形 -> 画布用 (12,6) 这种 2:1
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=args.dpi, constrained_layout=True)
        _plot_one(axes[0], data1, emb1, args, lim, title=args.title1)
        _plot_one(axes[1], data2, emb2, args, lim, title=args.title2)
    else:
        lim = _square_limits([emb1], pad=args.pad)
        fig = plt.figure(figsize=(6, 6), dpi=args.dpi)
        ax = plt.gca()
        _plot_one(ax, data1, emb1, args, lim, title=args.title1)

    plt.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
