import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: no progress bar (keeps script runnable even if tqdm isn't installed)
    def tqdm(it, **kwargs):  # type: ignore
        return it


@dataclass
class SplitStats:
    count: int
    min_points: Optional[int]
    max_points: Optional[int]
    le_10: int
    le_20: int
    le_30: int


def _point_count(x: torch.Tensor) -> int:
    # x: (N, 5)
    return int(x.shape[0])


def split_point_stats(ds, desc: str) -> SplitStats:
    n = len(ds)
    if n == 0:
        return SplitStats(count=0, min_points=None, max_points=None, le_10=0, le_20=0, le_30=0)

    min_p: Optional[int] = None
    max_p: Optional[int] = None
    le_10 = 0
    le_20 = 0
    le_30 = 0

    # Force tqdm on even when stdout isn't detected as a TTY (common in some IDE terminals / piping).
    for i in tqdm(
        range(n),
        desc=desc,
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        disable=False,
    ):
        x, _ = ds[i]
        p = _point_count(x)
        min_p = p if min_p is None else min(min_p, p)
        max_p = p if max_p is None else max(max_p, p)
        if p <= 10:
            le_10 += 1
        if p <= 20:
            le_20 += 1
        if p <= 30:
            le_30 += 1

    return SplitStats(count=n, min_points=min_p, max_points=max_p, le_10=le_10, le_20=le_20, le_30=le_30)


def _fmt_count_pct(cnt: int, total: int) -> str:
    if total <= 0:
        return f"{cnt} (n/a)"
    return f"{cnt} ({(cnt / total) * 100:.2f}%)"


def print_dataset_point_stats(datasets: Dict[str, Tuple[object, object]]) -> None:
    """
    datasets: dict[name] = (train_dataset, test_dataset)
    Each dataset item must return (feature, label) where feature is a tensor shaped (N, 5).
    """
    rows: List[Tuple[str, SplitStats, SplitStats]] = []
    for name, (train_ds, test_ds) in datasets.items():
        tr = split_point_stats(train_ds, f"{name}/train")
        te = split_point_stats(test_ds, f"{name}/test")
        rows.append((name, tr, te))

    print("\nStats (point min/max computed on full scan per split):")
    for name, tr, te in rows:
        print(
            f"- {name:6s} | train n={tr.count:6d} points(min/max)={tr.min_points}/{tr.max_points} "
            f"| test n={te.count:6d} points(min/max)={te.min_points}/{te.max_points}"
        )
        print(
            "         | "
            f"train <=10/<=20/<=30 = {_fmt_count_pct(tr.le_10, tr.count)} / {_fmt_count_pct(tr.le_20, tr.count)} / {_fmt_count_pct(tr.le_30, tr.count)} "
            f"| test <=10/<=20/<=30 = {_fmt_count_pct(te.le_10, te.count)} / {_fmt_count_pct(te.le_20, te.count)} / {_fmt_count_pct(te.le_30, te.count)}"
        )