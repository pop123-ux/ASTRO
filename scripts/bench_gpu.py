#!/usr/bin/env python3
"""Run the optimizer benchmark on real data and a real backbone, on a GPU.

    python scripts/bench_gpu.py --task finetune-convnext --trials 16 --seeds 3

The CPU suite in ``astro.bench`` establishes that the implementations are
correct and measures them on models of 10^4--10^5 parameters. It cannot say
anything about the scales the literature makes its claims at, and optimizer
rankings are known to change with scale -- that is the central finding of Wen et
al. (arXiv:2509.02046). This script runs the *same protocol* (Algorithm 4 in
docs/paper/paper.md: equal tuning budgets, multiple seeds, paired statistics) on
tasks large enough to be informative.

Two tasks are provided:

``finetune-convnext``
    Fine-tune an ImageNet-pretrained ``convnext_tiny`` from ``timm`` on a small
    labelled subset. This is the regime this repository actually trains in, and
    the one where Qu et al. (arXiv:2605.10468) report matrix optimizers losing
    to Adam.

``scratch-convnext``
    The same backbone, randomly initialised. This is where spectral methods are
    expected to win, and it is included so that a win in one regime is not
    mistaken for a win in both.

On Kaggle, run it after ``notebooks/kaggle_run.py --stage discover`` so the
competition data is cached, and pass ``--data`` to point at the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.protocol import (  # noqa: E402
    OptimizerFactory,
    TaskResult,
    evaluate,
    paired_comparison,
    tune,
)
from astro.bench.registry import build_ablation_spaces, build_spaces  # noqa: E402
from astro.bench.run import format_report  # noqa: E402


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device found. This script is for the GPU numbers; "
            "use `python -m astro.bench.run` for the CPU suite."
        )
    return torch.device("cuda")


def _backbone(classes: int, pretrained: bool) -> nn.Module:
    """ConvNeXt-Tiny, matching this repository's baseline encoder."""
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("this script needs `pip install timm`") from exc
    return timm.create_model(
        "convnext_tiny", pretrained=pretrained, num_classes=classes, drop_path_rate=0.1
    )


def _loaders(data_root: Path | None, image_size: int, train_n: int, batch: int):
    """Small labelled image dataset.

    Falls back to a deterministic synthetic set when ``--data`` is not given, so
    the script is runnable for a smoke test without a download. A synthetic
    fallback is *not* a substitute for the real measurement, and the report says
    so explicitly when it is used.
    """
    from torch.utils.data import DataLoader, TensorDataset

    if data_root is not None and data_root.exists():
        from torchvision import datasets, transforms

        tfm = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        full = datasets.ImageFolder(str(data_root), transform=tfm)
        train, valid = torch.utils.data.random_split(
            full,
            [train_n, len(full) - train_n],
            generator=torch.Generator().manual_seed(0),
        )
        classes = len(full.classes)
        return (
            DataLoader(train, batch_size=batch, shuffle=True, num_workers=4, drop_last=True),
            DataLoader(valid, batch_size=batch * 2, num_workers=4),
            classes,
            True,
        )

    generator = torch.Generator().manual_seed(0)
    classes = 10
    images = torch.randn(train_n + 512, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, classes, (train_n + 512,), generator=generator)
    images += F.one_hot(labels, classes).float()[:, :, None, None].repeat(
        1, 1, image_size, image_size
    )[:, :3] * 0.6
    dataset = TensorDataset(images, labels)
    train, valid = torch.utils.data.random_split(
        dataset, [train_n, 512], generator=torch.Generator().manual_seed(0)
    )
    return (
        DataLoader(train, batch_size=batch, shuffle=True, drop_last=True),
        DataLoader(valid, batch_size=batch * 2),
        classes,
        False,
    )


def make_task(*, pretrained: bool, data_root: Path | None, image_size: int, train_n: int,
              batch: int, epochs: int):
    """Build a benchmark task closure with the shared signature."""
    device = _device()
    train_loader, valid_loader, classes, real = _loaders(data_root, image_size, train_n, batch)

    def task(factory: OptimizerFactory, seed: int) -> TaskResult:
        torch.manual_seed(seed)
        model = _backbone(classes, pretrained).to(device)
        optimizer = factory(model)
        scaler = torch.amp.GradScaler("cuda")
        curve = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        steps = 0
        for _ in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                steps += 1
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        total += float(F.cross_entropy(model(images), labels)) * len(labels)
                    count += len(labels)
            curve.append(total / max(1, count))
        end.record()
        torch.cuda.synchronize()
        return TaskResult(
            final=curve[-1], curve=curve, steps=steps, seconds=start.elapsed_time(end) / 1000.0
        )

    return task, real


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="finetune-convnext",
                        choices=["finetune-convnext", "scratch-convnext"])
    parser.add_argument("--data", type=Path, default=None,
                        help="ImageFolder root. Omitted => synthetic smoke data.")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-n", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/bench_gpu"))
    args = parser.parse_args(argv)

    pretrained = args.task == "finetune-convnext"
    task, real_data = make_task(
        pretrained=pretrained, data_root=args.data, image_size=args.image_size,
        train_n=args.train_n, batch=args.batch, epochs=args.epochs,
    )
    spaces = (
        build_ablation_spaces(finetuning=pretrained)
        if args.ablation
        else build_spaces(finetuning=pretrained)
    )

    records = tune(task, spaces, trials=args.trials, seed=0)
    summaries = {
        space.name: evaluate(
            task, space, records[space.name].best_config, seeds=range(100, 100 + args.seeds)
        )
        for space in spaces
    }
    control = "astro_full" if args.ablation else "adamw"
    control = control if control in summaries else next(iter(summaries))

    result = {
        "task": args.task,
        "trials": args.trials,
        "seeds": args.seeds,
        "control": control,
        "tuning": {n: r.__dict__ for n, r in records.items()},
        "evaluation": {n: s.__dict__ for n, s in summaries.items()},
        "comparisons": {
            n: paired_comparison(s, summaries[control])
            for n, s in summaries.items()
            if n != control
        },
        "_summaries": summaries,
    }

    report = format_report(result)
    if not real_data:
        report += (
            "\n\n> **Synthetic data.** `--data` was not supplied, so this ran on generated "
            "images. Treat it as a smoke test of the harness, not as a measurement."
        )
    print(report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}.md").write_text(report)
    (args.out / f"{args.task}.json").write_text(
        json.dumps(
            {k: v for k, v in result.items() if not k.startswith("_")}, indent=2, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
