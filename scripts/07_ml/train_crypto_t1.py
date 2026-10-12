#!/usr/bin/env python3
"""Train rw0710-style sequence models on crypto next-bar excess returns.

Example:
  python scripts/train_crypto_t1.py --model resnet --interval 30m --loss mse_ic_spread
  tensorboard --logdir runs/crypto_t1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efficient_cryptos.features.frame import add_alpha_features, load_feature_frame  # noqa: E402
from efficient_cryptos.ml.sections import (  # noqa: E402
    Section, build_sections, section_tensors, select_features,
)
from efficient_cryptos.ml.losses import (  # noqa: E402
    SUPPORTED_LOSSES, build_loss, dispersive_loss, quantile_contrastive_loss,
)
from efficient_cryptos.backtest.metrics import summarize_metrics  # noqa: E402
from efficient_cryptos.ml.models import SUPPORTED_MODELS, build_model  # noqa: E402
from efficient_cryptos.ml.lgbm import train_lgbm, predict_lgbm  # noqa: E402
from efficient_cryptos.common.storage import LocalStore  # noqa: E402


class EmbeddingCapture:
    def __init__(self, model: nn.Module):
        head = None
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.out_features == 1:
                head = module
        if head is None:
            raise ValueError("Model has no scalar Linear prediction head")
        self.tensor: torch.Tensor | None = None
        self.handle = head.register_forward_pre_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.tensor = inputs[0] if inputs else None

    def close(self) -> None:
        self.handle.remove()


def choose_device(requested: str) -> torch.device:
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            "use --device auto, cuda, mps, or cpu"
        )
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA GPU is available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but Apple GPU support is unavailable")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rank_local_symbols(data_dir: str, limit: int) -> list[str]:
    store = LocalStore(data_dir)
    scores = []
    for symbol in store.list_symbols():
        try:
            df = pd.read_parquet(store.kline_path(symbol), columns=["quote_volume"])
            scores.append((float(df.quote_volume.tail(1440).median()), symbol))
        except Exception:
            continue
    return [symbol for _, symbol in sorted(scores, reverse=True)[:limit]]


def split_sections(sections: list[Section], train_frac: float,
                   val_frac: float) -> tuple[list[Section], list[Section], list[Section]]:
    n = len(sections)
    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    if val_end >= n:
        val_end = n - 1
    train, val, test = sections[:train_end], sections[train_end:val_end], sections[val_end:]
    if not train or not val or not test:
        raise ValueError(f"Not enough timestamps for train/val/test split: {n}")
    return train, val, test


def split_sections_by_date(sections: list[Section], args: argparse.Namespace):
    """Explicit chronological split; all date bounds are inclusive UTC dates."""
    def timestamp(value: str, *, end: bool = False) -> pd.Timestamp:
        result = pd.Timestamp(value)
        result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
        return result + pd.Timedelta(days=1) if end else result
    train_start, train_end = timestamp(args.train_start), timestamp(args.train_end, end=True)
    val_start, val_end = timestamp(args.validation_start), timestamp(args.validation_end, end=True)
    test_start, test_end = timestamp(args.test_start), timestamp(args.test_end, end=True)
    train = [s for s in sections if train_start <= s.timestamp < train_end]
    val = [s for s in sections if val_start <= s.timestamp < val_end]
    test = [s for s in sections if test_start <= s.timestamp < test_end]
    if not train or not val or not test:
        raise ValueError(
            f"Empty explicit split: train={len(train)} validation={len(val)} test={len(test)}"
        )
    return train, val, test


def merge_factor_caches(frame: pd.DataFrame, paths: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if not paths:
        return frame, []
    parts = []
    for value in paths:
        path = Path(value)
        factor = pd.read_parquet(path)
        required = {"timestamp", "symbol"}
        if not required.issubset(factor.columns):
            raise ValueError(f"factor cache missing timestamp/symbol: {path}")
        factor["timestamp"] = pd.to_datetime(factor["timestamp"], utc=True)
        factor["symbol"] = factor["symbol"].astype(str).str.upper()
        parts.append(factor)
    joined = pd.concat(parts, ignore_index=True, sort=False)
    feature_columns = sorted(column for column in joined.columns if column.startswith("factor__"))
    if not feature_columns:
        raise ValueError("factor cache contains no factor__ columns")
    joined = joined.groupby(["timestamp", "symbol"], as_index=False)[feature_columns].last()
    overlap = set(feature_columns).intersection(frame.columns)
    if overlap:
        frame = frame.drop(columns=sorted(overlap))
    result = frame.merge(joined, on=["timestamp", "symbol"], how="left", validate="one_to_one")
    return result, feature_columns


def run_epoch(model: nn.Module, sections: list[Section], criterion, device: torch.device,
              target_scale: float, optimizer: torch.optim.Optimizer | None,
              capture: EmbeddingCapture, loss_name: str, dispersion_weight: float,
              contrastive_weight: float, contrastive_tau: float, contrastive_bins: int,
              moe_entropy_weight: float, grad_clip: float) -> float:
    training = optimizer is not None
    model.train(training)
    order = np.random.permutation(len(sections)) if training else np.arange(len(sections))
    total, samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for index in order:
            x, y_raw = section_tensors(sections[int(index)], device)
            y = y_raw * target_scale
            capture.tensor = None
            prediction = model(x)
            loss = criterion(prediction, y)
            if loss_name == "mse_spread_disp" and dispersion_weight > 0:
                extra = dispersive_loss(capture.tensor)
                if extra is not None:
                    loss = loss + dispersion_weight * extra
            if loss_name in {"mse_spread_contrast", "mse_ic_spread_contrast"} and contrastive_weight > 0:
                extra = quantile_contrastive_loss(
                    capture.tensor, y, tau=contrastive_tau, bins=contrastive_bins
                )
                if extra is not None:
                    loss = loss + contrastive_weight * extra
            entropy = getattr(model, "last_gate_entropy", None)
            if moe_entropy_weight > 0 and entropy is not None:
                loss = loss - moe_entropy_weight * entropy
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
            samples += len(y)
    return total / max(samples, 1)


@torch.no_grad()
def predict_frame(model: nn.Module, sections: list[Section], device: torch.device,
                  target_scale: float) -> pd.DataFrame:
    model.eval()
    rows = []
    for section in sections:
        x, y = section_tensors(section, device)
        prediction = model(x).cpu().numpy() / target_scale
        for symbol, pred, target in zip(section.symbols, prediction, y.cpu().numpy()):
            rows.append({
                "timestamp": section.timestamp, "symbol": symbol,
                "prediction": float(pred), "target": float(target),
            })
    return pd.DataFrame(rows)


def log_metrics(writer: SummaryWriter, split: str, metrics: dict[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        if math.isfinite(value):
            writer.add_scalar(f"{split}/{name}", value, epoch)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Crypto cross-sectional T+1 sequence trainer")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", default="", help="Comma-separated; default ranks local symbols by liquidity")
    ap.add_argument("--max-symbols", type=int, default=40)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval", default="30m", help="Prediction bar; default is the 30-minute cross-section")
    ap.add_argument("--horizon", type=int, default=1, help="T+N bars; default T+1")
    ap.add_argument(
        "--label-mode", choices=["horizon", "utc_close", "funding_event"],
        default="horizon",
        help="label target: T+N bars, UTC-day close, or next actual funding event",
    )
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--min-symbols", type=int, default=8)
    ap.add_argument("--allow-missing-positioning", action="store_true",
                    help="Permit training without all large-trader/global ratio columns")
    ap.add_argument("--rebuild-cache", action="store_true", help="Recompute the feature cache")
    ap.add_argument("--no-cache", action="store_true", help="Do not read or write feature cache")
    ap.add_argument("--layered-cache", action="store_true",
                    help="Use reusable base-feature and label caches")
    ap.add_argument("--factor-cache", action="append", default=[],
                    help="Selected-factor feature parquet; repeat for train/test cache files")
    ap.add_argument("--alpha-families", default="",
                    help="Optional in-memory Alpha families, e.g. alpha101 or alpha101,alpha191")
    ap.add_argument("--alphas", default="all", help="Comma-separated alpha names or all")
    ap.add_argument("--max-alpha-features", type=int, default=24)
    ap.add_argument("--max-features", type=int, default=192,
                    help="Training-only feature cap after missing/variance filtering; 0 keeps all")
    ap.add_argument("--max-feature-missing", type=float, default=0.35)
    ap.add_argument("--model", choices=("resnet", "gru", "mlp", "lgbm"), default="resnet")
    ap.add_argument("--loss", choices=SUPPORTED_LOSSES, default="mse_ic_spread")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--resnet-blocks", type=int, default=3)
    ap.add_argument("--attention-layers", type=int, default=2)
    ap.add_argument("--attention-heads", type=int, default=8)
    ap.add_argument("--tlob-layers", type=int, default=4)
    ap.add_argument("--tlob-heads", type=int, default=1)
    ap.add_argument("--moe-tau", type=float, default=1.0)
    ap.add_argument("--moe-expert-hidden", type=int, default=8)
    ap.add_argument("--foundation-model-name", default="")
    ap.add_argument("--unfreeze-foundation", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--target-scale", type=float, default=1000.0)
    ap.add_argument("--ic-weight", type=float, default=0.1)
    ap.add_argument("--spread-weight", type=float, default=1.0)
    ap.add_argument("--spread-sharpness", type=float, default=5.0)
    ap.add_argument("--short-weight", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--rank-temperature", type=float, default=1.0)
    ap.add_argument("--dispersion-weight", type=float, default=0.01)
    ap.add_argument("--contrastive-weight", type=float, default=0.01)
    ap.add_argument("--contrastive-tau", type=float, default=0.2)
    ap.add_argument("--contrastive-bins", type=int, default=5)
    ap.add_argument("--moe-entropy-weight", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--validation-start")
    ap.add_argument("--validation-end")
    ap.add_argument("--test-start")
    ap.add_argument("--test-end")
    ap.add_argument("--epoch-metrics", default="train,test",
                    help="comma-separated additional prediction metrics logged each epoch")
    ap.add_argument("--metrics-every", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-dir", default="runs/crypto_t1")
    ap.add_argument("--output-dir", default="outputs/crypto_t1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    # LightGBM follows its own CPU/GPU configuration. PyTorch neural models
    # must use CUDA or Apple MPS and never silently fall back to CPU.
    device = torch.device("cpu") if args.model == "lgbm" else choose_device(args.device)
    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else rank_local_symbols(args.data_dir, args.max_symbols))
    if len(symbols) < args.min_symbols:
        raise ValueError(f"Only {len(symbols)} local symbols selected; need at least {args.min_symbols}")
    # Persist the exact training universe in every artifact.  Previously an
    # automatically ranked universe left args.symbols empty, so live inference
    # could load a valid checkpoint with zero symbols.
    args.symbols = ",".join(symbols)
    print(
        f"Loading {len(symbols)} symbols: interval={args.interval} "
        f"horizon=T+{args.horizon} label={args.label_mode}"
    )
    frame, features = load_feature_frame(
        args.data_dir, symbols, interval=args.interval, horizon=args.horizon, days=args.days,
        require_positioning=not args.allow_missing_positioning,
        cache=not args.no_cache, rebuild_cache=args.rebuild_cache,
        layered_cache=args.layered_cache, label_mode=args.label_mode,
    )
    frame, factor_features = merge_factor_caches(frame, args.factor_cache)
    features.extend(factor_features)
    if factor_features:
        print(f"Joined {len(factor_features)} selected-factor cache features")
    alpha_families = [x.strip() for x in args.alpha_families.split(",") if x.strip()]
    if alpha_families:
        frame, alpha_features = add_alpha_features(
            frame, alpha_families,
            [x.strip() for x in args.alphas.split(",") if x.strip()],
            max_features=args.max_alpha_features,
        )
        features.extend(alpha_features)
        print(f"Computed {len(alpha_features)} in-memory Alpha features")
    candidate_count = len(features)
    features = select_features(
        frame, features, train_fraction=args.train_frac,
        max_missing_rate=args.max_feature_missing, max_features=args.max_features,
        train_end=args.train_end,
    )
    if not features:
        raise ValueError("No features survived training-period filtering")
    # All retained inputs are already cross-sectionally standardized; zero is
    # therefore the neutral imputation for warm-up or sporadic missing values.
    frame[features] = frame[features].fillna(0.0)
    sections = build_sections(frame, features, args.seq_len, args.min_symbols)
    explicit = [args.train_start, args.train_end, args.validation_start,
                args.validation_end, args.test_start, args.test_end]
    if any(explicit) and not all(explicit):
        raise ValueError("explicit date split requires all six train/validation/test bounds")
    train_sections, val_sections, test_sections = (
        split_sections_by_date(sections, args) if all(explicit)
        else split_sections(sections, args.train_frac, args.val_frac)
    )
    print(
        f"Features candidates/selected={candidate_count}/{len(features)} sections train/val/test="
        f"{len(train_sections)}/{len(val_sections)}/{len(test_sections)} device={device}"
    )
    run_name = f"{args.model}_{args.loss}_{args.interval}_t{args.horizon}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "lgbm":
        model = train_lgbm(train_sections, seed=args.seed)
        val_pred = predict_lgbm(model, val_sections)
        test_pred = predict_lgbm(model, test_sections)
        def make_frame(sections, predictions):
            rows, offset = [], 0
            for section in sections:
                for symbol, target in zip(section.symbols, section.y):
                    rows.append({"timestamp": section.timestamp, "symbol": symbol,
                                 "prediction": float(predictions[offset]), "target": float(target)})
                    offset += 1
            return pd.DataFrame(rows)
        val_frame = make_frame(val_sections, val_pred)
        test_frame = make_frame(test_sections, test_pred)
        model.booster_.save_model(str(output_dir / "model.txt"))
        (output_dir / "features.json").write_text(
            json.dumps({"selected": features, "candidate_count": candidate_count,
                        "args": vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")
        val_frame.to_parquet(output_dir / "best_validation_predictions.parquet", index=False)
        test_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps({"validation": summarize_metrics(val_frame), "test": summarize_metrics(test_frame)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Artifacts: {output_dir}")
        return 0
    model = build_model(
        args.model, args.seq_len, len(features), args.hidden_dim, args.dropout,
        args.resnet_blocks, args.attention_layers, args.attention_heads,
        args.moe_tau, args.moe_expert_hidden, args.tlob_layers, args.tlob_heads,
        args.foundation_model_name, not args.unfreeze_foundation,
    ).to(device)
    criterion = build_loss(
        args.loss, ic_weight=args.ic_weight, spread_weight=args.spread_weight,
        spread_sharpness=args.spread_sharpness, short_weight=args.short_weight,
        huber_delta=args.huber_delta, rank_weight=args.rank_weight,
        rank_temperature=args.rank_temperature,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    capture = EmbeddingCapture(model)
    (output_dir / "features.json").write_text(
        json.dumps({"candidate_count": candidate_count, "selected": features},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(Path(args.run_dir) / run_name)
    writer.add_text("config/features", "\n".join(features))
    writer.add_text("config/arguments", json.dumps(vars(args), ensure_ascii=False, indent=2))
    writer.add_scalar("config/feature_count", len(features), 0)
    epoch_metric_splits = {value.strip().lower() for value in args.epoch_metrics.split(",") if value.strip()}
    unknown_metric_splits = epoch_metric_splits - {"train", "validation", "test"}
    if unknown_metric_splits:
        raise ValueError(f"unknown --epoch-metrics splits: {sorted(unknown_metric_splits)}")
    if "test" in epoch_metric_splits:
        print("NOTE: test metrics are logged for monitoring only; checkpoint selection remains validation RankIC only")
    best_rank_ic, stale = -math.inf, 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model, train_sections, criterion, device, args.target_scale, optimizer,
                capture, args.loss, args.dispersion_weight, args.contrastive_weight,
                args.contrastive_tau, args.contrastive_bins, args.moe_entropy_weight,
                args.grad_clip,
            )
            val_loss = run_epoch(
                model, val_sections, criterion, device, args.target_scale, None,
                capture, args.loss, args.dispersion_weight, args.contrastive_weight,
                args.contrastive_tau, args.contrastive_bins, args.moe_entropy_weight,
                args.grad_clip,
            )
            val_frame = predict_frame(model, val_sections, device, args.target_scale)
            metrics = summarize_metrics(val_frame)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", val_loss, epoch)
            writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            log_metrics(writer, "validation", metrics, epoch)
            epoch_metrics: dict[str, dict[str, float]] = {"validation": metrics}
            if args.metrics_every > 0 and epoch % args.metrics_every == 0:
                if "train" in epoch_metric_splits:
                    train_metrics = summarize_metrics(
                        predict_frame(model, train_sections, device, args.target_scale)
                    )
                    epoch_metrics["train"] = train_metrics
                    log_metrics(writer, "train", train_metrics, epoch)
                if "test" in epoch_metric_splits:
                    test_epoch_metrics = summarize_metrics(
                        predict_frame(model, test_sections, device, args.target_scale)
                    )
                    epoch_metrics["test"] = test_epoch_metrics
                    log_metrics(writer, "test", test_epoch_metrics, epoch)
            score = metrics["rank_ic"]
            extra = ""
            if "train" in epoch_metrics:
                item = epoch_metrics["train"]
                extra += f" TrainIC={item['ic']:.4f} TrainRankICIR={item['rank_icir']:.3f}"
            if "test" in epoch_metrics:
                item = epoch_metrics["test"]
                extra += f" TestIC={item['ic']:.4f} TestICIR={item['icir']:.3f} TestRankICIR={item['rank_icir']:.3f}"
            print(
                f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
                f"IC={metrics['ic']:.4f} RankIC={metrics['rank_ic']:.4f} "
                f"ICIR={metrics['icir']:.3f} RankICIR={metrics['rank_icir']:.3f}{extra}"
            )
            if epoch == 1 or (math.isfinite(score) and score > best_rank_ic):
                if math.isfinite(score):
                    best_rank_ic = score
                stale = 0
                torch.save({
                    "model_state": model.state_dict(), "features": features,
                    "args": vars(args), "epoch": epoch, "validation": metrics,
                }, output_dir / "best_model.pt")
                val_frame.to_parquet(output_dir / "best_validation_predictions.parquet", index=False)
            else:
                stale += 1
            scheduler.step()
            writer.flush()
            if stale >= args.patience:
                print(f"Early stopping after {stale} epochs without RankIC improvement")
                break
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_frame = predict_frame(model, test_sections, device, args.target_scale)
        test_metrics = summarize_metrics(test_frame)
        test_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)
        (output_dir / "metrics.json").write_text(
            json.dumps({"validation": checkpoint["validation"], "test": test_metrics},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_metrics(writer, "test", test_metrics, checkpoint["epoch"])
        writer.add_hparams(
            {"model": args.model, "loss": args.loss, "interval": args.interval,
             "seq_len": args.seq_len, "lr": args.lr},
            {f"hparam/test_{k}": v for k, v in test_metrics.items() if math.isfinite(v)},
        )
        print("Test:", json.dumps(test_metrics, ensure_ascii=False))
        print(f"Artifacts: {output_dir}")
    finally:
        capture.close()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
