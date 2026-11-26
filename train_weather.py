import argparse
import collections
import json
import logging
import sys

from weather_pca import cache, config, evaluate, features, loader, ovr

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an OvR Perceptron weather classifier."
    )
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--bias", type=float, default=config.BIAS_INIT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--include-rainfall",
        action="store_true",
        default=config.INCLUDE_RAINFALL,
    )
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args(argv)


def _feature_names(include_rainfall: bool) -> list[str]:
    names = ["bias", *config.BASE_FEATURES]
    if include_rainfall:
        names.append(config.RAINFALL_FEATURE)
    return names


def _format_metric_row(
    split_name: str, row_name: str, metrics: dict[str, float | None] | None
) -> str:
    if metrics is None:
        return f"{split_name:<5}  {row_name:<5}  {'N/A':<11}  {'N/A':<8}  {'N/A':<9}  N/A"
    sse_s = f"{metrics['SSE']:.6f}"
    mse_s = f"{metrics['MSE']:.6f}"
    mape_s = f"{metrics['MAPE']:.6f}"
    r2_s = "undefined" if metrics["R2"] is None else f"{metrics['R2']:.6f}"
    return f"{split_name:<5}  {row_name:<5}  {sse_s:<11}  {mse_s:<8}  {mape_s:<9}  {r2_s}"


def _print_report(
    args: argparse.Namespace,
    feature_names: list[str],
    models: dict[str, dict | None],
    train_metrics: dict[str, dict | None],
    test_metrics: dict[str, dict | None],
    test_acc: float,
    baseline_cls: str,
    baseline_acc: float,
    cm,
    data_summary: dict[str, int],
    test_dates: list[str],
) -> None:
    print(
        f"=== Final Weights (OvR Perceptron | eta={args.lr} | bias_init={args.bias} "
        f"| max_epochs={args.epochs} | seed={args.seed}) ==="
    )
    print(f"Features: {', '.join(feature_names)}")
    for cls in config.CLASSES:
        m = models.get(cls)
        if m is None:
            print(f"{cls:<5} : skipped (no training samples)")
        else:
            w_str = ", ".join(f"{float(v):.6f}" for v in m["weights"])
            print(
                f"{cls:<5} : converged={m['converged']} "
                f"stopped_epoch={m['stopped_epoch']} "
                f"train_errors={m['train_errors']} "
                f"w=[{w_str}]"
            )

    print()
    print("=== Metrics ===")
    print("Split  Class  SSE          MSE       MAPE(%)    R2")
    for split_name, split_metrics in (("train", train_metrics), ("test", test_metrics)):
        for row_name in (*config.CLASSES, "Mean"):
            print(_format_metric_row(split_name, row_name, split_metrics.get(row_name)))

    print()
    test_label = ", ".join(test_dates) if test_dates else "unknown"
    print(f"=== OvR Classification (test: {test_label}) ===")
    print(
        f"Accuracy: {test_acc:.6f} | "
        f"Majority baseline ({baseline_cls}): {baseline_acc:.6f}"
    )
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("        " + "  ".join(f"{cls:<6}" for cls in config.CLASSES))
    for i, cls in enumerate(config.CLASSES):
        row_vals = "  ".join(f"{int(cm[i, j]):<6}" for j in range(len(config.CLASSES)))
        print(f"{cls:<5}   {row_vals}")

    print()
    print("=== Data Summary ===")
    print(
        f"train_rows={data_summary['train_rows']} "
        f"test_rows={data_summary['test_rows']} "
        f"file_bad_header={data_summary['file_bad_header']} "
        f"row_bad_ncols={data_summary['row_bad_ncols']} "
        f"row_unknown_label={data_summary['row_unknown_label']} "
        f"row_bad_value={data_summary['row_bad_value']} "
        f"row_out_of_bounds={data_summary['row_out_of_bounds']} "
        f"range_swaps={data_summary['range_swaps']}"
    )


def _serialize_models(models: dict[str, dict | None]) -> dict:
    out = {}
    for cls in config.CLASSES:
        m = models.get(cls)
        if m is None:
            out[cls] = None
        else:
            out[cls] = {
                "weights": [float(v) for v in m["weights"]],
                "converged": bool(m["converged"]),
                "stopped_epoch": int(m["stopped_epoch"]),
                "train_errors": int(m["train_errors"]),
            }
    return out


def _serialize_metrics(split_metrics: dict[str, dict | None]) -> dict:
    out = {}
    for key, m in split_metrics.items():
        if m is None:
            out[key] = None
        else:
            out[key] = {
                "SSE": float(m["SSE"]),
                "MSE": float(m["MSE"]),
                "MAPE": float(m["MAPE"]),
                "R2": None if m["R2"] is None else float(m["R2"]),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        train_files, test_files, train_dates, test_dates = loader.discover_files(
            config.TRAIN_ROOT,
            config.TEST_ROOT,
            config.OUTLOOK_DIR_GLOB,
            limit_files=args.limit_files,
        )
        logger.info(
            "Discovered %d train files (%s) and %d test files (%s)",
            len(train_files),
            ", ".join(train_dates),
            len(test_files),
            ", ".join(test_dates),
        )

        cache_file = cache.cache_path_for(
            config.CACHE_DIR, train_files, test_files, args.include_rainfall
        )
        if not args.no_cache and cache_file.is_file():
            logger.info("Loading dataset from cache: %s", cache_file)
            X_train_raw, y_train, X_test_raw, y_test, counters = cache.load_cached(
                cache_file
            )
        else:
            counters = collections.Counter()
            X_train_raw, y_train = loader.load_split(
                train_files, args.include_rainfall, counters
            )
            X_test_raw, y_test = loader.load_split(
                test_files, args.include_rainfall, counters
            )
            if not args.no_cache:
                logger.info("Saving dataset to cache: %s", cache_file)
                cache.save_cached(
                    cache_file, X_train_raw, y_train, X_test_raw, y_test, counters
                )

        X_train_raw, y_train, train_oob = features.filter_out_of_bounds(
            X_train_raw, y_train, include_rainfall=args.include_rainfall
        )
        X_test_raw, y_test, test_oob = features.filter_out_of_bounds(
            X_test_raw, y_test, include_rainfall=args.include_rainfall
        )
        row_out_of_bounds = train_oob + test_oob
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    mean, std = features.fit_standardizer(X_train_raw)
    X_train_std = features.apply_standardizer(X_train_raw, mean, std)
    X_test_std = features.apply_standardizer(X_test_raw, mean, std)
    X_train = features.add_bias_column(X_train_std)
    X_test = features.add_bias_column(X_test_std)

    models = ovr.train_ovr(
        X_train,
        y_train,
        lr=args.lr,
        bias_init=args.bias,
        max_epochs=args.epochs,
        seed=args.seed,
    )

    train_metrics = evaluate.evaluate_split_metrics(X_train, y_train, models)
    test_metrics = evaluate.evaluate_split_metrics(X_test, y_test, models)

    y_pred = ovr.predict(X_test, models)
    test_acc = evaluate.accuracy(y_test, y_pred)
    baseline_cls, baseline_acc = evaluate.majority_baseline(y_train, y_test)
    cm = evaluate.confusion_matrix(y_test, y_pred)

    data_summary = {
        "train_rows": int(X_train_raw.shape[0]),
        "test_rows": int(X_test_raw.shape[0]),
        "file_bad_header": int(counters["file_bad_header"]),
        "row_bad_ncols": int(counters["row_bad_ncols"]),
        "row_unknown_label": int(counters["row_unknown_label"]),
        "row_bad_value": int(counters["row_bad_value"]),
        "row_out_of_bounds": int(row_out_of_bounds),
        "range_swaps": int(counters["range_swaps"]),
    }
    feature_names = _feature_names(args.include_rainfall)

    _print_report(
        args=args,
        feature_names=feature_names,
        models=models,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        test_acc=test_acc,
        baseline_cls=baseline_cls,
        baseline_acc=baseline_acc,
        cm=cm,
        data_summary=data_summary,
        test_dates=test_dates,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_payload = {
        "config": {
            "learning_rate": float(args.lr),
            "bias_init": float(args.bias),
            "max_epochs": int(args.epochs),
            "seed": int(args.seed),
            "include_rainfall": bool(args.include_rainfall),
            "limit_files": None if args.limit_files is None else int(args.limit_files),
            "train_dates": list(train_dates),
            "test_dates": list(test_dates),
        },
        "features": feature_names,
        "standardizer": {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
        },
        "models": _serialize_models(models),
        "metrics": {
            "train": _serialize_metrics(train_metrics),
            "test": _serialize_metrics(test_metrics),
        },
        "classification": {
            "test_accuracy": float(test_acc),
            "baseline_class": str(baseline_cls),
            "baseline_accuracy": float(baseline_acc),
            "confusion_matrix": [[int(v) for v in row] for row in cm],
        },
        "data_summary": data_summary,
    }
    with open(config.OUTPUT_DIR / "weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
