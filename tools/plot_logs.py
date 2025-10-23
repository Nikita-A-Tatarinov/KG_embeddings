# tools/plot_logs.py
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def ema(series: pd.Series, alpha: float) -> pd.Series:
    """Exponential moving average; ignores NaNs."""
    if series.isna().all() or not (0 < alpha < 1):
        return series
    return series.ffill().ewm(alpha=1 - alpha, adjust=False).mean()


def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert object columns to numeric by coercion.
    If coercion yields at least one non-NaN, adopt the numeric version.
    Leaves fully non-numeric columns untouched.
    """
    for c in df.columns:
        s = df[c]
        # Skip if it's already numeric
        if pd.api.types.is_numeric_dtype(s):
            continue
        # Try coercion
        maybe = pd.to_numeric(s, errors="coerce")
        # Adopt if we got any numeric info out (not all NaN)
        if maybe.notna().any():
            df[c] = maybe
    return df


def plot_xy(x, y, title, xlabel, ylabel, path, smooth=None, marker=None, ylim=None):
    plt.figure()
    yy = y.copy()
    if smooth is not None and smooth > 0 and smooth < 1:
        yy = ema(yy, smooth)
    plt.plot(x, yy, marker=marker if marker else None, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()


def main(args):
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = to_numeric_cols(df)

    # Basic columns
    step = df["step"]
    loss = df.get("loss")
    lr = df.get("lr")

    # MED columns (may not exist)
    L_total = df.get("L_total")
    L_ml = df.get("L_ml")
    L_ei = df.get("L_ei")

    # Validation metrics (may be NaN on train rows)
    vMR = df.get("val_MR")
    vMRR = df.get("val_MRR")
    vH1 = df.get("val_H@1")
    vH3 = df.get("val_H@3")
    vH10 = df.get("val_H@10")

    # Masks
    train_mask = loss.notna() if loss is not None else pd.Series([False] * len(df))
    eval_mask = vMRR.notna() if vMRR is not None else pd.Series([False] * len(df))

    # 1) Training loss
    if loss is not None:
        plot_xy(
            step[train_mask],
            loss[train_mask],
            title="Training Loss vs Step",
            xlabel="Step",
            ylabel="Loss",
            path=os.path.join(args.out, "train_loss.png"),
            smooth=args.smooth,
        )

    # 2) MED terms (if available)
    if L_total is not None:
        plot_xy(
            step[train_mask],
            L_total[train_mask],
            title="MED: L_total vs Step",
            xlabel="Step",
            ylabel="L_total",
            path=os.path.join(args.out, "med_L_total.png"),
            smooth=args.smooth,
        )

    if L_ml is not None and L_ml.notna().any():
        plot_xy(
            step[train_mask],
            L_ml[train_mask],
            title="MED: L_ml vs Step",
            xlabel="Step",
            ylabel="L_ml",
            path=os.path.join(args.out, "med_L_ml.png"),
            smooth=args.smooth,
        )

    if L_ei is not None and L_ei.notna().any():
        plot_xy(
            step[train_mask],
            L_ei[train_mask],
            title="MED: L_ei vs Step",
            xlabel="Step",
            ylabel="L_ei",
            path=os.path.join(args.out, "med_L_ei.png"),
            smooth=args.smooth,
        )

    # 3) Learning rate
    if lr is not None:
        plot_xy(
            step,
            lr,
            title="Learning Rate vs Step",
            xlabel="Step",
            ylabel="LR",
            path=os.path.join(args.out, "lr.png"),
            smooth=None,
        )

    # 4) Validation metrics
    if eval_mask.any():
        s_eval = step[eval_mask]
        if vMRR is not None and vMRR.notna().any():
            plot_xy(
                s_eval,
                vMRR[eval_mask],
                title="Validation MRR vs Step",
                xlabel="Step",
                ylabel="MRR",
                path=os.path.join(args.out, "val_mrr.png"),
                smooth=None,
                marker="o",
                ylim=(0, 1),
            )

        if vH1 is not None and vH1.notna().any():
            plot_xy(
                s_eval,
                vH1[eval_mask],
                title="Validation Hits@1 vs Step",
                xlabel="Step",
                ylabel="Hits@1",
                path=os.path.join(args.out, "val_hits1.png"),
                smooth=None,
                marker="o",
                ylim=(0, 1),
            )

        if vH3 is not None and vH3.notna().any():
            plot_xy(
                s_eval,
                vH3[eval_mask],
                title="Validation Hits@3 vs Step",
                xlabel="Step",
                ylabel="Hits@3",
                path=os.path.join(args.out, "val_hits3.png"),
                smooth=None,
                marker="o",
                ylim=(0, 1),
            )

        if vH10 is not None and vH10.notna().any():
            plot_xy(
                s_eval,
                vH10[eval_mask],
                title="Validation Hits@10 vs Step",
                xlabel="Step",
                ylabel="Hits@10",
                path=os.path.join(args.out, "val_hits10.png"),
                smooth=None,
                marker="o",
                ylim=(0, 1),
            )

        if vMR is not None and vMR.notna().any():
            plot_xy(
                s_eval,
                vMR[eval_mask],
                title="Validation Mean Rank (MR) vs Step",
                xlabel="Step",
                ylabel="MR (lower is better)",
                path=os.path.join(args.out, "val_mr.png"),
                smooth=None,
                marker="o",
            )

    print(f"Saved plots to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to log.csv")
    ap.add_argument("--out", type=str, default="./plots", help="Output directory for images")
    ap.add_argument(
        "--smooth", type=float, default=0.9, help="EMA smoothing factor in [0,1). Higher=more smoothing; 0 disables."
    )
    ap.add_argument("--dpi", type=int, default=140, help="Save resolution (DPI)")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()
    main(args)
