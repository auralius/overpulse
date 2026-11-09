#!/usr/bin/env python3
"""Run a *trained exported* Noodle2 PeakNet1D-PPG on a real MAX30100 CSV.

No serial connection or firmware changes are needed for this PC-first test.
Requires generate_ppg_dataset.py and train_peaknet_ppg_qat.py in this directory,
or on PYTHONPATH, plus numpy and matplotlib.

Example:
  python test_real_ppg.py \\
    --csv ppg_session01.csv \\
    --export runs/peaknet_ppg_qat/noodle2_export \\
    --out runs/peaknet_ppg_real --warmup-sec 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from generate_ppg_dataset import WINDOW, FS, raw_to_int8
from train_peaknet_ppg_qat import load_export, run_exact, peaks_from_score


def load_csv(path):
    t, seq, ir, red = [], [], [], []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        needed = {"seq", "readout_est_us", "ir", "red"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain {sorted(needed)}, got {reader.fieldnames}")
        for number, row in enumerate(reader, start=2):
            try:
                seq.append(int(row["seq"]))
                t.append(int(row["readout_est_us"]))
                ir.append(int(row["ir"]))
                red.append(int(row["red"]))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid data in CSV line {number}: {e}") from e
    if len(t) < WINDOW:
        raise ValueError("Recording must have at least 256 rows")
    seq = np.asarray(seq, dtype=np.int64)
    time_us = np.asarray(t, dtype=np.int64)
    raw = np.array([ir, red], dtype=np.int64)
    if np.any((raw < 0) | (raw > 65535)):
        raise ValueError("IR/Red must be uint16 sensor counts in [0,65535]")
    if np.any(np.diff(seq) != 1):
        raise ValueError("Sequence gaps or nonmonotonic samples: handle gaps before inference")
    if np.any(np.diff(time_us) <= 0):
        raise ValueError("Nonmonotonic acquisition timestamps")
    ts = (time_us - time_us[0]) / 1_000_000.0
    return seq, ts, raw


def moving_average(x, n):
    pad = int(n) // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(n) / n, mode="valid")[:len(x)]


def analyze(raw, ts, layers, stride, threshold, refractory,
            warmup_sec, min_median, edge=13):
    n = raw.shape[1]
    starts = list(range(0, n - WINDOW + 1, stride))
    if starts[-1] != n - WINDOW:
        starts.append(n - WINDOW)
    sums = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.uint16)
    accepted = rejected = 0
    for start in starts:
        end = start + WINDOW
        win = raw[:, start:end]
        # Exclude windows with low/absent optical DC (finger-off).
        if min(np.median(win[0]), np.median(win[1])) < min_median:
            rejected += 1
            continue
        q = raw_to_int8(win)      # exact 2 x 256 CHW integer preprocessing
        logits = run_exact(q, layers)[0]  # exact int8 Noodle2 arithmetic
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / 16.0))
        sl = slice(start + edge, end - edge)
        sums[sl] += p[edge:WINDOW-edge]
        count[sl] += 1
        accepted += 1
    valid = (count > 0) & (ts >= warmup_sec)
    score = np.zeros(n, dtype=np.float64)
    score[valid] = sums[valid] / count[valid]
    peaks = np.array([i for i in peaks_from_score(score, threshold,
                                                 refractory) if valid[i]], dtype=np.int64)
    return score, peaks, valid, count, accepted, rejected


def save_figures(ts, raw, score, peaks, valid, out, zoom_from, zoom_length):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ir = raw[0].astype(np.float64)
    red = raw[1].astype(np.float64)
    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(ts, ir, lw=0.8, label="IR raw")
    if len(peaks):
        ax[0].scatter(ts[peaks], ir[peaks], s=18, color="tab:red", label="Predicted systolic", zorder=3)
    ax[0].set_ylabel("IR raw counts")
    ax[0].legend(loc="best")
    ax[1].plot(ts, red, lw=0.8, color="tab:orange", label="Red raw")
    ax[1].set_ylabel("Red raw counts")
    ax[1].legend(loc="best")
    ax[2].plot(ts, score, lw=0.9, color="tab:blue", label="INT8 peak score")
    if len(peaks):
        ax[2].scatter(ts[peaks], score[peaks], s=18, color="tab:red", zorder=3)
    ax[2].set_ylim(-0.03, 1.03)
    ax[2].set_ylabel("Peak score")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="best")
    fig.suptitle("MAX30100 real IR / Red: PeakNet1D Noodle2-exact INT8")
    fig.tight_layout()
    fig.savefig(out / "real_ppg_overview.png", dpi=160)
    plt.close(fig)

    select = (ts >= zoom_from) & (ts <= zoom_from + zoom_length)
    ind = np.flatnonzero(select)
    if not len(ind):
        return
    ac = ir - moving_average(ir, 151)  # display-only high-pass, NOT model preprocessing
    pk = peaks[(ts[peaks] >= zoom_from) & (ts[peaks] <= zoom_from + zoom_length)]
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(ts[select], ac[select], color="tab:blue", lw=1.0, label="IR minus local 1.51s mean (display only)")
    if len(pk):
        ax[0].scatter(ts[pk], ac[pk], s=32, color="tab:red", label="Predicted systolic peaks", zorder=4)
    ax[0].set_ylabel("IR AC (counts)")
    ax[0].legend(loc="best")
    ax[1].plot(ts[select], score[select], lw=1.0, color="tab:blue")
    if len(pk):
        ax[1].scatter(ts[pk], score[pk], s=32, color="tab:red", zorder=4)
    ax[1].set_ylabel("Peak probability")
    ax[1].set_ylim(-0.03, 1.03)
    ax[1].set_xlabel("Time (s)")
    fig.suptitle(f"MAX30100 detailed view: {zoom_from:.1f}-{zoom_from + zoom_length:.1f}s")
    fig.tight_layout()
    fig.savefig(out / "real_ppg_zoom.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Real MAX30100 capture CSV")
    ap.add_argument("--export", required=True,
                    help="Trained Noodle2 export folder with model_manifest.json and int8.bin")
    ap.add_argument("--out", default="runs/peaknet_ppg_real")
    ap.add_argument("--stride", type=int, default=64,
                    help="Sliding-window step in samples; default 64")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--refractory", type=int, default=32,
                    help="Minimum peak separation in samples, default 32 (320ms)")
    ap.add_argument("--warmup-sec", type=float, default=5.0,
                    help="Ignore early sensor-startup seconds; default 5")
    ap.add_argument("--min-median", type=float, default=1000.0,
                    help="Reject windows whose IR or Red median is below this count")
    ap.add_argument("--zoom-from", type=float, default=10.0)
    ap.add_argument("--zoom-length", type=float, default=12.0)
    args = ap.parse_args()
    if args.stride < 1 or args.stride > WINDOW-26:
        ap.error("--stride must be 1..230 to avoid uncovered window gaps")
    if not 0 < args.threshold < 1:
        ap.error("--threshold must be between zero and one")
    if args.refractory < 1 or args.warmup_sec < 0:
        ap.error("--refractory must be positive and --warmup-sec nonnegative")

    layers, manifest = load_export(args.export)
    if manifest.get("channels") != ["ir", "red"] or manifest.get("window") != WINDOW:
        raise ValueError("Model is not two-channel CHW 256-sample PPG")
    if manifest.get("sample_rate_hz") != FS:
        raise ValueError("Model sampling rate mismatch")
    if not np.isclose(manifest.get("output_logit_scale"), 1/16):
        raise ValueError("Model logit scale mismatch")
    seq, ts, raw = load_csv(args.csv)
    sample_dt = np.diff(ts)
    if np.any(sample_dt > 0.015):
        print(f"WARNING: {np.count_nonzero(sample_dt > .015)} timestamp intervals >15ms;"
              " interpolation/resampling is not performed.")
    rate = (len(ts)-1)/(ts[-1]-ts[0])
    if abs(rate - FS) > 3:
        print(f"WARNING: sample rate {rate:.2f}Hz differs from expected {FS}Hz.")
    score, peaks, valid, count, ok, rej = analyze(
        raw,ts,layers,args.stride,args.threshold,args.refractory,
        args.warmup_sec,args.min_median)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    is_peak = np.zeros(len(ts), dtype=np.uint8)
    is_peak[peaks] = 1
    with (out / "real_ppg_predictions.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq", "timestamp_s", "ir", "red", "peak_score", "predicted_systolic",
                     "valid_inference", "window_votes"])
        for i in range(len(ts)):
            wr.writerow([int(seq[i]), f"{ts[i]:.6f}", int(raw[0, i]), int(raw[1, i]),
                         f"{score[i]:.7f}", int(is_peak[i]), int(valid[i]), int(count[i])])
    duration = (ts[peaks[-1]] - ts[peaks[0]]) if len(peaks) >= 2 else None
    estimated_bpm = (60 * (len(peaks)-1) / duration) if duration is not None and duration > 0 else None
    report = {
        "recording_samples":len(ts),
        "duration_s":float(ts[-1]),
        "mean_sample_rate_hz":float(rate),
        "detected_systolic_count":len(peaks),
        "peak_times_s": [round(float(ts[i]),4) for i in peaks],
        "rough_detected_bpm": None if estimated_bpm is None else round(estimated_bpm,2),
        "window_length":WINDOW,
        "stride":args.stride,
        "threshold":args.threshold,
        "refractory_samples":args.refractory,
        "warmup_sec":args.warmup_sec,
        "accepted_windows":ok,
        "rejected_low_light_windows":rej,
        "scored_samples":int(valid.sum()),
        "warning":"Unlabeled real PPG: this is a visual inspection, not precision/recall/F1 validation."
    }
    (out / "real_ppg_summary.json").write_text(json.dumps(report,indent=2) + "\n")
    save_figures(ts,raw,score,peaks,valid,out,args.zoom_from,args.zoom_length)
    print(json.dumps({k:v for k,v in report.items() if k!="peak_times_s"}, indent=2))
    print(f"Saved: {out.resolve()}/real_ppg_predictions.csv")
    print(f"Saved: {out.resolve()}/real_ppg_overview.png")
    print(f"Saved: {out.resolve()}/real_ppg_zoom.png")
    print(f"Saved: {out.resolve()}/real_ppg_summary.json")
    print("No ground-truth peak labels exist for this CSV; visually inspect before reporting accuracy.")


if __name__ == "__main__":
    main()
