#!/usr/bin/env python
r"""
Run NSS and AUC-J for ONE saliency family and write the two summary tables.

This is the project's own driver, modelled on the metrics harness in the
UNISAL repository (metrics/run_metrics.py). It deliberately does NOT import
that file, so this folder stays self-contained:

  * it calls THIS folder's scripts -- nss_score_new.py and auc_score_new.py,
    which are algorithmically identical to the copies in the UNISAL repo --
  * it writes exactly the same output structure and the same column names,
    so numbers from the two projects can be put side by side:

        <output_dir>/nss/<task>/<task>_nss_summary.csv            per participant
        <output_dir>/auc/<task>/<task>_auc_summary.csv            per participant
        <output_dir>/auc/<task>/<task>_auc_pooled_summary.csv     whole video
        <output_dir>/summary_all_tasks.csv                        one row per task + ALL
        <output_dir>/summary_all_participants.csv                 one row per participant

Two differences from the UNISAL harness, both on purpose:

  1. --model picks the map family, so one run = one model and the output
     folder says which model it was:

        --object   saliency_frames             -> metrics_output_object
        --face     saliency_frames_face        -> metrics_output_face
        --hand     saliency_frames_hand        -> metrics_output_hand
        --social   saliency_frames_social      -> metrics_output_social
        --unisal   (needs --saliency_root)     -> metrics_output_unisal

  2. the map layout is detected automatically. This project writes
     <root>/<task>/frame_0001.png while the UNISAL predictions sit in
     <root>/<task>/maps/frame_000001.png; either is accepted, and the
     file-name pattern is derived from the first file found. Nothing has to
     be renamed or copied to score either one.

Only tasks that have BOTH maps and gaze files are scored; everything else is
listed as skipped with the reason. That is the default behaviour, as asked.

Usage
    python metrics\run_metrics.py --object                     # the usual case
    python metrics\run_metrics.py --face
    python metrics\run_metrics.py --object --limit 3 --force   # quick check
    python metrics\run_metrics.py --object --dry_run           # show the plan only

    # another machine / another copy of the study
    python metrics\run_metrics.py --object --gaze_root "D:\path\to\Resample_Output"

    # score the UNISAL predictions with the same harness
    python metrics\run_metrics.py --unisal --saliency_root "D:\...\unisal-master\outputs_chunked"

    # verify the plumbing on made-up data; reads none of yours
    python metrics\run_metrics.py --self_test

Requires pandas, numpy and opencv-python for the two metric scripts; --python
picks the interpreter and otherwise this one is used.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

# This file lives in <project>/metrics/. The metric scripts sit next to it,
# while the maps, the gaze tables and the result folders live one level up in
# the project root.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

NSS_SCRIPT = HERE / "nss_score_new.py"
AUC_SCRIPT = HERE / "auc_score_new.py"

# map family -> folder in the project root
MODEL_ROOTS = {
    "object": "saliency_frames",
    "face": "saliency_frames_face",
    "hand": "saliency_frames_hand",
    "social": "saliency_frames_social",
}
MODEL_ORDER = ("object", "face", "hand", "social", "unisal")

DEFAULT_GAZE_ROOT = PROJECT_ROOT / "Resample_Output"

# file names the metric scripts write themselves; never treated as gaze input
RESULT_SUFFIXES = ("_nss_summary.csv", "_auc_summary.csv",
                   "_auc_pooled_by_frame.csv", "_auc_pooled_summary.csv")

FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)

# identical to metrics/run_metrics.py in the UNISAL repo
TASK_COLUMNS = ["task", "participants", "gaze_frames", "mean_nss", "sd_nss",
                "weighted_nss", "mean_auc_j", "sd_auc_j", "weighted_auc_j",
                "pooled_auc_j", "pooled_auc_sd", "pooled_frames"]
PARTICIPANT_COLUMNS = ["task", "gaze_file", "rows_in_file", "rows_scored",
                       "mean_nss", "std_nss", "min_nss", "max_nss",
                       "mean_auc_j", "std_auc_j", "min_auc_j", "max_auc_j",
                       "skipped_no_map", "skipped_out_of_bounds",
                       "skipped_missing_xy"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def find_python(explicit=None):
    """Return an interpreter that can import pandas, numpy and cv2."""
    candidates = [explicit] if explicit else [sys.executable, "python", "python3"]
    tried = []
    for candidate in candidates:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        try:
            result = subprocess.run(
                [candidate, "-c", "import pandas, numpy, cv2"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    print("ERROR: no Python with pandas/numpy/cv2 found. Tried:")
    for candidate in tried:
        print(f"  {candidate}")
    print("Pass --python <path> to choose one.")
    return None


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def stdev(values):
    """Sample standard deviation (n-1), like the UNISAL harness."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def weighted_mean(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights)
             if v is not None and w is not None and w > 0]
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def is_result_csv(name):
    return any(name.endswith(suffix) for suffix in RESULT_SUFFIXES)


def frame_digits(filename):
    """'frame_0001.png' -> 4, 'frame_000001.png' -> 6, unknown -> 4."""
    match = FRAME_NAME_RE.match(filename)
    return len(match.group(1)) if match else 4


# --------------------------------------------------------------------------
# Task discovery
# --------------------------------------------------------------------------
def detect_map_layout(task_dir):
    """(maps_dir, pattern, n_maps) for one task folder, or (None, None, 0).

    Accepts both layouts:
        <task>/maps/frame_000001.png     (UNISAL predictions)
        <task>/frame_0001.png            (this project's maps)
    The zero-padding is read off the first file, so --pattern never has to be
    typed by hand.
    """
    candidates = []
    sub = task_dir / "maps"
    if sub.is_dir():
        candidates.append(sub)
    candidates.append(task_dir)

    for folder in candidates:
        pngs = sorted(folder.glob("*.png"))
        if pngs:
            digits = frame_digits(pngs[0].name)
            return folder, f"frame_{{frame:0{digits}d}}.png", len(pngs)
    return None, None, 0


def find_tasks(saliency_root, gaze_root, pattern_override=None):
    """Tasks with both maps and gaze, plus the reason for every skip."""
    saliency_root = Path(saliency_root)
    gaze_root = Path(gaze_root)
    if not saliency_root.is_dir():
        raise NotADirectoryError(f"saliency root not found: {saliency_root}")
    if not gaze_root.is_dir():
        raise NotADirectoryError(f"gaze root not found: {gaze_root}")

    tasks = []
    skipped = []
    for task_dir in sorted(saliency_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        task = task_dir.name

        maps_dir, pattern, n_maps = detect_map_layout(task_dir)
        if not n_maps:
            skipped.append((task, "no map images"))
            continue
        if pattern_override:
            pattern = pattern_override

        gaze_dir = gaze_root / task
        gaze_files = []
        if gaze_dir.is_dir():
            gaze_files = sorted(p for p in gaze_dir.glob("*.csv")
                                if not is_result_csv(p.name))
        if not gaze_files:
            skipped.append((task, "no gaze files"))
            continue

        tasks.append({
            "task": task,
            "maps": maps_dir,
            "pattern": pattern,
            "n_maps": n_maps,
            "gaze_dir": gaze_dir,
            "n_gaze": len(gaze_files),
        })
    return tasks, skipped


# --------------------------------------------------------------------------
# Running the two metric scripts
# --------------------------------------------------------------------------
def run_script(python, script, arguments, timeout):
    """Run one metric script; return (ok, tail_of_output)."""
    command = [python, str(script)] + [str(a) for a in arguments]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    output = result.stdout.decode("utf-8", "replace").strip().splitlines()
    return result.returncode == 0, "\n".join(output[-15:])


def metric_commands(task, args):
    """The argument lists for the NSS and the AUC run of one task."""
    common = [
        "--saliency_dir", task["maps"],
        "--pattern", task["pattern"],
        "--group", task["task"],
        "--coords", args.coords,
        "--kernel", args.kernel,
        "--sigma_deg", args.sigma_deg,
    ]
    if args.frame_col:
        common += ["--frame_col", args.frame_col]
    if args.x_col:
        common += ["--x_col", args.x_col]
    if args.y_col:
        common += ["--y_col", args.y_col]
    if args.limit:
        common += ["--limit", args.limit]

    nss = ["--gaze_dir", task["gaze_dir"],
           "--output_dir", Path(args.output_dir) / "nss"] + common
    auc = ["--gaze_dir", task["gaze_dir"],
           "--output_dir", Path(args.output_dir) / "auc"] + common
    return nss, auc


def already_done(args, task):
    nss_summary = Path(args.output_dir) / "nss" / task["task"] / (
        f"{task['task']}_nss_summary.csv")
    auc_summary = Path(args.output_dir) / "auc" / task["task"] / (
        f"{task['task']}_auc_summary.csv")
    return nss_summary.is_file() and auc_summary.is_file()


# --------------------------------------------------------------------------
# Aggregation - same logic and same columns as the UNISAL harness
# --------------------------------------------------------------------------
def collect_results(args, task):
    """Per-participant rows of one task, merged NSS + AUC by gaze file name."""
    task_name = task["task"]
    base = Path(args.output_dir)
    nss_rows = read_csv_rows(base / "nss" / task_name / f"{task_name}_nss_summary.csv")
    auc_rows = read_csv_rows(base / "auc" / task_name / f"{task_name}_auc_summary.csv")
    auc_pooled = read_csv_rows(base / "auc" / task_name
                               / f"{task_name}_auc_pooled_summary.csv")
    if not nss_rows:
        return None, None

    auc_by_file = {row["gaze_file"]: row for row in auc_rows}
    participants = []
    for row in nss_rows:
        name = row["gaze_file"]
        other = auc_by_file.get(name, {})
        participants.append({
            "task": task_name,
            "gaze_file": name,
            "rows_in_file": to_float(row.get("rows_in_file")),
            "rows_scored": to_float(row.get("rows_scored")),
            "mean_nss": to_float(row.get("mean_nss")),
            "std_nss": to_float(row.get("std_nss")),
            "min_nss": to_float(row.get("min_nss")),
            "max_nss": to_float(row.get("max_nss")),
            "mean_auc_j": to_float(other.get("mean_auc_j")),
            "std_auc_j": to_float(other.get("std_auc_j")),
            "min_auc_j": to_float(other.get("min_auc_j")),
            "max_auc_j": to_float(other.get("max_auc_j")),
            "skipped_no_map": to_float(row.get("skipped_no_map")),
            "skipped_out_of_bounds": to_float(row.get("skipped_out_of_bounds")),
            "skipped_missing_xy": to_float(row.get("skipped_missing_xy")),
        })
    pooled = auc_pooled[0] if auc_pooled else {}
    return participants, pooled


def summarise_task(task, participants, pooled):
    nss = [p["mean_nss"] for p in participants]
    auc = [p["mean_auc_j"] for p in participants]
    weights = [p["rows_scored"] for p in participants]
    return {
        "task": task["task"],
        "participants": len(participants),
        "gaze_frames": int(sum(w for w in weights if w)) if weights else 0,
        "mean_nss": mean(nss),
        "sd_nss": stdev(nss),
        "weighted_nss": weighted_mean(nss, weights),
        "mean_auc_j": mean(auc),
        "sd_auc_j": stdev(auc),
        "weighted_auc_j": weighted_mean(auc, weights),
        "pooled_auc_j": to_float(pooled.get("mean_auc_j")),
        "pooled_auc_sd": to_float(pooled.get("std_auc_j")),
        "pooled_frames": to_float(pooled.get("n_frames")),
    }


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary_table(model, task_rows, all_row):
    print()
    print("=" * 96)
    print(f"SUMMARY - {model} maps (macro mean = mean of participant means)")
    print("=" * 96)
    print(f"{'task':<18}{'subj':>5}{'frames':>8}{'NSS':>8}{'sd':>7}"
          f"{'AUC-J':>8}{'sd':>7}{'pooledAUC':>10}")
    print("-" * 96)
    for row in task_rows + [all_row]:
        label = row["task"] if row["task"] != "*" else "ALL (macro)"
        print(f"{label:<18}{row['participants']:>5}{row['gaze_frames']:>8}"
              f"{fmt(row['mean_nss'], 3):>8}{fmt(row['sd_nss'], 3):>7}"
              f"{fmt(row['mean_auc_j'], 3):>8}{fmt(row['sd_auc_j'], 3):>7}"
              f"{fmt(row['pooled_auc_j'], 3):>10}")
    print("-" * 96)


# --------------------------------------------------------------------------
# Self test - synthetic data, reads none of the study data
# --------------------------------------------------------------------------
def self_test():
    """Build a tiny synthetic study and run the whole chain on it."""
    import numpy as np

    root = PROJECT_ROOT / "_self_test_metrics"
    shutil.rmtree(root, ignore_errors=True)
    maps_root = root / "saliency_frames"
    gaze_root = root / "Resample_Output"
    out_dir = root / "metrics_output_object"

    ok = True

    def check(label, condition):
        nonlocal ok
        ok = ok and bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}")

    try:
        width, height, n_frames, n_subjects = 320, 180, 12, 2
        for task, offset in (("taskA", 0), ("taskB", 40)):
            task_maps = maps_root / task
            task_maps.mkdir(parents=True, exist_ok=True)
            for frame in range(1, n_frames + 1):
                image = np.zeros((height, width), np.uint8)
                # a moving bright blob the gaze will sit on
                cx = 60 + offset + frame * 6
                cy = 90
                image[cy - 25:cy + 25, cx - 25:cx + 25] = 255
                import cv2
                cv2.imwrite(str(task_maps / f"frame_{frame:04d}.png"), image)

            task_gaze = gaze_root / task
            task_gaze.mkdir(parents=True, exist_ok=True)
            for subject in range(1, n_subjects + 1):
                rows = ["frame,x,y"]
                for frame in range(1, n_frames + 1):
                    cx = (60 + offset + frame * 6) / width
                    cy = 90 / height
                    for dx, dy in ((0.0, 0.0), (0.02, 0.0), (0.0, 0.02)):
                        rows.append(f"{frame},{cx + dx:.6f},{cy + dy:.6f}")
                (task_gaze / f"P{subject:02d}_{task}.csv").write_text(
                    "\n".join(rows) + "\n", encoding="utf-8")

        print("Self test: map auto-detection, metric run, summary format")
        tasks, skipped = find_tasks(maps_root, gaze_root)
        check("found both synthetic tasks", len(tasks) == 2)
        check("nothing skipped", not skipped)
        check("map layout detected as flat <task>/*.png",
              all(t["maps"] == maps_root / t["task"] for t in tasks))
        check("4-digit pattern derived from the files",
              all(t["pattern"] == "frame_{frame:04d}.png" for t in tasks))

        python = find_python(None)
        check("an interpreter with pandas/numpy/cv2 was found", python is not None)
        if python is None:
            return 1

        args = argparse.Namespace(
            model="object", saliency_root=str(maps_root), gaze_root=str(gaze_root),
            output_dir=str(out_dir), pattern=None, only=None, limit=0,
            force=True, dry_run=False, python=python, kernel="gaussian",
            sigma_deg=1.0, coords="normalized", frame_col=None, x_col=None,
            y_col=None, timeout=600)

        result = main_from_args(args, tasks, skipped)
        check("driver reported success", result == 0)

        task_csv = out_dir / "summary_all_tasks.csv"
        part_csv = out_dir / "summary_all_participants.csv"
        check("summary_all_tasks.csv written", task_csv.is_file())
        check("summary_all_participants.csv written", part_csv.is_file())

        if task_csv.is_file():
            header = task_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
            check("task table has the UNISAL column list", header == TASK_COLUMNS)
            body = read_csv_rows(task_csv)
            check("one row per task plus the ALL row", len(body) == 3)
            check("ALL row is present", any(r["task"] == "*" for r in body))
            all_row = next((r for r in body if r["task"] == "*"), {})
            check("ALL row aggregates 4 participants",
                  to_float(all_row.get("participants")) == 4)
            nss_values = [to_float(r["mean_nss"]) for r in body if r["task"] != "*"]
            check("NSS is finite for every task",
                  all(v is not None for v in nss_values))
            auc_values = [to_float(r["mean_auc_j"]) for r in body if r["task"] != "*"]
            check("AUC-J is between 0 and 1",
                  all(v is not None and 0.0 <= v <= 1.0 for v in auc_values))

        if part_csv.is_file():
            header = part_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
            check("participant table has the UNISAL column list",
                  header == PARTICIPANT_COLUMNS)
            check("one row per participant", len(read_csv_rows(part_csv)) == 4)

        per_task = out_dir / "nss" / "taskA" / "taskA_nss_summary.csv"
        per_task_auc = out_dir / "auc" / "taskA" / "taskA_auc_summary.csv"
        check("per-task NSS summary in the UNISAL place", per_task.is_file())
        check("per-task AUC summary in the UNISAL place", per_task_auc.is_file())
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nSelf test:", "all checks passed" if ok else "FAILURES above")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # not `required=True`: --self_test must work without naming a model
    model = parser.add_mutually_exclusive_group(required=False)
    for name, text in (("object", "score saliency_frames (the object model)"),
                       ("face", "score saliency_frames_face"),
                       ("hand", "score saliency_frames_hand"),
                       ("social", "score saliency_frames_social"),
                       ("unisal", "score an external folder given by --saliency_root")):
        model.add_argument(f"--{name}", dest="model", action="store_const",
                           const=name, help=text)
    model.add_argument("--model", dest="model", choices=list(MODEL_ORDER),
                       help="same as the flags above, as a value")

    parser.add_argument("--saliency_root", default=None,
                        help="Map root. Defaults to the folder for --model; "
                             "required for --unisal.")
    parser.add_argument("--gaze_root", default=str(DEFAULT_GAZE_ROOT),
                        help=f"Root with one <task> folder of gaze CSVs "
                             f"(default: {DEFAULT_GAZE_ROOT})")
    parser.add_argument("--output_dir", default=None,
                        help="Where results go (default: metrics_output_<model>)")
    parser.add_argument("--pattern", default=None,
                        help="Map file name pattern (default: derived from the "
                             "first file found, e.g. frame_{frame:04d}.png)")

    parser.add_argument("--only", default=None,
                        help="Comma separated glob patterns of task names.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only the first N gaze files per task (quick check).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run tasks whose summaries already exist.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would run.")
    parser.add_argument("--python", default=None,
                        help="Interpreter with pandas for the metric scripts.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Seconds before one metric run is killed.")

    gaze = parser.add_argument_group("gaze model (keep these at the defaults)")
    gaze.add_argument("--kernel", default="gaussian",
                      choices=["gaussian", "disk", "point"])
    gaze.add_argument("--sigma_deg", type=float, default=1.0)
    gaze.add_argument("--coords", default="normalized",
                      choices=["normalized", "pixel"])
    gaze.add_argument("--frame_col", default=None)
    gaze.add_argument("--x_col", default=None)
    gaze.add_argument("--y_col", default=None)

    parser.add_argument("--self_test", action="store_true",
                        help="Run the synthetic end-to-end check and exit.")
    return parser


def resolve_roots(args):
    """Fill in --saliency_root and --output_dir from the model preset."""
    if args.saliency_root:
        args.saliency_root = str(Path(args.saliency_root).expanduser())
    elif args.model in MODEL_ROOTS:
        args.saliency_root = str(PROJECT_ROOT / MODEL_ROOTS[args.model])
    else:
        return (False, "--unisal needs --saliency_root pointing at the folder "
                       "that holds <task>/maps/*.png")
    if args.output_dir:
        args.output_dir = str(Path(args.output_dir).expanduser())
    else:
        args.output_dir = str(PROJECT_ROOT / f"metrics_output_{args.model}")
    args.gaze_root = str(Path(args.gaze_root).expanduser())
    return True, None


def main_from_args(args, tasks, skipped):
    """Run the metrics for the given tasks and write the two summaries."""
    failures = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['task']}: {task['n_maps']} maps "
              f"({task['pattern']}), {task['n_gaze']} gaze files")
        if args.dry_run:
            nss_args, _ = metric_commands(task, args)
            print(f"    nss: {NSS_SCRIPT.name} {' '.join(map(str, nss_args))}")
            continue
        if not args.force and already_done(args, task):
            print("    summaries already exist, reusing (use --force to redo)")
            continue

        for name, script, arguments in zip(
                ("NSS", "AUC-J"), (NSS_SCRIPT, AUC_SCRIPT), metric_commands(task, args)):
            ok, tail = run_script(args.python, script, arguments, args.timeout)
            print(f"    {name}: {'ok' if ok else 'FAILED'}")
            if not ok:
                failures.append((task["task"], name, tail))
                print("\n".join("      " + line for line in tail.splitlines()))

    if args.dry_run:
        return 0

    task_rows = []
    all_participants = []
    for task in tasks:
        participants, pooled = collect_results(args, task)
        if not participants:
            print(f"WARNING: no usable summaries for {task['task']}; skipped in "
                  "the aggregation")
            continue
        all_participants.extend(participants)
        task_rows.append(summarise_task(task, participants, pooled))

    if not task_rows:
        print("\nNo results to aggregate - every task failed.")
        return 1

    all_nss = [p["mean_nss"] for p in all_participants]
    all_auc = [p["mean_auc_j"] for p in all_participants]
    all_weights = [p["rows_scored"] for p in all_participants]
    all_row = {
        "task": "*",
        "participants": len(all_participants),
        "gaze_frames": int(sum(w for w in all_weights if w)),
        "mean_nss": mean(all_nss),
        "sd_nss": stdev(all_nss),
        "weighted_nss": weighted_mean(all_nss, all_weights),
        "mean_auc_j": mean(all_auc),
        "sd_auc_j": stdev(all_auc),
        "weighted_auc_j": weighted_mean(all_auc, all_weights),
        "pooled_auc_j": mean([row["pooled_auc_j"] for row in task_rows]),
        "pooled_auc_sd": None,
        "pooled_frames": sum(row["pooled_frames"] or 0 for row in task_rows),
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "summary_all_tasks.csv", task_rows + [all_row],
              TASK_COLUMNS)
    write_csv(output_dir / "summary_all_participants.csv", all_participants,
              PARTICIPANT_COLUMNS)

    print_summary_table(args.model, task_rows, all_row)
    print(f"Per task       : {output_dir / 'summary_all_tasks.csv'}")
    print(f"Per participant: {output_dir / 'summary_all_participants.csv'}")

    if skipped:
        print(f"\nSkipped {len(skipped)} task folder(s) without maps or gaze:")
        for task, reason in skipped[:15]:
            print(f"  {task}: {reason}")
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more")

    if failures:
        print(f"\n{len(failures)} metric run(s) failed:")
        for task_name, name, _ in failures:
            print(f"  {task_name} {name}")
        return 1
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.model:
        parser.error("name the model to score: --object, --face, --hand, "
                     "--social or --unisal (or use --model)")

    ok, problem = resolve_roots(args)
    if not ok:
        print(f"ERROR: {problem}")
        return 1

    print(f"Model          : {args.model}")
    print(f"Saliency root  : {args.saliency_root}")
    print(f"Gaze root      : {args.gaze_root}")
    print(f"Output root    : {args.output_dir}")
    print(f"Gaze model     : {args.kernel}, sigma {args.sigma_deg} deg, "
          f"{args.coords} coordinates")

    try:
        tasks, skipped = find_tasks(args.saliency_root, args.gaze_root, args.pattern)
    except NotADirectoryError as exc:
        print(f"ERROR: {exc}")
        return 1

    patterns = [p.strip() for p in (args.only or "").split(",") if p.strip()]
    if patterns:
        tasks = [t for t in tasks
                 if any(fnmatch(t["task"], p) for p in patterns)]

    print(f"Tasks to score : {len(tasks)}")
    print(f"Skipped        : {len(skipped)} (no maps or no gaze)")
    print()

    if not tasks:
        print("No task has both maps and gaze files. Check the two roots above.")
        if skipped:
            print("\nTasks seen:")
            for task, reason in skipped[:20]:
                print(f"  {task}: {reason}")
        return 1

    if not args.dry_run:
        python = find_python(args.python)
        if python is None:
            return 1
        args.python = python
        print(f"Metric python  : {python}")

    return main_from_args(args, tasks, skipped)


if __name__ == "__main__":
    sys.exit(main())
