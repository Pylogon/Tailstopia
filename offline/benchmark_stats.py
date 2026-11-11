"""Calibrate on development, lock policy, then benchmark saved stat crops only."""
import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pytesseract

from offline.stat_templates import (ALGORITHM_VERSION, Case, Policy, Templates, calibrate,
                                    corpus_signature, load_corpus)
from offline.stat_benchmark_metrics import ChildMeter, host_details, process_memory_cpu
from offline import stat_templates


ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).with_name("stat_template_policy.json")


def detector_digest():
    # Remain reproducible across Git's LF/CRLF checkout settings on Windows.
    text = Path(stat_templates.__file__).read_text(encoding="utf-8-sig")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ocr(root):
    """Compile only pure OCR helpers; importing the live reroller would touch Win32."""
    path = Path(root) / "tailstopia_auto_reroll.py"
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8-sig"))
    allowed = {"prepared_ocr_image", "read_digit", "ocr_number_fallback", "ocr_number_fast",
               "ocr_number_robust", "target_met", "screenshot_stable"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in allowed]
    if {n.name for n in nodes} != allowed:
        raise ValueError("OCR helper layout changed; review offline adapter")
    centers = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "STAT_CENTERS" for t in n.targets))
    ns = {"Image": Image, "ImageOps": ImageOps, "re": re, "Counter": Counter,
          "pytesseract": pytesseract, "STAT_CENTERS": ast.literal_eval(centers)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "offline_ocr_helpers", "exec"), ns)
    executable = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    executable = str(executable) if executable.is_file() else shutil.which("tesseract")
    if not executable:
        raise RuntimeError("Tesseract is unavailable")
    pytesseract.pytesseract.tesseract_cmd = executable
    return ns, hashlib.sha256(raw).hexdigest(), ast.unparse(next(n for n in nodes if n.name == "target_met"))


def write_calibration(root, destination):
    development, _, _ = load_corpus(root)
    result = {"algorithm_version": ALGORITHM_VERSION,
              "detector_sha256": detector_digest(),
              "development_signature": corpus_signature(development), "methods": {},
              "selection": "zero false reads in development leave-one-image-out; maximize correct, then stricter policy",
              "caveat": "Retrospective, tiny two-source corpus; not an independent new-data validation of OCR."}
    for kind in ("whole", "digit"):
        policy, selected, candidates = calibrate(development, kind)
        model = Templates(development, kind, policy)
        result["methods"][kind] = dict(policy=asdict(policy), selected=selected, grid=candidates,
                                       supported=model.supported, training_failures=model.training_failures)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def distribution(values):
    if not values:
        return None
    return {key: round(float(value), 6) for key, value in zip(("median", "p95", "max"),
            (np.median(values), np.percentile(values, 95), max(values)))}


def fixture_inventory(root):
    development, evaluation, excluded = load_corpus(root)
    return {name: [dict(name=c.name, value=c.value, source=c.source, pixel_hash=c.pixel_hash)
                   for c in cases] for name, cases in
            (("development", development), ("evaluation", evaluation), ("excluded_duplicate", excluded))}


def measured(call):
    with ChildMeter() as children:
        parent_cpu = time.process_time()
        start = time.perf_counter()
        result = call()
        wall_ms = (time.perf_counter()-start)*1000
        parent_ms = (time.process_time()-parent_cpu)*1000
    child_values = [r["cpu_ms"] for r in children.records if r and r["cpu_ms"] is not None]
    child_cpu = sum(child_values) if len(child_values) == len(children.records) else None
    peaks = [r["peak_rss_bytes"] for r in children.records if r and r["peak_rss_bytes"] is not None]
    return result, dict(wall_ms=wall_ms, parent_cpu_ms=parent_ms,
                        child_cpu_ms=child_cpu,
                        total_cpu_ms=parent_ms+child_cpu if child_cpu is not None else None,
                        tesseract_processes=len(children.records), child_peak_rss_bytes=max(peaks, default=None))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["method"])].append(row)
    summaries = []
    for (split, method), group in sorted(groups.items()):
        positives = [r for r in group if r["truth"] is not None]
        negatives = [r for r in group if r["truth"] is None]
        correct = sum(r["value"] == r["truth"] for r in positives)
        unknown = sum(r["value"] is None for r in group)
        wrong = sum(r["value"] is not None and r["value"] != r["truth"] for r in group)
        summaries.append(dict(split=split, method=method, images=len(group), correct=correct,
            wrong=wrong, unknown=unknown, exact_accuracy=correct/len(positives) if positives else None,
            negative_false_accepts=sum(r["value"] is not None for r in negatives),
            negative_rejections=sum(r["value"] is None for r in negatives),
            unknown_rate=unknown/len(group), false_read_rate=wrong/len(group),
            processing_ms=distribution([t["wall_ms"] for r in group for t in r["times"]]),
            total_cpu_ms=distribution([t["total_cpu_ms"] for r in group for t in r["times"] if t["total_cpu_ms"] is not None]),
            parent_cpu_ms=distribution([t["parent_cpu_ms"] for r in group for t in r["times"]]),
            child_cpu_ms=distribution([t["child_cpu_ms"] for r in group for t in r["times"] if t["child_cpu_ms"] is not None]),
            tesseract_processes=sum(t["tesseract_processes"] for r in group for t in r["times"]),
            child_peak_rss_bytes=max((t["child_peak_rss_bytes"] for r in group for t in r["times"] if t["child_peak_rss_bytes"] is not None), default=None)))
    return summaries


def panel_diagnostic(root, rows, target_met):
    """Retrospective two-panel diagnostic, explicitly NOT an independent trial.

    Reuse the identical excluded CHA crop's existing prediction, without
    counting the duplicate as another evaluation image.
    """
    development, evaluation, excluded = load_corpus(root)
    all_cases = development + evaluation + excluded
    index = {(row["case"], row["method"]): row["value"] for row in rows
             if row["split"] in ("development_replay", "source_holdout")}
    methods = sorted({row["method"] for row in rows})
    result = []
    for source in sorted({c.source for c in all_cases}):
        source_cases = {int(c.name.rsplit("_", 2)[1]): c for c in all_cases if c.source == source}
        positions = {"INT": 4, "CHA": 5, "TAL": 6, "AGI": 2, "VIT": 3}
        truth = {stat: source_cases[i].value for stat, i in positions.items()}
        for method in methods:
            prediction = {}
            for stat, i in positions.items():
                case = source_cases[i]
                name = case.name
                if (name, method) not in index:
                    name = next(c.name for c in development if c.pixel_hash == case.pixel_hash)
                prediction[stat] = index[name, method]
            actual_pass = target_met(truth)
            predicted_pass = target_met(prediction) if all(v is not None for v in prediction.values()) else None
            result.append(dict(source=source, method=method, truth=truth, prediction=prediction,
                               actual_pass=actual_pass, predicted_pass=predicted_pass,
                               false_accept=bool(not actual_pass and predicted_pass is True)))
    return {"independent_panels": 0,
            "warning": "Only two historical negative panels, one training source and one with an excluded duplicate. No positive/boundary panels. Not a false-pass risk estimate.",
            "rows": result}


def template_cpu_batch(models, evaluation, cycles=200):
    """Amortize Windows' 15.625ms CPU clock; do not interpret 0ms as free CPU."""
    result = {}
    for kind, model in models.items():
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        count = 0
        for _ in range(cycles):
            for case in evaluation:
                model.read(case.image)
                count += 1
        cpu_ms = (time.process_time()-start_cpu)*1000
        wall_ms = (time.perf_counter()-start_wall)*1000
        result[kind] = dict(calls=count, total_cpu_ms=cpu_ms, mean_cpu_ms=cpu_ms/count,
                            mean_wall_ms=wall_ms/count)
    return result


def robust_audit(ocr, evaluation):
    """Supplemental candidate-reader diagnostic; one pass per saved holdout crop."""
    rows = []
    for case in evaluation:
        value, timing = measured(lambda: ocr["ocr_number_robust"](case.image))
        rows.append(dict(split="robust_supplement", method="ocr_robust", case=case.name,
                         truth=case.value, value=value, times=[timing]))
    return {"repeats": 1, "summary": summarize(rows), "rows": rows}


def run(root, policy_path, repeats, robust=False, stress=False, audit_robust=False):
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    cv2.setNumThreads(1)
    development, evaluation, excluded = load_corpus(root)
    locked_bytes = Path(policy_path).read_bytes()
    locked = json.loads(locked_bytes)
    if locked["algorithm_version"] != ALGORITHM_VERSION or locked["development_signature"] != corpus_signature(development):
        raise ValueError("Frozen development policy no longer matches corpus/algorithm")
    if locked["detector_sha256"] != detector_digest():
        raise ValueError("Detector changed after calibration; evaluation policy is invalid")
    models = {kind: Templates(development, kind, Policy(**locked["methods"][kind]["policy"]))
              for kind in ("whole", "digit")}
    ocr, source_hash, target = load_ocr(root)
    methods = {"ocr_fast": lambda im: ocr["ocr_number_fast"](im),
               **{kind: model.read for kind, model in models.items()}}
    if robust:
        methods["ocr_robust"] = lambda im: ocr["ocr_number_robust"](im)
    # Warm up only on development; exclude initialization, training and disk IO.
    for method in methods.values():
        method(development[0].image)
    setup_memory = process_memory_cpu()
    samples = [("development_replay", c, c.image) for c in development]
    samples += [("source_holdout", c, c.image) for c in evaluation]
    negatives = ROOT / "tests/fixtures/stat_template_negative"
    for row in json.loads((negatives / "manifest.json").read_text()):
        with Image.open(negatives / row["file"]) as im:
            im = im.convert("RGB").copy()
        digest = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        case = Case(row["file"], None, row["source"], digest, im)
        samples.append(("saved_negative", case, im))
    if stress:
        # Derived images are always separate stress rows, never independent data.
        for case in evaluation:
            samples += [("derived_blur_0.6", case, case.image.filter(ImageFilter.GaussianBlur(.6))),
                        ("derived_blur_1.0", case, case.image.filter(ImageFilter.GaussianBlur(1.0)))]
    rows = []
    for sample_index, (split, case, image) in enumerate(samples):
        records = {name: dict(times=[], predictions=[]) for name in methods}
        names = list(methods)
        for repeat in range(repeats):
            offset = (sample_index+repeat) % len(names)
            for name in names[offset:] + names[:offset]:
                result, timing = measured(lambda: methods[name](image))
                value = result.value if hasattr(result, "value") else result
                records[name]["predictions"].append(value)
                records[name]["times"].append(timing)
                if hasattr(result, "value"):
                    records[name]["match"] = asdict(result)
        for name, record in records.items():
            if len(set(record["predictions"])) != 1:
                raise RuntimeError("Non-deterministic prediction on the same saved image")
            rows.append(dict(split=split, method=name, case=case.name, truth=case.value,
                             value=record["predictions"][0], **record))
        print(f"Benchmarked {sample_index+1}/{len(samples)}: {split} / {case.name}", flush=True)
    holdout_support = {kind: {"supported": [], "unsupported": []} for kind in models}
    for kind, model in models.items():
        for case in evaluation:
            available = (case.value in model.supported if kind == "whole" else
                         all(int(d) in model.supported for d in str(case.value)))
            holdout_support[kind]["supported" if available else "unsupported"].append(case.name)
    return dict(host=host_details(), opencv=cv2.__version__, tesseract=str(pytesseract.get_tesseract_version()),
        repeats=repeats, source_sha256=source_hash, target_condition=target,
        policy_sha256=hashlib.sha256(locked_bytes).hexdigest(), policy=locked,
        source_split=dict(development=len(development), holdout=len(evaluation),
                          excluded_duplicate_names=[c.name for c in excluded]),
        fixture_inventory=fixture_inventory(root),
        memory=dict(shared_parent_after_setup=setup_memory, shared_parent_after_run=process_memory_cpu(),
                    template_bytes={k: m.template_bytes for k, m in models.items()},
                    caveat="Parent RSS is shared imports/caches, not isolated per-method RAM. Child peak is per Tesseract process."),
        support=holdout_support,
        target_diagnostic=panel_diagnostic(root, rows, ocr["target_met"]),
        template_batch_cpu=template_cpu_batch(models, evaluation),
        robust_audit=robust_audit(ocr, evaluation) if audit_robust else None,
        limitations=["Only two retrospective source images; holdout was previously used to develop baseline OCR.",
                     "No held-out real whole-number 1, 2, 4, 10 or new font/UI/scale/animation frames.",
                     "No independent complete five-stat panel or positive/boundary panels; two historical negative diagnostics cannot establish false-pass risk.",
                     "Derived blur is a stress diagnostic, not independent evidence or a real loading/animation frame.",
                     "Processing excludes capture, fixture decoding, training and warm-up; includes actual OCR temp-file IO, child startup and process-meter overhead."],
        summary=summarize(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--policy", type=Path, default=POLICY)
    bench = sub.add_parser("run")
    bench.add_argument("--policy", type=Path, default=POLICY)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--robust", action="store_true")
    bench.add_argument("--audit-robust", action="store_true",
                       help="Run one supplemental robust OCR pass per holdout crop")
    bench.add_argument("--stress", action="store_true")
    bench.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        result = write_calibration(args.root, args.policy)
        print(json.dumps({k: v["selected"] for k, v in result["methods"].items()}, indent=2))
    else:
        output = args.output or args.root / "workflow_runs/stat_offline/benchmark.json"
        result = run(args.root, args.policy, args.repeats, args.robust, args.stress, args.audit_robust)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print("Results:", output.resolve())


if __name__ == "__main__":
    main()
