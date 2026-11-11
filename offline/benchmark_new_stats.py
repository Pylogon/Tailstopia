"""Saved new-game images only: calibrate first, evaluate after policy freeze."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2

from offline import new_stat_templates, stat_templates
from offline.new_stat_templates import NativeTemplates, calibrate_sources, load_samples
from offline.stat_templates import Policy
from offline.benchmark_stats import (distribution, load_ocr, measured, summarize,
                                      template_cpu_batch)
from offline.stat_benchmark_metrics import host_details, process_memory_cpu


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/"tests/fixtures/stat_new"
POLICY = Path(__file__).with_name("new_stat_policy.json")


def code_hash():
    return hashlib.sha256("\n".join(Path(m.__file__).read_text(encoding="utf-8-sig")
                                  for m in (new_stat_templates, stat_templates)).encode()).hexdigest()


def development_hash(samples):
    rows = sorted((s.name, s.value, s.source, s.pixel_hash) for s in samples)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def calibrate(folder, policy_path):
    development = [s for s in load_samples(folder) if s.split == "development"]
    result = dict(detector_sha256=code_hash(), development_signature=development_hash(development), methods={})
    for kind in ("whole", "digit"):
        selected, grid = calibrate_sources(development, kind)
        model = NativeTemplates(development, kind, Policy(selected["threshold"], selected["margin"]))
        result["methods"][kind] = dict(selected=selected, grid=grid, supported=model.supported,
                                      training_failures=model.training_failures)
    Path(policy_path).write_text(json.dumps(result, indent=2))
    return result


def run(root, corpus, policy_path, repeats):
    cv2.setNumThreads(1)
    samples = load_samples(corpus)
    dev = [s for s in samples if s.split == "development"]
    raw = [s for s in samples if s.split == "evaluation"]
    frozen = json.loads(Path(policy_path).read_text())
    if frozen["detector_sha256"] != code_hash() or frozen["development_signature"] != development_hash(dev):
        raise ValueError("Frozen algorithm/development signature changed")
    seen = {s.pixel_hash for s in dev}
    evaluation, excluded = [], []
    for sample in raw:
        if sample.pixel_hash in seen:
            excluded.append(sample.name)
        else:
            seen.add(sample.pixel_hash)
            evaluation.append(sample)
    models = {kind: NativeTemplates(dev, kind, Policy(frozen["methods"][kind]["selected"]["threshold"],
                                                    frozen["methods"][kind]["selected"]["margin"]))
              for kind in ("whole", "digit")}
    ocr, source_hash, target = load_ocr(root)
    methods = {"ocr_fast": ocr["ocr_number_fast"], **{kind: model.read for kind, model in models.items()}}
    for method in methods.values():
        method(dev[0].image)
    rows = []
    for i, sample in enumerate(evaluation):
        record = {kind: dict(times=[], predictions=[]) for kind in methods}
        names = list(methods)
        for repeat in range(repeats):
            offset = (i+repeat) % len(names)
            for kind in names[offset:]+names[:offset]:
                result, timing = measured(lambda: methods[kind](sample.image))
                value = result.value if hasattr(result, "value") else result
                record[kind]["times"].append(timing)
                record[kind]["predictions"].append(value)
                if hasattr(result, "value"):
                    record[kind]["match"] = asdict(result)
        for kind, r in record.items():
            if len(set(r["predictions"])) != 1:
                raise ValueError("Saved-image reads are not deterministic")
            rows.append(dict(split="source_holdout_new_game", method=kind, case=sample.name,
                             truth=sample.value, value=r["predictions"][0], **r))
        print(f"Benchmarked {i+1}/{len(evaluation)}: {sample.name}", flush=True)
    # Also report reused/duplicate screenshot replay, without counting it as holdout.
    replay = {kind: [dict(case=s.name, truth=s.value, value=model.read(s.image).value)
                     for s in raw] for kind, model in models.items()}
    glyph_overlap = {}
    for kind, model in models.items():
        glyph_overlap[kind] = dict(template_entries=len(model.bank), template_bytes=model.template_bytes)
    return dict(host=host_details(), frozen_policy=frozen, production_sha256=source_hash,
                raw_evaluation_crops=len(raw), unique_evaluation_crops=len(evaluation),
                excluded_pixel_duplicates=excluded, development_crops=len(dev),
                summary=summarize(rows), rows=rows, duplicate_including_replay=replay,
                models=glyph_overlap, template_batch_cpu=template_cpu_batch(models, evaluation),
                shared_parent_memory=process_memory_cpu(),
                caveats=["Only two same-session evaluation screenshots; static UI/font, no animation or new size.",
                         "Repeated glyphs on different background crops can share exactly the same binary shape.",
                         "No target-positive/boundary stat panels; not a production false-pass validation.",
                         "Processing includes OCR temp files/child startup, excludes fixture decoding and capture."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    if args.command == "calibrate":
        result = calibrate(args.corpus, args.policy)
        print(json.dumps({k: v["selected"] for k, v in result["methods"].items()}, indent=2))
    else:
        result = run(args.root, args.corpus, args.policy, args.repeats)
        output = args.output or args.root/"workflow_runs/stat_offline/new_game_20261111.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summary"], indent=2))
        print("Output:", output)


if __name__ == "__main__":
    main()
