Offline Talk experiment for mission_95
=====================================

Run from the repository root with the existing mission dependencies installed:

```powershell
python -B -m unittest discover -s tests -p "test_talk*.py" -v
python -B -m offline.benchmark_talk --repeats 3
```

The benchmark writes `workflow_runs/talk_offline/benchmark.json` (ignored by Git).
Use `--split holdout` for the independent recordings alone. It reads only saved
ROI fixtures and reference templates. It cannot capture a screen, send inputs,
or start the mission runner. Production code never imports this package.

Four comparison arms:

* `legacy`: current template bank, ROI and threshold unchanged.
* `legacy_expanded`: the same matcher with ROI `[280,170,1000,520]` to isolate
  the benefit of ROI expansion from scale recognition.
* `normalize`: current fast path, then HSV glyph components, bounded word
  proposals, aspect checks, and normalized glyph correlation.
* `coarse_to_fine`: current fast path, then half-resolution masked proposals
  over a scale bank and full-resolution local scale/position refinement.

Final acceptance stays at the live config threshold `0.82`. The coarse `0.45`
cutoff only proposes locations; it cannot accept a detection. Fallbacks reject
distinct passing labels and return no point when uncertain. Normalization caps
word proposals at 128; coarse refinement caps candidates at 8. Neither method
uses OCR. A passing offline proposal is not permission to click in a live run.

`tests/fixtures/talk_dynamic/manifest.json` records source runs, hashes, client
ROI, manual word boxes, and dataset splits. Raw full-screen logs are not needed
to reproduce this experiment. The expanded ROI extends down by 50px because
known recorded Talk reached y=484; it still excludes the bottom action bar.

Dataset limitations matter: loop16/loop81 and reference images already supplied
the live templates. Loop30 was development. The large stopped-run image already
informed the ROI, so it is **challenge regression**, never generalization.
Only one independently held-out readable Talk and ten held-out negative scenes
remain. Both prototypes miss that held-out Talk. No algorithm parameters were
changed after inspecting its prediction. More independent scale/background
examples are needed before selecting or enabling any live fallback.

The benchmark reports localized detections (point inside manually annotated
word), false positives, click error, CPU time, and processing median/p95/max.
Timer excludes file IO, constructor and capture. Each saved frame's fast-path
timing is shared across hybrid arms; reported hybrid processing = measured fast
time + measured exclusive fallback, including expanded crop/conversion.
Warm-up uses development only. Run tests separately from performance measurement.
Working set covers the whole benchmark process, not isolated OpenCV allocations.
Inspect the recorded hardware metadata when comparing hosts. Offline timings
exclude game capture and do not establish live reliability.
