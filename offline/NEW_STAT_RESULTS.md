# Larger new-game stat font: saved-image experiment, 11 November 2026

The new font is a better template candidate. Development glyphs are **11px high**,
compared with the old game's 7–8px. Keep the two game profiles separate: the old
offline geometry intentionally rejects the new font; production is unchanged.

Sources `resource/stat_new/1.png`–`3.png` supplied 24 development crops. Sources
`4.png`–`5.png` were labeled/evaluated only after freezing the policy. All 40
number-only crops, source screenshot hashes, coordinates, values and split roles
are in `tests/fixtures/stat_new/manifest.json`. Full desktop screenshots are not
needed for reproduction and are not included in the commit.

Both readers use cached OpenCV NCC features and score/margin checks. The larger
geometry was measured from development only. Whole templates cover 3–12;
digit templates 0–9 come from real development digits, including 0 from 10 and
1/2 from 11/12. Exact duplicate features are cached only once. Calibration uses
leave-one-**source**-out on the three development screenshots, also removing
pixel duplicates from that fold's training bank. It selects **score >= 0.97,
margin >= 0.12**, with 17 correct / 0 wrong / 7 UNKNOWN development fold reads.
No threshold was lowered after inspecting evaluation.

## Evaluation and timing

There are 16 raw evaluation crops, but **13 are pixel-identical to development
crops**. Only three genuinely different crop images remain: VIT=7, LCK=8, VIT=5.
Even these come from just two screenshots in one session and the same font/UI.
Repeated digits on different parchment backgrounds can have identical ink masks.
These results do not establish robustness to other sessions, fonts or animation.

| Reader | Correct / wrong / UNKNOWN, unique crops | Median / P95, per number |
|---|---:|---:|
| Existing fast OCR settings, unchanged | 0 / 1 / 2 | 1972.10 / 2605.01 ms |
| Whole-number templates | 2 / 0 / 1 | 1.116 / 2.115 ms |
| Digit-level templates | 2 / 0 / 1 | 1.475 / 4.936 ms |

All readers see the same decoded 24x18 RGB crops, three interleaved repetitions.
Times include preprocessing, actual OCR child startup/temp files, and child
counter instrumentation. Source decoding, capture, model construction/calibration
and development warm-up are excluded. P95 has only nine measurements per method.
The OCR settings were developed for the old game, not tuned for this font;
this comparison does not show that Tesseract itself cannot read the new font.

On the complete 16-crop screenshot replay, including duplicates, both template
readers return 15 correct / 0 wrong / 1 UNKNOWN. This **93.75% replay result is
not independent accuracy**. Do not replace the three-crop result with 16 samples.

Detailed unique-crop results:

* `4_VIT_7`: templates return 7, score 0.99999994; old OCR returns UNKNOWN.
* `4_LCK_8`: templates return UNKNOWN, score 0.9642812 below locked 0.97;
  old OCR also returns UNKNOWN. Keep this as an abstention regression.
* `5_VIT_5`: templates return 5, score 0.9820485; old fast OCR returns **9**.
  Its two matching preprocessing reads are not sufficient evidence for a new font.

Whole/digit banks each contain 11 cached features and use **16,896 bytes** for
template arrays, excluding Python/OpenCV/NumPy overhead. Over a separate 600-call
batch per reader, mean CPU is **0.625 / 0.807 ms per crop**. Batching avoids
misreporting sub-tick Windows CPU samples as zero cost. Full working-set/child
CPU details are in the local benchmark JSON, not an isolated total RAM claim.

## Decision

For this new fixed font, whole-number matching is the simpler promising choice:
same observed decisions as digit matching, fewer segmentation assumptions and
less CPU. Keep it offline for now. Do not transfer the old game's OCR settings,
ROI, window title, stat layout, conditions or input automation to this game.

Collect more independently captured frames, especially new 8 variants and rare
11/12, standalone 1/2/0 if they can actually occur, real animation frames, and
near-target complete stat panels. A new development 8 sample can inform tuning,
but new unseen evaluation frames must then be reserved. Do not lower 0.97 using
the observed 0.964 test score and claim the same old frames validate that change.
There are no target-positive/boundary panels or negative-scene benchmark in this
new dataset, so a production false-pass rate cannot be claimed.

## Reproduce without opening the game

```powershell
py -B -m unittest discover -s tests -p "test_new_stat_templates.py" -v
py -B -m offline.benchmark_new_stats run --repeats 3
```

Default output: `workflow_runs/stat_offline/new_game_20261111.json` (Git-ignored).
The shipped policy is frozen. `calibrate` is available for future development
datasets, but tuning after seeing results requires a new held-out dataset.
