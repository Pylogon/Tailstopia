# Stat reader experiment — 11 November 2026

**Recommendation: keep the current production OCR.** Both template prototypes
are much cheaper per read, but accept only 1/7 held-out saved number crops.
They are not ready to drive rerolls. This experiment changes no production
flow, condition, configuration, capture timing, candidate confirmation or input.

## Reproduce offline

Run from the repository root with the existing OpenCV/Pillow/Tesseract environment:

```powershell
py -B -m unittest discover -s tests -p "test_stat_templates_offline.py" -v
py -B -m offline.benchmark_stats run --repeats 3 --stress --audit-robust
```

The script uses saved pixels only. Its default result is
`workflow_runs/stat_offline/benchmark.json` (ignored by Git). `--robust` adds
robust OCR to *every* comparison instead; `--audit-robust` is a smaller,
one-pass supplemental audit of the seven held-out numbers. All templates are
automatically built and cached in RAM from the development crops.

The shipped `stat_template_policy.json` is already frozen. To calibrate a new
development dataset, use `py -B -m offline.benchmark_stats calibrate`. Once
evaluation results have been inspected, further algorithm/threshold tuning
requires a **new evaluation set**. Do not report the old set as new-data validation.

## Data, split and calibration

* Existing corpus: 16 crops, only two historical source screenshots; 15 unique
  pixel images. Source `historical_local` supplies eight development crops.
* Source `failure_20261104` supplies seven evaluation crops after excluding
  `failure_20261104_5_7.png`, which is pixel-identical to a development image.
* These seven are source-disjoint for **this template experiment**. They were
  already used to repair production OCR in November 2026, so this is retrospective
  replay, not prospective generalization evidence for OCR. One screenshot also
  cannot stand in for seven independent sessions.
* Seven saved negative crops contain actual panel labels, icons and background.
  They come from the same failure panel; none becomes a template.
* Blur stress uses Gaussian blur sigma 0.6/1.0 applied to real evaluation crops.
  No new numbers or fonts are rendered. Stress rows are separate, correlated,
  derived diagnostics, not real animation recordings or extra independent data.
* Development-only leave-one-image-out searches score 0.80–0.99 and margin
  0.03–0.12, requiring zero wrong development reads, then maximizing correct
  reads and choosing the stricter tied policy. Both select **score >= 0.80,
  margin >= 0.12**. Leave-one-out results are 2 correct, 6 UNKNOWN, 0 wrong.
  There are only two duplicated labels to help calibrate variation; evidence
  for a generally suitable threshold is weak. Neither policy was changed
  after evaluating the held-out source.
* Policy stores development pixel signatures and detector source hash. A changed
  bank/detector invalidates the frozen policy. Production thresholds are unrelated
  and remain unchanged.

## Methods

The baseline compiles only the current `prepared_ocr_image`, `read_digit`, fast,
fallback and robust helper functions via AST. It never imports the live reroller
or initializes Win32 input. The original fast reader enlarges 8x, binarizes at
140/145, optionally retries 135, then requires agreement between normalized
grayscale and higher-contrast binary recovery. Robust confirmation retains
its four-vote/conflict rule. The old failing VIT=11 now reads correctly, but
needs five Tesseract processes per fast read instead of the usual two. Robust
reading uses six, or twelve when recovery is necessary.

Whole-number matching uses tight binary ink masks, aspect checks, normalized
16x24 features and `TM_CCOEFF_NORMED`, ranking distinct labels. Digit matching
uses weak column valleys before matching and reassembling one/two digits.
The saved `12` is one connected component: column analysis is needed to split it.
Both reject missing/invalid geometry, insufficient scores, close competing labels,
leading zeros and values outside the baseline's 0–30 acceptance range.

## Measured results

Hardware: Intel i3-6006U, four logical CPUs, Windows 10, Python 3.13.17,
OpenCV 5.0.0, Tesseract 5.4.0.20240606. Background load was not controlled.
Each primary timing distribution has 21 observations: seven same saved images,
three repetitions per method, interleaved method order. Preprocessing and actual
Tesseract child startup are included; capture, disk IO, training and warm-up
are excluded. Here disk IO means loading/decoding the saved fixtures; Tesseract's
own temporary-image file IO remains inside its measured call. Repetitions do not
increase the independent accuracy sample size.

| Method | Exact match | Wrong reads | UNKNOWN | Median / P95 per number |
|---|---:|---:|---:|---:|
| Current fast OCR | 7/7 (100%) | 0/7 | 0/7 | 1207.08 / 2475.50 ms |
| Whole-number template | 1/7 (14.3%) | 0/7 | 6/7 (85.7%) | 0.893 / 1.429 ms |
| Digit-level template | 1/7 (14.3%) | 0/7 | 6/7 (85.7%) | 0.803 / 1.514 ms |

Both templates read only the held-out `7`. They also reject supported labels
`3`, `5`, `6`, `11`, so missing classes alone do not explain their poor coverage.
For example the held-out `5` scores 0.694 with only a 0.039 margin; whole `11`
scores 0.376, while the digit reader rejects a segment at 0.781. Tiny changes in
stroke rasterization/binarization matter. Training-image replay is 8/8 for both
models, which **must not be cited as held-out accuracy**.

All methods reject all seven saved negative crops: observed nonnumeric false
accepts 0/7. This tiny, correlated sample does not establish a zero real-world rate.

| Derived stress | OCR correct / wrong / UNKNOWN | Whole | Digit |
|---|---:|---:|---:|
| Blur sigma 0.6 | 6 / 0 / 1 | 1 / 0 / 6 | 1 / 0 / 6 |
| Blur sigma 1.0 | 2 / 0 / 5 | 0 / 0 / 7 | 0 / 0 / 7 |

Supplemental robust OCR reads 7/7 correctly in one pass per image, median
3595.32 ms and P95 14067.98 ms (max 17772.38 ms). This is just seven samples
and includes a costly recovery; it is not a stable live tail-latency estimate.
It measures one number read, not the existing two-fresh-screen candidate protocol.
Agreement between preprocessing variants of the same crop is not independent
ground truth; consistent font-specific OCR errors still need new labeled evidence.

## CPU and memory

Fast OCR's held-out CPU time, including child processes, is median 875 ms,
P95 2171.88 ms and mean 1069.20 ms per number. The 21 reads launch 51 Tesseract
processes. Counting Python CPU alone would omit most OCR work.

Windows CPU-clock granularity makes many sub-millisecond template samples appear
as 0 ms. A separate batch of 1400 calls per template model, over the same seven
saved crops, measures mean CPU **0.636 ms whole / 0.703 ms digit**. These are
batch means, not per-image CPU medians or proof of zero cost. No processes are
spawned by either template reader.

Template arrays occupy 12,288 bytes (whole) and 15,360 bytes (digit). These sizes
exclude Python/OpenCV/NumPy overhead. Shared benchmark parent peak working set
is about 39.71 MiB; it includes all imported methods and cannot be assigned to
one reader. Maximum Tesseract child peak working set on clean holdout reads is
about 27.41 MiB per process. The experiment does not establish isolated total
RAM savings for a future production integration.

## False passes and difficult pairs

There are no new, independent complete five-stat panels near the current target.
Both historical panels fail the actual target function; the diagnostic finds
no false pass for any method, but one panel trained the templates and the other
contains an excluded duplicate. Its independent sample count is **zero**.
Do not interpret this as a measured production false-pass rate.

* `1/7`: real standalone held-out `1` is absent. Digit `1` templates are genuine
  segments of development `11/12`, not independently collected single `1` values.
* `3/8`: OCR reads both held-out values; templates return UNKNOWN for both.
  No development `8` template exists.
* `5/6`: OCR reads both; both template readers return UNKNOWN despite having
  development examples of both labels.
* `11/1`: OCR recovery reads held-out `11`; templates reject it. Unit tests
  check two-segment preservation, but no independent single `1` is available.
* Real animation, changed fonts, new UI sizes and new DPI are not validated.
  Normalization can translate/resize only within its strict measured geometry;
  no broad font/scale robustness claim is supported.

## Needed data and production decision

Whole templates exist only for `3,5,6,7,11,12`. Digit templates exist only for
`1,2,3,5,6,7`; missing digit templates are **0,4,8,9**. Collect real standalone
`1,2,4,8,9,10`, and `0` if the game can show it. Other 13–30 values require real
examples only if they actually occur; never fabricate them to fill coverage.
Collect additional `3,5,6,7,11,12` on both panel columns and different rolls.
Keep entire source screenshots in one split before cropping, remove duplicates,
and reserve new sessions for evaluation before any tuning.

Also collect real consecutive frames during reroll animation, focus/UI/DPI
changes, and complete target panels on both sides of the actual boundary:
CHA/AGI just below/equal/above their configured values, sums just below/equal/above
the configured total, including true accepted candidates. Manually label them.

Keep production OCR now. A future hybrid may be worth a **shadow-only** comparison
once coverage and new-source evidence improve, retaining OCR for UNKNOWN and the
existing candidate verification. Confirming accepted candidates alone does not
protect against a bad template read that causes a good roll to be discarded.
Extra template inventories, segmentation, thresholds and font/scale maintenance
are not justified by the present 1/7 acceptance rate, despite the clear speed gap.
