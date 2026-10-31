# TailsTopia offline workflow

Added alongside the original recorder and reroller; neither original script is imported or modified.
There is no live input adapter, game connection, screenshot capture, or anti-cheat integration.
Python 3.10+ is required. Mock mode uses only the standard library. Recorded images require Pillow.

## Folder structure

```text
tailstopia_input_recorder.py       existing recorder, unchanged
tailstopia_auto_reroll.py          existing reroller, unchanged
tailstopia_workflow/
  config.py                      schema checks and config-relative paths
  macro_parser.py                existing recorder JSON/CSV -> ordered events
  input_adapter.py               mock input and release-all
  screen.py                      MockScreen / RecordedScreen
  reward_selector.py             priority + deterministic position + fallback
  control.py                     cooperative pause/resume/stop
  controller.py                  state transitions and retry boundaries
  logger.py                      streaming JSONL execution/action timeline
  __main__.py                    offline CLI
examples/
  workflow.json                  all coordinates, waits, policies, loop count
  macro_a.json / macro_b.csv      recorder-format sample fixtures
  recorded_smoke.json             image backend smoke demo using existing panel
 tests/test_workflow.py           unit and integration tests
workflow_runs/                    generated logs and failure PNGs
```

## Module development sequence

1. Parser/config: validate both macros before any action; preserve original event ordering.
2. Input/control: mock effects, release held inputs, interrupt waits, pause active deadlines.
3. Screen: scripted mock outcomes, then recorded frames with fixed ROI template comparison.
4. Reward selector: rank recognized names, tie-break by Y then X, explicit fallback or fail closed.
5. Controller/logger: run ordered states, retry observations only, stop with image evidence.
6. Integration: happy path and failure injection; validate actual start/stage/complete templates later.

Steps 1-6 are implemented for offline use. Actual workflow screenshots and calibrated ROIs are
still needed before recorded-frame checks can establish the semantics of the real workflow.

## State contract / transition table

A state is entered only after its predecessor succeeds. Each entry gets a fresh timeout.
The normal route follows this table; any exception or expired deadline goes to safe stop.
`retries` means additional observations, so retries=2 allows at most three checks.

| State | Entry condition | Expected result / next state | Retry and failure policy |
|---|---|---|---|
| VERIFY_START | Initial loop or previous return verified | Start template matches -> PLAY_MACRO_A | Observation retry; fail on budget/deadline |
| PLAY_MACRO_A | Start verified | All A events dispatched -> ENTER_STAGE | Never replay on failure |
| ENTER_STAGE | A dispatched | Four configured clicks and waits finish -> VERIFY_STAGE | Never replay on failure |
| VERIFY_STAGE | Enter commands dispatched | Stage template matches -> PLAY_MACRO_B | Observation retry only |
| PLAY_MACRO_B | Stage verified | All B events dispatched -> VERIFY_COMPLETE | Never replay on failure |
| VERIFY_COMPLETE | B dispatched | Complete template matches -> SCAN_REWARD | Observation retry only |
| SCAN_REWARD | Completion verified | Valid name/X/Y list, including empty -> SELECT_REWARD | Recognition exception fails; no scan retry |
| SELECT_REWARD | Scan succeeded | Priority/fallback click dispatched -> RETURN_START | Missing fallback fails; never replay click |
| RETURN_START | Reward selection dispatched | Return commands dispatched, start template matches -> VERIFY_START or finish | Retry start observation only; never replay return |

Every state uses `states.STATE.timeout` in seconds; the sample sets 10 seconds each.
`retry_delay` and `retries` are configured per state; they are used only by observations,
including the start check inside RETURN_START. Action states do not consume retry settings.
RETURN_START verifies return even on the final loop. A loop counts as complete only after this check.
The final run result is SUCCESS or FAIL; observation attempts emit RETRY with the reason.
KeyboardInterrupt and explicit Stop produce FAIL and safe stop.

## Run

From the project directory in PowerShell:

```powershell
python -m unittest discover -s tests -v
python -m tailstopia_workflow examples/workflow.json
python -m tailstopia_workflow examples/workflow.json --realtime --interactive
python -m tailstopia_workflow examples/recorded_smoke.json
```

Use `py` instead of `python` if that is your installed Windows launcher.
Install the optional recorded-image dependency if needed: `py -m pip install Pillow`.
Tests using Pillow are explicitly skipped if it is absent; the mock suite still runs.

Expected tests: 16 pass when Pillow is installed. The mock demo returns:

```text
SUCCESS: Loop limit reached; return verified; loops=2; state=RETURN_START
```

The recorded smoke demo returns SUCCESS with loops=1. It deliberately compares the existing
stat panel against itself for all image states. This tests image loading/ROI integration ONLY;
it does not demonstrate correct start, stage, completion, or reward recognition.
The automated recorded-image tests also use synthetic fixtures, including mismatch and debug-image checks.
The OCR-fail test injects a recognition exception; no OCR engine is installed or invoked by this package.

In interactive mode type `p`, `r`, or `s` followed by Enter for pause/resume/stop.
These are console commands, not global hotkeys. Ctrl+C also requests safe stop.
Programmatic use: pass a shared `Control` to `Controller` and call pause/resume/stop from another thread.
The CLI starts a command thread only when requested. Normal runs are single-threaded.
Pause preserves the exact position in the sequence and freezes active deadlines; it does not restart a state.
A paused mock hold remains in the mock adapter until resume/release or stop. Stop clears all mock holds.

## Recordings and timing

Point `macro_a` and `macro_b` at your recorder's JSON/CSV files. Paths resolve relative to the config,
not the shell's working directory. Both files are loaded and validated before dispatch begins.
The current repository contained no real recording files, so samples follow the recorder's save format.

`key_hold`/`mouse_hold` are summary rows, not independent actions. Ordinary summaries are ignored;
when ended_by=pause/stop closes a still-held input, the parser synthesizes an up at that timestamp.
Overlapping key/mouse events retain their timestamps and ordering; they are not converted to serial holds.
Malformed timing, unknown types, duplicate downs, unmatched ups, and unclosed holds fail preflight.
Hold-summary-only external formats are not supported.

Fast dry-run advances a virtual time offset for recorded gaps and configured waits, checking deadlines
without sleeping. Realtime honors those gaps with interruptible waits. Neither mode sends OS inputs.
Action timestamps are absolute offsets within each macro, with wait deltas between adjacent events.
The recorder's trailing active_duration_seconds after its last event is metadata, not a replay action;
put any required trailing delay in config actions. Positions use one fixed screenshot coordinate space;
there is no automatic scaling or window offset conversion.

## Recorded screenshot configuration

Use `screen.backend = "recorded"`. See recorded_smoke.json for the complete schema.
Replace each of frames.start/stage/complete/rewards with actual recorded image paths.
A list advances once per observation and repeats its last frame after exhaustion; it is a replay fixture,
not a source of fresh live frames. RETURN_START consumes frames.start too.

For each start/stage/complete check, set template, roi=[left,top,right,bottom], and max_error.
The ROI and template must have the same pixel dimensions. Images convert to RGB, then compare
normalized mean absolute channel difference; max_error=0 is exact equality. Default is 0.02.
Reject out-of-bounds ROI and inconsistent sizes. Crop tight, stable UI anchors and calibrate with
both positive and negative frames; matching a template is not proof of semantic screen correctness.

For reward detection, add screen.rewards entries:

```json
{"name":"rare", "template":"rare.png", "roi":[20,30,44,54],
 "max_error":0.02, "x":32, "y":42}
```

The ROI is where a candidate icon is checked; X/Y is the configured selection position.
Multiple entries can describe different slots. Reward scan emits detected item names and positions.
No OCR, full-screen sliding search, or per-frame external process is needed.
Empty/unknown rewards use reward.fallback; set it to null to stop instead.

## Logs, safe stop and limits

execution.jsonl records UTC timestamps, run ID, loop/state, action index and macro offset,
state entry, SUCCESS/RETRY/FAIL with reason, active/virtual elapsed time and actual wall elapsed time.
Reward scans/selections and release-all are logged. The file is appended across runs; group by run_id.
No full timeline is accumulated in RAM; memory is bounded by loaded recordings/configured frame lists.
Large recordings still load into memory; split them before running on a 4GB machine.

On failure the controller stops, clears mock held inputs, and saves a uniquely named failure PNG.
RecordedScreen saves the last loaded frame. MockScreen saves a marked-in-log synthetic 1x1 PNG;
it is not desktop evidence. If no recorded frame has been read, or image writing fails, debug_error
records why evidence is unavailable. No later action state runs after failure.
Timeouts are cooperative, checked around recognition/input calls and during interruptible waits.
They cannot forcibly cancel a backend that blocks inside one call. Supplied backends are local/offline;
future blocking backends require their own bounded call timeout. Disk/logging failures can prevent
complete diagnostics and are not hidden. This is not a live-game reliability certification.
