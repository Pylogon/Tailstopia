"""Offline saved-ROI benchmark. No screen capture, hotkeys or input adapters.

python -B -m offline.benchmark_camp --repeats 100
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from offline.camp_checkpoint import HouseCheckpoint
from tailstopia_workflow.start_detection import StartDetector


ROOT = Path(__file__).resolve().parents[1]


def distribution(values):
    return {'median': round(float(np.median(values)), 6),
            'p95': round(float(np.percentile(values, 95)), 6)}


def summarize(rows):
    groups = {}
    for split in ('reference', 'development', 'evaluation'):
        cases = [r for r in rows if r['split'] == split and not r['development_duplicate']]
        for arm in ('raw', 'disk_mask'):
            sub = [r for r in cases if r['arm'] == arm]
            positives = [r for r in sub if r['scene'] == 'camp_open']
            negatives = [r for r in sub if r['scene'] != 'camp_open']
            # Camp menu is still Camp, but not a valid departure point.
            noncamp = [r for r in sub if not r['scene'].startswith('camp_')]
            missed = [r['source'] for r in positives if not r['joint']]
            groups[f'{split}/{arm}'] = {
                'n': len(sub), 'scenes': dict(Counter(r['scene'] for r in sub)),
                'house_positive_hits': sum(r['house'] for r in positives),
                'joint_positive_hits': sum(r['joint'] for r in positives),
                'positives': len(positives),
                'joint_false_positive': sum(r['joint'] for r in negatives),
                'negatives': len(negatives),
                'house_noncamp_false_positive': sum(r['house'] for r in noncamp),
                'noncamp': len(noncamp), 'joint_misses': missed,
                'house_positive_scores': distribution([r['score'] for r in positives]) if positives else None,
                'max_noncamp_score': max((r['score'] for r in noncamp), default=None),
                'house_wall_ms': distribution([t for r in sub for t in r['house_wall_ms']]),
                'joint_wall_ms': distribution([t for r in sub for t in r['joint_wall_ms']]),
                'joint_cpu_batch_average_ms': distribution([r['joint_cpu_batch_average_ms'] for r in sub]),
            }
    return groups


def working_set():
    if os.name != 'nt':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=counters.WorkingSetSize,
                peak_working_set_bytes=counters.PeakWorkingSetSize)


def run(root, repeats):
    corpus = root/'tests/fixtures/camp_checkpoint'
    manifest = json.loads((corpus/'manifest.json').read_text(encoding='utf8'))
    with Image.open(corpus/'house_reference.png') as ref:
        detectors = {'raw': HouseCheckpoint(ref, False), 'disk_mask': HouseCheckpoint(ref, True)}
    hud = StartDetector(root/'examples/step1.json')
    rows = []
    memory_before = working_set()
    for case in manifest['samples']:
        with Image.open(corpus/case['file']) as packed:
            house = packed.crop((0, 0, 50, 110)).convert('RGB')
            time_roi = packed.crop((50, 0, 93, 25)).convert('RGB')
        for arm, detector in detectors.items():
            result = detector.inspect(house)
            time_result = hud.inspect(time_roi)
            # Warm up; disk decoding, reference initialization and I/O excluded.
            house_ms, joint_ms = [], []
            batch_cpu = time.process_time_ns()
            for _ in range(repeats):
                begin = time.perf_counter_ns()
                detector.inspect(house)
                midpoint = time.perf_counter_ns()
                hud.inspect(time_roi)
                end = time.perf_counter_ns()
                house_ms.append((midpoint-begin)/1e6)
                joint_ms.append((end-begin)/1e6)
            cpu_average = (time.process_time_ns()-batch_cpu)/1e6/repeats
            rows.append(dict(case, arm=arm, house=result.found, score=result.score,
                             center=result.center, hud=time_result.status,
                             joint=result.found and time_result.status == 'READY',
                             house_wall_ms=house_ms, joint_wall_ms=joint_ms,
                             joint_cpu_batch_average_ms=cpu_average))
    return {'scope': 'Offline saved ROIs; no live accuracy or input-readiness claim',
            'opencv': cv2.__version__, 'platform': platform.platform(),
            'processor': platform.processor(), 'repeats': repeats,
            'memory_before': memory_before, 'memory_after': working_set(),
            'memory_note': 'Whole benchmark process with Python/Pillow/OpenCV and report buffers; not incremental live RAM',
            'cpu_note': 'Batch-average CPU per call; Windows process CPU clock is quantized, unlike perf_counter wall time',
            'timing_units': 'milliseconds; decoding/capture/logging excluded',
            'prototype_cache_bytes': sum(d.template.nbytes+d.mask.nbytes for d in detectors.values()),
            'summary': summarize(rows), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    result = run(args.root, args.repeats)
    output = args.output or args.root/'workflow_runs/camp_offline/benchmark.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result['summary'], indent=2))
    print('Report:', output.resolve())


if __name__ == '__main__':
    main()
