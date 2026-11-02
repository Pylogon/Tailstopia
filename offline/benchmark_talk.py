"""Run with python -B -m offline.benchmark_talk --repeats 3. No game access."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from offline.talk_detection import TalkExperiment
from tailstopia_workflow.mission_config import load_mission_config

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'tests/fixtures/talk_dynamic'


def load_client(case, corpus=CORPUS, roi=(280,170,1000,520)):
    path=corpus/case['image']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=case['image_sha256']:
        raise ValueError('Fixture hash mismatch: '+case['id'])
    with Image.open(path) as crop:
        if crop.size!=(roi[2]-roi[0],roi[3]-roi[1]):raise ValueError('Wrong fixture dimensions')
        # All pixels inspected by any method are recorded, not synthesized.
        client=Image.new('RGB',(1280,600))
        client.paste(crop,roi[:2])
        return client


def measured(call):
    cpu=time.process_time();wall=time.perf_counter()
    result=call()
    return result, (time.perf_counter()-wall)*1000, (time.process_time()-cpu)*1000


def distribution(values):
    return {k:round(float(v),3) for k,v in zip(('median','p95','max'),
            (np.median(values),np.percentile(values,95),max(values)))}


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['split'],row['method'])].append(row)
    result=[]
    for (split,method),items in sorted(groups.items()):
        pos=[r for r in items if r['talk_box'] is not None]
        neg=[r for r in items if r['talk_box'] is None]
        localized=sum(r['localized'] for r in pos)
        result.append(dict(split=split,method=method,positives=len(pos),negatives=len(neg),
            detected=sum(r['result']['found'] for r in pos),localized=localized,
            detection_rate=localized/len(pos) if pos else None,
            false_positives=sum(r['result']['found'] for r in neg),
            false_positive_rate=sum(r['result']['found'] for r in neg)/len(neg) if neg else None,
            processing_ms=distribution([v for r in items for v in r['processing_ms']]),
            fallback_only_ms=distribution([v for r in items for v in r['fallback_only_ms']]),
            cpu_ms=distribution([v for r in items for v in r['cpu_ms']]),
            max_click_error_px=max((r['click_error_px'] for r in pos if r['localized']),default=None)))
    return result


def working_set():
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_bytes':counters.WorkingSetSize,'peak_working_set_bytes':counters.PeakWorkingSetSize}


def host_details():
    details={}
    if os.name=='nt':
        import ctypes
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            details['processor_name']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[
                (name,ctypes.c_ulonglong) for name in ('total','available','total_page',
                 'available_page','total_virtual','available_virtual','extended')]
        memory=Memory();memory.length=ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            details['physical_ram_bytes']=memory.total
    return details


def run(repeats, split=None):
    if repeats<1:raise ValueError('repeats must be positive')
    manifest=json.loads((CORPUS/'manifest.json').read_text(encoding='utf-8'))
    config,base=load_mission_config(ROOT/'examples/mission_95.json')
    detector=TalkExperiment(config,base,manifest['roi'])
    if detector.threshold!=manifest['threshold']:raise ValueError('Threshold changed; benchmark invalid')
    # Warm up on development only. Timer excludes file IO and constructor.
    dev=next(c for c in manifest['cases'] if c['id']=='loop30')
    with load_client(dev) as image:
        detector.baseline(image)
        rgb=detector._frame(image)
        detector.normalize(rgb);detector.coarse_to_fine(rgb)
    memory_before=working_set()
    rows=[]
    for case in manifest['cases']:
        if split and case['split']!=split:continue
        records={name:dict(processing_ms=[],fallback_only_ms=[],cpu_ms=[]) for name in
                 ('legacy','legacy_expanded','normalize','coarse_to_fine')}
        with load_client(case) as image:
            for _ in range(repeats):
                fast,fast_ms,fast_cpu=measured(lambda:detector.baseline(image))
                expanded,expanded_ms,expanded_cpu=measured(lambda:detector.baseline(image,True))
                for name,value,ms,cpu in [('legacy',fast,fast_ms,fast_cpu),
                                          ('legacy_expanded',expanded,expanded_ms,expanded_cpu)]:
                    records[name]['result']=asdict(value)
                    records[name]['processing_ms'].append(ms);records[name]['cpu_ms'].append(cpu)
                    records[name]['fallback_only_ms'].append(0)
                for name in ('normalize','coarse_to_fine'):
                    if fast.found:
                        value,extra_ms,extra_cpu=fast,0,0
                    else:
                        def fallback():
                            rgb=detector._frame(image)
                            return detector.normalize(rgb) if name=='normalize' else detector.coarse_to_fine(rgb)
                        value,extra_ms,extra_cpu=measured(fallback)
                    records[name]['result']=asdict(value)
                    records[name]['processing_ms'].append(fast_ms+extra_ms)
                    records[name]['fallback_only_ms'].append(extra_ms)
                    records[name]['cpu_ms'].append(fast_cpu+extra_cpu)
        for method,record in records.items():
            box=case['talk_box'];point=record['result']['point'];localized=False;error=None
            if box and point:
                localized=box[0]<=point[0]<box[2] and box[1]<=point[1]<box[3]
                error=round(float(np.hypot(point[0]-(box[0]+box[2])/2,point[1]-(box[1]+box[3])/2)),3)
            rows.append(dict(id=case['id'],split=case['split'],method=method,talk_box=box,
                             localized=localized,click_error_px=error,**record))
        print(case['id'],[(k,v['result']['found'],round(v['result']['score'],4),v['result']['point'])
                          for k,v in records.items()],flush=True)
    return dict(source_commit=manifest['source_commit'],
                algorithm_sha256=hashlib.sha256((ROOT/'offline/talk_detection.py').read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256((CORPUS/'manifest.json').read_bytes()).hexdigest(),
                environment=dict(python=platform.python_version(),opencv=cv2.__version__,
                    platform=platform.platform(),processor=os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),
                    logical_cpus=os.cpu_count(),opencv_threads=cv2.getNumThreads(),**host_details()),
                repeats=repeats,roi=manifest['roi'],threshold=detector.threshold,
                timing_policy='File IO, construction and capture excluded. Warmed development frame. '
                    'Hybrid total = measured shared fast path + measured exclusive fallback; '
                    'not four simultaneous detectors. Sequential, one OpenCV thread.',
                memory_policy='Working set is entire benchmark process, including Python/Pillow/OpenCV, '
                    'not isolated fallback allocation or target HP RAM measurement.',
                prototype_cache_bytes=detector.cache_bytes,memory_before=memory_before,
                memory_after=working_set(),summary=summarize(rows),cases=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--split',choices=('development','holdout','challenge_regression',
                                          'template_regression','negative_regression'))
    parser.add_argument('--output',type=Path,default=ROOT/'workflow_runs/talk_offline/benchmark.json')
    args=parser.parse_args()
    report=run(args.repeats,args.split)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Report:',args.output.resolve(),flush=True)
    for row in report['summary']:print(json.dumps(row),flush=True)


if __name__=='__main__':main()
