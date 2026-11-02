"""Offline controller simulation; never captures a screen or sends OS input.

python -B -m offline.benchmark_return_polling

Logged HUD labels are sparse observations, not a continuous recording. Build
two piecewise-constant scenarios: each observed change happens just after the
previous sample (early), or at its first recorded sample (late). Neither is a
guaranteed real-world bound: unobserved flicker/geometry/focus changes are absent.
"""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tailstopia_workflow.mission_config import load_mission_config
from tailstopia_workflow.mission_flow import MissionFlow

ROOT=Path(__file__).resolve().parents[1]
STATES=('RETURN_CAMP','WAIT_RETURN_LOADING','VERIFY_RETURN_START','VERIFY_START')
CUTOFF='20261102_080358_570353_live'


def stats(values):
    return dict(n=len(values),median=round(float(np.median(values)),6),
                p95=round(float(np.percentile(values,95)),6),mean=round(float(np.mean(values)),6))


def read_cases(logs,cutoff=CUTOFF):
    for run in sorted(logs.glob('*_live')):
        if run.name>cutoff:continue
        by=defaultdict(list)
        with (run/'execution.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if row.get('state') in STATES:by[(row.get('loop'),row['state'])].append(row)
        for loop,state in sorted(by):
            if state!='VERIFY_START' or loop<=1:continue
            groups={s:by.get((loop if s=='VERIFY_START' else loop-1,s),[]) for s in STATES}
            results={s:next((e for e in groups[s] if e['kind']=='state_result' and e['status']=='SUCCESS'),None) for s in STATES}
            entries={s:next((e for e in groups[s] if e['kind']=='state_entry'),None) for s in STATES}
            if not all(results.values()) or not all(entries.values()):continue
            if any(abs(e.get('wall_elapsed',e['elapsed'])-e['elapsed'])>.1 for e in results.values()):continue
            hud={s:[e['status'] for e in groups[s] if e['kind']=='hud'] for s in STATES[1:]}
            costs={s:results[s]['timing']['vision']['seconds']/results[s]['timing']['vision']['calls'] for s in STATES}
            yield dict(run=run.name,loop=loop-1,hud=hud,costs=costs,
                       camp_seconds=results['RETURN_CAMP']['elapsed'],
                       measured_seconds=results['VERIFY_START']['active_at']-entries['RETURN_CAMP']['active_at'],
                       reused=any(e['kind']=='return_verification_reused' for e in groups['VERIFY_START']))


class ScenarioSession:
    def __init__(self,case,signal=None):
        self.case,self.signal=case,signal
        self.time=0.;self.loop=case['loop'];self.state='RETURN_CAMP'
        self.paused_total,self.pause_requested=0.,False
        self.capture_context=(1,43,95,1280,600)
        self.api=SimpleNamespace(held_keys=set(),held_buttons=set())
        self.indices=Counter();self.samples=[];self.captures=Counter();self.events=[];self.clicks=[]

    def now(self):return self.time
    def sleep(self,seconds):self.time+=seconds
    def check(self):pass
    def event(self,kind,**fields):self.events.append(dict(kind=kind,**fields))

    def match(self,name,capture):
        assert name=='camp'
        self.captures[self.state]+=1
        self.sleep(self.case['costs'][self.state])
        return SimpleNamespace(found=True,point=(853,457),score=1)

    def click(self,point,label):
        assert label=='camp'
        self.clicks.append(label)
        self.sleep(max(0,self.case['camp_seconds']-self.case['costs']['RETURN_CAMP']))

    def capture(self,box):
        state=self.state
        if self.signal is None:
            labels=self.case['hud'][state];index=self.indices[state]
            if index>=len(labels):raise AssertionError('Trace exhausted before confirmation')
            status=labels[index];self.indices[state]+=1
        else:
            times,labels=self.signal
            status=labels[max(0,bisect_right(times,self.time+1e-9)-1)]
        self.samples.append((self.time,status));self.captures[state]+=1
        self.sleep(self.case['costs'][state])
        return SimpleNamespace(status=status,score=1. if status=='READY' else 0.,
                               white_fraction=1. if status=='LOADING' else 0.)


def simulate(config,case,adaptive,signal=None):
    cfg=copy.deepcopy(config)
    # Historical traces contain Time HUD only, never Camp identity evidence.
    cfg['return'].pop('camp_checkpoint',None)
    if not adaptive:cfg['return'].pop('not_ready_poll',None)
    session=ScenarioSession(case,signal)
    flow=MissionFlow.__new__(MissionFlow)
    flow.config,flow.session=cfg,session
    flow.timings={};flow.reset_return_verification()
    flow.vision=SimpleNamespace(hud=lambda capture:capture(None),match=session.match,
                               start=SimpleNamespace(config={'poll_seconds':3.,'ready_confirmations':2}))
    waits=[];original_wait=flow.wait
    def wait(seconds):waits.append((session.state,seconds));return original_wait(seconds)
    flow.wait=wait
    measurements={}
    with patch('tailstopia_workflow.mission_flow.time.perf_counter',side_effect=session.now):
        for state in STATES:
            flow.state=session.state=state
            session.loop=case['loop']+(state=='VERIFY_START')
            flow.deadline=session.now()+cfg['timeouts'][state]
            flow.timings={};start=session.now()
            flow.execute_state()
            measurements[state]=dict(seconds=session.now()-start,
                wait_seconds=sum(v for s,v in waits if s==state),captures=session.captures[state])
    assert session.clicks==['camp']
    if signal is None:assert dict(session.indices)=={s:len(v) for s,v in case['hud'].items()}
    return dict(seconds=session.now(),wait_seconds=sum(v for _,v in waits),
                captures=sum(session.captures.values()),states=measurements,
                reused=any(e['kind']=='return_verification_reused' for e in session.events)),session.samples


def scenario(samples,early):
    times=[samples[0][0]];labels=[samples[0][1]]
    for i,(when,label) in enumerate(samples[1:],1):
        if label!=samples[i-1][1]:
            times.append(samples[i-1][0]+1e-7 if early else when)
            labels.append(label)
    return times,labels


def run(logs,cutoff=CUTOFF):
    config,_=load_mission_config(ROOT/'examples/mission_95.json')
    assert {k:v for k,v in config['return'].items() if k != 'camp_checkpoint'}==dict(loading_poll=.5,not_ready_poll=1,time_poll=3,confirmations=2)
    cases=list(read_cases(logs,cutoff))
    if not cases:raise ValueError('No complete live transitions available')
    rows=[]
    for case in cases:
        calibrated,samples=simulate(config,case,False)
        assert calibrated['reused']==case['reused'],'Logged proof result differs from simulation'
        # Preserve each trace's measured non-modelled overhead in both latency
        # estimates. This is an explicit assumption, not a new measurement.
        residual=case['measured_seconds']-calibrated['seconds']
        scenarios={}
        for name,early in [('early_change',True),('late_change',False)]:
            signal=scenario(samples,early)
            before,_=simulate(config,case,False,signal)
            assert abs(before['seconds']-calibrated['seconds'])<1e-6,'Model does not reproduce baseline trace'
            after,_=simulate(config,case,True,signal)
            assert after['reused']==before['reused'],'Proof behavior changed'
            scenarios[name]=dict(before=before,after=after,
                                 estimated_before_seconds=before['seconds']+residual,
                                 estimated_after_seconds=after['seconds']+residual,
                                 seconds_saved=before['seconds']-after['seconds'],
                                 capture_delta=after['captures']-before['captures'])
        rows.append(dict(run=case['run'],loop=case['loop'],measured_seconds=case['measured_seconds'],
                         calibrated=calibrated,preserved_residual_seconds=residual,scenarios=scenarios))
    summary={}
    for name in ('early_change','late_change'):
        data=[r['scenarios'][name] for r in rows]
        summary[name]=dict(before_seconds=stats([v['estimated_before_seconds'] for v in data]),
            after_seconds=stats([v['estimated_after_seconds'] for v in data]),
            seconds_saved=stats([v['seconds_saved'] for v in data]),
            before_wait=stats([v['before']['wait_seconds'] for v in data]),
            after_wait=stats([v['after']['wait_seconds'] for v in data]),
            before_captures=stats([v['before']['captures'] for v in data]),
            after_captures=stats([v['after']['captures'] for v in data]),
            capture_deltas=dict(Counter(v['capture_delta'] for v in data)),
            reused_before=sum(v['before']['reused'] for v in data),reused_after=sum(v['after']['reused'] for v in data))
    return dict(cutoff=cutoff,n=len(rows),measured_baseline_seconds=stats([r['measured_seconds'] for r in rows]),
                calibration_error_seconds=stats([r['calibrated']['seconds']-r['measured_seconds'] for r in rows]),
                assumptions=['SIMULATION ONLY: no game, screenshots, OCR, input dispatch or actual new latency measured.',
                    'Sparse HUD labels; no unobserved flicker. Early/late are scenarios, not guaranteed bounds.',
                    'Original schedule reconstructed with average recorded vision cost per state and nominal waits.',
                    'Capture cost remains the recorded state average; ignores new CPU contention, pauses, geometry changes.',
                    'Per-trace measured minus raw-model residual is held constant in both estimated total latencies; not in sampling decisions.',
                    'Final observed HUD persists; existing proof guard code executes without modification.'],
                summary=summary,cases=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,default=ROOT/'workflow_runs/mission_95')
    parser.add_argument('--through-run',default=CUTOFF)
    parser.add_argument('--output',type=Path,default=ROOT/'workflow_runs/return_polling/simulation.json')
    args=parser.parse_args();report=run(args.logs,args.through_run)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    print('Simulation report:',args.output.resolve())


if __name__=='__main__':main()
