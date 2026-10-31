"""Stage 95 uses a verified Vote button instead of the desert HUD/camera gate."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tailstopia_macro_replay import StopRequested,recording
from tailstopia_workflow.mission_config import STATES,mission_states,load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT=Path(__file__).resolve().parents[1]


class Stage95Tests(unittest.TestCase):
    def setUp(self):
        self.config,self.base=load_mission_config(ROOT/'examples/mission_95.json')
        self.vision=MissionVision(self.config,self.base)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_case(self,kind=FixtureSession,through=None):
        session=kind(self.config,self.temp.name,ROOT/'resource')
        self.addCleanup(lambda:session.last_image.close() if session.last_image is not None else None)
        flow=MissionFlow(self.config,self.base,session,self.vision)
        with session.log,contextlib.redirect_stdout(io.StringIO()):result=flow.run(through)
        rows=[json.loads(s) for s in (Path(self.temp.name)/'execution.jsonl').read_text().splitlines()]
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        return result,rows,session

    def test_two_loops_vote_once_then_replay_then_three_rewards_and_camp(self):
        self.config['max_loops']=2
        result,rows,_=self.run_case()
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',2))
        entries=[r['state'] for r in rows if r['kind']=='state_entry']
        self.assertEqual(entries,list(mission_states(self.config))*2)
        self.assertFalse(set(entries)&{'OPEN_ROOM_LIST','WAIT_STAGE_LOADING','VERIFY_STAGE','CALIBRATE_CAMERA'})
        self.assertNotIn('room_list',self.config['targets'])
        self.assertFalse(any(r['kind']=='recognition' and r['target']=='room_list' for r in rows))
        self.assertFalse(any(r['kind']=='click' and r['label']=='room_list' for r in rows))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='vote' for r in rows),2)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),2)
        reward=[r['label'] for r in rows if r['kind']=='click' and r['label'].startswith('reward_')]
        self.assertEqual(reward,['reward_1','reward_2','reward_3']*2)
        self.assertEqual([r['slots'] for r in rows if r['kind']=='reward_row_clicked'],[3,3])
        self.assertEqual(sum(r['kind']=='recording_start' and r['recording']=='mission' for r in rows),2)
        self.assertFalse(any(r['kind']=='action' and r.get('key') in {'c','z','win','winleft','cmd'} for r in rows))

    def test_vote_polling_and_post_click_wait_follow_config(self):
        result,rows,_=self.run_case(through='PLAY_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        start=next(r['active_at'] for r in rows if r['kind']=='state_result' and r['state']=='START_STAGE')
        first=next(r['active_at'] for r in rows if r['kind']=='recognition' and r['target']=='vote')
        self.assertAlmostEqual(first-start,self.config['vote']['first_wait'])
        checks=[r for r in rows if r['kind']=='recognition' and r['target']=='vote']
        self.assertEqual(len(checks),2)  # Discovery then a fresh confirmation before clicking.
        self.assertTrue(all(r['active_at']-start>=self.config['vote']['first_wait']-1e-8 for r in checks))
        click=next(r for r in rows if r['kind']=='click' and r['label']=='vote')
        self.assertEqual(click['client_point'],[1231,26])
        self.assertEqual(click['desktop_point'],[1274,121])
        release=next(r['active_at'] for r in rows if r['state']=='VOTE_STAGE' and r['kind']=='action' and r['method']=='mouseUp')
        mission=next(r['active_at'] for r in rows if r['kind']=='recording_start' and r['recording']=='mission')
        self.assertAlmostEqual(mission-release,self.config['vote']['after_click_wait'])
        actions=[r for r in rows if r['kind']=='macro_action' and r['recording']=='mission']
        self.assertEqual(len(actions),14)
        self.assertAlmostEqual(actions[0]['active_at']-mission,.3646)
        self.assertTrue(all(abs(r['lateness_ms'])<1e-5 for r in actions))

    def test_vote_ready_does_not_require_time_hud_present_in_stage(self):
        with Image.open(ROOT/'resource/95/vote.png') as image:
            capture=lambda box:self.vision.crop_reference(image,box,[43,95])
            self.assertEqual(self.vision.hud(capture).status,'UNKNOWN')
            self.assertTrue(self.vision.match('vote',capture).found)
        result,_,_=self.run_case(through='PLAY_MISSION')
        self.assertEqual(result.status,'SUCCESS')

    def test_vote_reference_and_updated_npc_position_match(self):
        with Image.open(ROOT/'resource/95/talk.png') as image:
            capture=lambda box:self.vision.crop_reference(image,box,[43,95])
            talk=self.vision.match('talk',capture)
            self.assertTrue(talk.found)
            self.assertGreater(talk.score,.99)
            self.assertEqual(talk.point,tuple(self.config['talk']['point_client']))
            self.assertEqual(self.vision.hud(capture).status,'READY')
            self.assertFalse(self.vision.match('vote',capture).found)

    def test_vote_rejects_camp_loading_and_other_recorded_scenes(self):
        paths=list((ROOT/'resource').glob('*.png'))+[ROOT/'resource/95/talk.png',ROOT/'resource/95/create.png']
        for path in paths:
            with self.subTest(image=path.name),Image.open(path) as image:
                origin=(0,0) if image.size==(1280,600) else (43,95)
                self.assertFalse(self.vision.match('vote',lambda box:self.vision.crop_reference(image,box,origin)).found)
                if path != ROOT/'resource/95/create.png':
                    self.assertFalse(self.vision.match('mission_selected',lambda box:self.vision.crop_reference(image,box,origin)).found)

    def test_delayed_vote_polls_without_reclicking_start(self):
        class Delayed(FixtureSession):
            vote_entry=None
            captures=[]
            def enter(session,state,loop):
                super().enter(state,loop)
                if state=='WAIT_VOTE':session.vote_entry=session.now();session.captures=[]
            def capture(session,box):
                if session.state=='WAIT_VOTE':
                    elapsed=session.now()-session.vote_entry
                    session.captures.append(elapsed)
                    session.frame='95/vote.png' if elapsed>=14-1e-8 else '95/talk.png'
                return super().capture(box)
        result,rows,session=self.run_case(Delayed,'PLAY_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual([round(t,6) for t in session.captures],[11,12,13,14])
        self.assertEqual(sum(r['kind']=='click' and r['label']=='start' for r in rows),1)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='vote' for r in rows),1)

    def test_vote_disappears_before_click_then_returns_without_unverified_click(self):
        class Blink(FixtureSession):
            checks=0
            def capture(session,box):
                if session.state=='VOTE_STAGE':
                    session.checks+=1
                    session.frame='95/talk.png' if session.checks==1 else '95/vote.png'
                return super().capture(box)
        result,rows,session=self.run_case(Blink,'PLAY_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(session.checks,2)
        checks=[r for r in rows if r['kind']=='recognition' and r['state']=='VOTE_STAGE']
        self.assertEqual([r['found'] for r in checks],[False,True])
        self.assertAlmostEqual(checks[1]['active_at']-checks[0]['active_at'],1)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='vote' for r in rows),1)

    def test_missing_vote_times_out_without_vote_click_or_mission_and_saves_frame(self):
        self.config['timeouts']['WAIT_VOTE']=12.2
        class Missing(FixtureSession):
            def enter(session,state,loop):
                super().enter(state,loop)
                if state=='WAIT_VOTE':session.frame='95/talk.png'
        result,rows,_=self.run_case(Missing)
        self.assertEqual((result.status,result.state),('FAIL','WAIT_VOTE'))
        self.assertIn('TimeoutError',result.reason)
        self.assertFalse(any(r['kind']=='click' and r['label']=='vote' for r in rows))
        self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='start' for r in rows),1)
        self.assertTrue(list(Path(self.temp.name).glob('*_WAIT_VOTE_failure.png')))

    def test_focus_loss_or_f9_during_polling_safe_stops(self):
        for error in (RuntimeError('Game lost focus'),StopRequested('F9')):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                class Interrupted(FixtureSession):
                    began=None
                    def enter(session,state,loop):
                        super().enter(state,loop)
                        if state=='WAIT_VOTE':session.began=session.now()
                    def check(session):
                        if session.state=='WAIT_VOTE' and session.now()-session.began>=11.1-1e-8:raise error
                    def capture(session,box):
                        if session.state=='WAIT_VOTE':session.frame='95/talk.png'
                        return super().capture(box)
                result,rows,_=self.run_case(Interrupted)
                self.assertEqual((result.status,result.state),('FAIL','WAIT_VOTE'))
                self.assertEqual(result.stopped,isinstance(error,StopRequested))
                self.assertFalse(any(r['kind']=='click' and r['label']=='vote' for r in rows))
                self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
                self.assertTrue(list(Path(self.temp.name).glob('*_WAIT_VOTE_*.png')))

    def test_interrupted_five_second_settle_never_repeats_vote_or_starts_mission(self):
        class Interrupted(FixtureSession):
            voted=None
            def click_effect(session,label):
                super().click_effect(label)
                if label=='vote':session.voted=session.now()
            def check(session):
                if session.state=='VOTE_STAGE' and session.voted is not None and session.now()-session.voted>=.5-1e-8:
                    raise StopRequested('F9')
        result,rows,_=self.run_case(Interrupted)
        self.assertEqual((result.status,result.state,result.stopped),('FAIL','VOTE_STAGE',True))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='vote' for r in rows),1)
        self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))

    def test_wrong_mission_is_selected_and_title_confirmed_before_create(self):
        class Unselected(FixtureSession):
            def capture(session,box):
                if session.state=='SELECT_MISSION' and tuple(box)==tuple(session.config['targets']['mission_selected']['search_roi_client']) and not session.click_counts.get('mission_selection'):
                    return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,_=self.run_case(Unselected,'START_STAGE')
        self.assertEqual(result.status,'SUCCESS')
        choose=[r for r in rows if r['kind']=='click' and r['label']=='mission_selection']
        self.assertEqual(len(choose),1)
        self.assertEqual(choose[0]['client_point'],[850,249])
        title=[r for r in rows if r['kind']=='recognition' and r['target']=='mission_selected']
        self.assertEqual([r['found'] for r in title],[False,True])
        create=next(r for r in rows if r['kind']=='click' and r['label']=='create')
        self.assertEqual(create['client_point'],[1195,385])
        self.assertGreater(rows.index(create),rows.index(title[-1]))

    def test_unconfirmed_mission_title_safe_stops_before_creating_wrong_stage(self):
        class Unconfirmed(FixtureSession):
            def capture(session,box):
                if session.state=='SELECT_MISSION' and tuple(box)==tuple(session.config['targets']['mission_selected']['search_roi_client']):
                    return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,_=self.run_case(Unconfirmed)
        self.assertEqual((result.status,result.state),('FAIL','SELECT_MISSION'))
        self.assertFalse(any(r['kind']=='click' and r['label'] in {'create','start','vote'} for r in rows))
        self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='mission_selection' for r in rows),3)

    def test_already_selected_95_skips_selection_click_and_reference_create_matches(self):
        with Image.open(ROOT/'resource/95/create.png') as image:
            capture=lambda box:self.vision.crop_reference(image,box,[43,95])
            for target in ('mission_list','mission_selected','create'):
                self.assertTrue(self.vision.match(target,capture).found)
        result,rows,_=self.run_case(through='CREATE_ROOM')
        self.assertEqual(result.status,'SUCCESS')
        self.assertFalse(any(r['kind']=='click' and r['label']=='mission_selection' for r in rows))

    def test_talk_opens_mission_list_directly_with_only_two_entry_captures(self):
        result,rows,_=self.run_case(through='OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        checks=[r for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='recognition']
        self.assertEqual([r['target'] for r in checks],['mission_list','mission_list'])
        self.assertEqual([r['found'] for r in checks],[False,True])
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        state=next(r for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='state_result')
        self.assertAlmostEqual(state['elapsed'],5.05)

    def test_already_open_direct_list_never_clicks_talk_again(self):
        class AlreadyOpen(FixtureSession):
            def enter(session,state,loop):
                super().enter(state,loop)
                if state=='OPEN_MISSION':session.frame='95/create.png'
        result,rows,_=self.run_case(AlreadyOpen,'OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertFalse(any(r['kind']=='click' and r['label']=='talk' for r in rows))
        state=next(r for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='state_result')
        self.assertEqual(state['elapsed'],0)

    def test_direct_list_delayed_during_talk_polling_does_not_reclick_talk(self):
        class Delayed(FixtureSession):
            talked=None
            def click_effect(session,label):
                if label=='talk':session.talked=session.now()
                else:super().click_effect(label)
            def capture(session,box):
                if session.state=='OPEN_MISSION' and session.talked is not None and session.now()-session.talked>=7.3:
                    session.frame='95/create.png'
                return super().capture(box)
        result,rows,_=self.run_case(Delayed,'OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        checks=[r for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='recognition']
        self.assertTrue(all(r['target']=='mission_list' for r in checks))
        self.assertEqual(len(checks),7)

    def test_verified_text_fallback_opens_direct_list_after_original_waits(self):
        class Fallback(FixtureSession):
            def click_effect(session,label):
                if label=='talk' and not session.click_counts.get(label):session.click_counts[label]=1
                else:super().click_effect(label)
        result,rows,_=self.run_case(Fallback,'OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),2)
        self.assertFalse(any(r['kind']=='recognition' and r['target']=='room_list' for r in rows))
        state=next(r for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='state_result')
        self.assertAlmostEqual(state['elapsed'],15.6)

    def test_delayed_direct_list_during_absent_talk_confirmation_prevents_reclick(self):
        class DuringConfirmation(FixtureSession):
            talked=None
            def click_effect(session,label):
                if label=='talk':session.talked=session.now()
                else:super().click_effect(label)
            def capture(session,box):
                if session.state=='OPEN_MISSION' and session.talked is not None:
                    if session.now()-session.talked>=10.4:session.frame='95/create.png'
                    if tuple(box)==tuple(session.config['targets']['talk']['search_roi_client']):
                        return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,_=self.run_case(DuringConfirmation,'OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        self.assertFalse(any(r['kind']=='recognition' and r['target']=='room_list' for r in rows))

    def test_direct_list_never_appears_safe_stops_before_create(self):
        class Missing(FixtureSession):
            def click_effect(session,label):
                if label!='talk':super().click_effect(label)
        result,rows,_=self.run_case(Missing)
        self.assertEqual((result.status,result.state),('FAIL','OPEN_MISSION'))
        self.assertIn('Talk fallback did not open Mission List',result.reason)
        self.assertFalse(any(r['kind']=='click' and r['label'] in {'room_list','create','start','vote'} for r in rows))
        self.assertTrue(list(Path(self.temp.name).glob('*_OPEN_MISSION_failure.png')))

    def test_invalid_talk_destination_is_rejected(self):
        for target in ('create','missing',''):
            cfg=copy.deepcopy(self.config);cfg['talk']['opened_target']=target
            with self.assertRaises(ValueError):validate_mission_config(cfg)
        cfg=copy.deepcopy(self.config);del cfg['targets']['mission_list']
        with self.assertRaises(ValueError):validate_mission_config(cfg)

    def test_original_profile_keeps_original_state_sequence_and_seven_slots(self):
        old,_=load_mission_config(ROOT/'examples/mission_flow.json')
        self.assertEqual(mission_states(old),STATES)
        self.assertNotIn('vote',old)
        self.assertEqual(len(old['reward']['slots_client']),7)
        self.assertEqual(old['start_recording'],'../recordings/start.json')
        self.assertEqual(old['mission_recording'],'../recordings/mission.json')

    def test_skill_timestamps_and_original_total_duration_preserved_without_windows_key(self):
        actions,duration=recording(self.base/self.config['mission_recording'])
        self.assertEqual(duration,5.9475)
        self.assertEqual([(a['t'],a['type'],a['key']) for a in actions],
                         [(.3646,'key_down','1'),(.5161,'key_up','1'),(.8138,'key_down','2'),(.902,'key_up','2'),
                          (.9594,'key_down','2'),(1.0395,'key_up','2'),(1.0946,'key_down','2'),(1.2497,'key_up','2'),
                          (1.378,'key_down','3'),(1.4588,'key_up','3'),(1.5244,'key_down','3'),(1.6017,'key_up','3'),
                          (1.6584,'key_down','3'),(1.7517,'key_up','3')])
        csv_actions,_=recording(ROOT/'recordings/95/mission.csv')
        self.assertEqual([(a['t'],a['type'],a['key']) for a in csv_actions],[(a['t'],a['type'],a['key']) for a in actions])
        data=json.loads((ROOT/'recordings/95/mission.json').read_text())
        self.assertEqual(data['event_count'],len(data['events']))
        _,start_duration=recording(self.base/self.config['start_recording'])
        self.assertEqual(start_duration,4.6614)

    def test_stage95_disallows_old_camera_stop_target_before_sending_input(self):
        with self.assertRaisesRegex(ValueError,'not enabled'):
            self.run_case(through='CALIBRATE_CAMERA')

    def test_bad_vote_settings_and_reward_counts_rejected(self):
        for key,value in [('first_wait',-1),('first_wait',True),('poll',0),('poll',float('nan')),('after_click_wait',-1)]:
            with self.subTest(key=key,value=value):
                cfg=copy.deepcopy(self.config);cfg['vote'][key]=value
                with self.assertRaises(ValueError):validate_mission_config(cfg)
        for n in (0,8):
            cfg=copy.deepcopy(self.config);cfg['reward']['slots_client']=[[416,290]]*n
            with self.assertRaises(ValueError):validate_mission_config(cfg)
        for threshold in (0,True,float('nan'),1.1):
            cfg=copy.deepcopy(self.config);cfg['targets']['vote']['threshold']=threshold
            with self.assertRaises(ValueError):validate_mission_config(cfg)
        cfg=copy.deepcopy(self.config);del cfg['targets']['vote']
        with self.assertRaisesRegex(ValueError,'vote target'):validate_mission_config(cfg)


if __name__=='__main__':unittest.main()
