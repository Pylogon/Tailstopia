"""Ordered, bounded execution of all ten steps in note.txt."""
from dataclasses import dataclass
import time

from tailstopia_macro_replay import recording, replay, validate_inputs, StopRequested
from .mission_config import mission_states, TALK_RECOVERY_DEFAULTS, validate_mission_config


@dataclass(frozen=True)
class FlowResult:
    status: str
    reason: str
    state: str
    loops_completed: int
    stopped: bool = False


class MissionFlow:
    RETURN_READY_MAX_AGE = 1.0  # Wall time, including pauses and slow captures.

    def __init__(self,config,base,session,vision,verbose_actions=False):
        self.config = validate_mission_config(config)
        self.session,self.vision = session,vision
        self.state,self.completed,self.deadline = "PREFLIGHT",0,None
        self.mission_started,self.stage_clicked = None,None
        self.timings = {}
        self.verbose_actions = verbose_actions
        self.reset_mission_counts()
        self.reset_return_verification()
        # Both recordings and all keys are validated before dispatching input.
        self.macros = {}
        for name,key in (("start","start_recording"),("mission","mission_recording")):
            actions,duration = recording(base/config[key])
            self.macros[name] = (validate_inputs(actions,session.api),duration)

    def reset_return_verification(self):
        self._return_clicked_loop = None
        self._return_loading_loop = None
        self._return_loading_context = None
        self._return_confirmation = None

    def pause_marker(self):
        return (getattr(self.session,"paused_total",0),getattr(self.session,"pause_requested",False))

    def camp_policy_signature(self):
        policy = self.config["return"]["camp_checkpoint"]
        return (policy["template_image"],tuple(policy["search_roi_client"]),policy["threshold"],
                policy["poll"],policy["confirmation_interval"],policy["timeout"])

    def return_confirmation_reason(self,proof,poll,confirmations):
        loop = getattr(self.session,"loop",None)
        if loop is None or loop <= 1 or proof["loop"] != loop-1:
            return "Not the immediately following loop"
        age = time.perf_counter()-proof["at"]
        if not 0 <= age <= self.RETURN_READY_MAX_AGE:
            return "Previous return confirmation expired"
        if self.pause_marker() != proof["pause"] or self.pause_marker()[1]:
            return "Pause state changed"
        if getattr(self.session,"capture_context",None) != proof["context"]:
            return "Game window or capture geometry changed"
        if self.session.api.held_keys or self.session.api.held_buttons:
            return "Inputs are still held"
        if proof.get("kind") == "camp_checkpoint":
            policy = self.config["return"].get("camp_checkpoint",{})
            if not policy.get("enabled") or proof["policy"] != self.camp_policy_signature():
                return "Camp checkpoint policy changed or disabled"
            if proof["start_policy"] != (poll,confirmations):
                return "Start confirmation policy changed"
            if (self._return_clicked_loop != proof["loop"] or self._return_loading_loop != proof["loop"] or
                    self._return_loading_context != proof["context"]):
                return "Verified return click/loading evidence changed"
            if proof["confirmations"] < max(2,confirmations) or proof["poll"] < policy["confirmation_interval"]:
                return "Camp checkpoint confirmation policy is weaker"
        elif proof["confirmations"] < max(2,confirmations) or proof["poll"] < poll:
            return "Previous confirmation policy is weaker"
        return None

    def measure(self,category,operation,*args,**kwargs):
        capture_before = self.timings.get("vision_capture",{}).get("seconds",0.0)
        started = time.perf_counter()
        try:
            return operation(*args,**kwargs)
        finally:
            # Aggregate only; no timing log, capture, or recognition per wait poll.
            stats = self.timings.setdefault(category,{"calls":0,"seconds":0.0})
            stats["calls"] += 1
            elapsed = time.perf_counter()-started
            stats["seconds"] += elapsed
            if category == "vision":
                captured = self.timings.get("vision_capture",{}).get("seconds",0.0)-capture_before
                processing = self.timings.setdefault("vision_processing",{"calls":0,"seconds":0.0})
                processing["calls"] += 1
                processing["seconds"] += max(0.0,elapsed-captured)

    def timing_summary(self):
        summary = {name:{"calls":value["calls"],"seconds":round(value["seconds"],6)}
                   for name,value in self.timings.items()}
        summary["logging"] = self.session.log.timing_summary()
        return summary

    def capture(self,box):
        # These are the existing vision captures, timed without extra frames.
        return self.measure("vision_capture",self.session.capture,box)

    def check(self):
        self.session.check()
        if self.deadline is not None and self.session.now() >= self.deadline:
            raise TimeoutError(f"{self.state}: state deadline exceeded")

    def wait(self,seconds):
        return self.measure("wait",self._wait,seconds)

    def _wait(self,seconds):
        target = self.session.now()+max(0,seconds)
        while True:
            self.check()
            remaining = target-self.session.now()
            if remaining <= 0:
                return
            self.session.sleep(min(.05,remaining))

    def match(self,name):
        self.check()
        match = self.measure("vision",self.vision.match,name,self.capture)
        self.check()
        details = {}
        if getattr(match,"method","template") == "normalize":
            details = dict(detector="normalize",fast_score=match.fast_score,
                           reason=match.reason,candidates=match.candidates)
        self.session.event("recognition",target=name,found=match.found,score=match.score,point=match.point,
                           threshold=self.config["targets"][name]["threshold"],**details)
        if details and not match.found and ("ambiguous" in match.reason or "budget" in match.reason):
            raise RuntimeError(f"Talk normalization uncertain: {match.reason}; no click sent")
        return match

    def hud(self):
        self.check()
        # Timestamp before capture/logging so slow work cannot make an old frame fresh.
        self._last_hud_at = time.perf_counter()
        observation = self.measure("vision",self.vision.hud,self.capture)
        self.check()
        self.record_hud(observation)
        return observation

    def record_hud(self,observation):
        loop = getattr(self.session,"loop",None)
        if (self.state in ("WAIT_RETURN_LOADING","VERIFY_RETURN_START") and
                loop is not None and getattr(self,"_return_clicked_loop",None) == loop and
                observation.status == "LOADING"):
            self._return_loading_loop = loop
            self._return_loading_context = getattr(self.session,"capture_context",None)
        self.session.event("hud",status=observation.status,score=observation.score,
                           white_fraction=observation.white_fraction)

    def camp_hud(self):
        self.check()
        self._last_hud_at = time.perf_counter()
        observation = self.measure("vision",self.vision.camp_hud,self.capture)
        self.check()
        self.record_hud(observation.hud)
        self.session.event("camp_checkpoint",detector=self.vision.camp.VERSION,
                           found=observation.found,score=observation.score,center=observation.center,
                           threshold=self.config["return"]["camp_checkpoint"]["threshold"],
                           hud_status=observation.hud.status,hud_score=observation.hud.score)
        return observation

    def try_camp_return(self):
        """Bounded experiment; only a verified click + observed LOADING can opt in."""
        policy = self.config["return"]["camp_checkpoint"]
        loop,pause = self.session.loop,self.pause_marker()
        context = getattr(self.session,"capture_context",None)
        if (context is None or self._return_clicked_loop != loop or self._return_loading_loop != loop or
                self._return_loading_context != context or pause[1] or
                self.session.api.held_keys or self.session.api.held_buttons):
            self.session.event("camp_return_fallback",reason="No eligible return click/loading/context proof")
            return None,None
        signature = self.camp_policy_signature()
        count,first,last = 0,None,None
        started = self.session.now()
        budget = started+policy["timeout"]
        required = max(2,self.config["return"]["confirmations"],self.vision.start.config["ready_confirmations"])
        self.session.event("camp_return_trial",detector=self.vision.camp.VERSION,
                           poll=policy["poll"],confirmation_interval=policy["confirmation_interval"],
                           threshold=policy["threshold"],timeout=policy["timeout"],confirmations=required)
        reason = "Camp checkpoint trial budget exhausted"
        while self.session.now() < budget:
            self.check()
            if self.pause_marker() != pause or self.session.capture_context != context:
                reason = "Pause or capture context changed during Camp trial"
                break
            sample = self.camp_hud()
            last = sample.hud
            if (self.pause_marker() != pause or self.session.capture_context != context or
                    self.session.api.held_keys or self.session.api.held_buttons or
                    signature != self.camp_policy_signature()):
                reason = "Camp evidence interrupted or policy changed"
                break
            if self.session.now() >= budget:
                break
            if sample.found and last.status == "READY":
                count += 1
                if first is None:
                    first = self._last_hud_at
                if count >= required and self._last_hud_at-first >= policy["confirmation_interval"]*(required-1):
                    proof = dict(kind="camp_checkpoint",loop=loop,at=self._last_hud_at,pause=pause,context=context,
                                 confirmations=count,poll=policy["confirmation_interval"],policy=signature,
                                 start_policy=(self.vision.start.config["poll_seconds"],self.vision.start.config["ready_confirmations"]))
                    self.session.event("camp_return_verified",proof_kind=proof["kind"],confirmations=count,
                                       elapsed=self.session.now()-started,score=sample.score,hud_score=last.score,
                                       confirmation_span=self._last_hud_at-first)
                    return proof,None
            else:
                count,first = 0,None
            self.wait(min(policy["confirmation_interval"] if count else policy["poll"],budget-self.session.now()))
        self.check()
        self.session.event("camp_return_fallback",reason=reason,elapsed=self.session.now()-started,
                           last_hud=last.status if last is not None else None)
        # The joint frame already contains Time; don't immediately capture it twice.
        if self.pause_marker() != pause or self.session.capture_context != context:
            last = None  # An interrupted frame cannot seed fresh legacy confirmation.
        return None,last

    def reuse_camp_return(self,proof,poll,confirmations):
        reason = self.return_confirmation_reason(proof,poll,confirmations)
        first = None
        if reason is None:
            sample = self.camp_hud()
            first = sample.hud
            reason = self.return_confirmation_reason(proof,poll,confirmations)
            if reason is None and sample.found and first.status == "READY":
                self.session.event("return_verification_reused",proof_kind="camp_checkpoint",source_loop=proof["loop"],
                                   age_seconds=time.perf_counter()-proof["at"],score=first.score,camp_score=sample.score)
                return
            reason = reason or f"Fresh Camp={sample.found}; HUD={first.status}"
        self.session.event("return_verification_fallback",proof_kind="camp_checkpoint",source_loop=proof["loop"],reason=reason)
        self.wait_hud(poll,confirmations,first_observation=first)

    def click(self,point,label):
        self.check()
        # UI clicks are paired down/up and tracked independently of macro inputs.
        self.measure("click",self.session.click,point,label)
        self.check()

    def click_matched(self,name):
        match = self.match(name)
        if not match.found:
            raise RuntimeError(f"Cannot find {name}; no click sent")
        self.click(match.point,name)

    def wait_hud(self,poll,confirmations,return_proof=None,not_ready_poll=None,first_observation=None):
        count = 0
        while True:
            if first_observation is None:
                observation = self.hud()
            else:
                self.check()
                observation,first_observation = first_observation,None
            count = count+1 if observation.status == "READY" else 0
            if return_proof is not None:
                reason = self.return_confirmation_reason(return_proof,poll,confirmations)
                if reason is None and observation.status == "READY":
                    self.session.event("return_verification_reused",source_loop=return_proof["loop"],
                                       age_seconds=time.perf_counter()-return_proof["at"],score=observation.score)
                    return
                self.session.event("return_verification_fallback",source_loop=return_proof["loop"],
                                   reason=reason or f"Fresh HUD={observation.status}")
                return_proof = None
                # Keep this first observation/count; do not capture again immediately.
            if count >= confirmations:
                return
            self.session.event("retry",status="RETRY",reason=f"HUD={observation.status}; confirmations={count}/{confirmations}")
            # Only return verification opts in. READY always keeps the original
            # confirmation interval; any non-READY resets the count above.
            self.wait(not_ready_poll if observation.status != "READY" and not_ready_poll is not None else poll)

    def wait_loading(self,poll):
        destination = "Stage" if self.state == "WAIT_STAGE_LOADING" else "Camp"
        policy = self.config["stage"] if destination == "Stage" else self.config["return"]
        ready_count = 0
        # A missed loading frame is allowed. On entry, reject early source HUD
        # until the existing minimum wait after Start has elapsed.
        ready_after = (self.stage_clicked+policy["min_entry_wait"]
                       if destination == "Stage" else self.session.now())
        while True:
            observation = self.hud()
            if observation.status == "LOADING":
                # The next VERIFY state still requires stable Time HUD.
                return
            eligible = self.session.now() >= ready_after
            ready_count = ready_count+1 if observation.status == "READY" and eligible else 0
            if ready_count >= policy["confirmations"]:
                self.session.event("loading_skipped",destination=destination,
                                   reason="Stable Time HUD accepted without an observed loading frame",
                                   ready_confirmations=ready_count,score=observation.score)
                return
            self.session.event("retry",status="RETRY",reason="Waiting for loading or stable Time HUD; no action repeated",
                               hud_status=observation.status,destination=destination,
                               ready_confirmations=ready_count,minimum_wait_remaining=max(0,ready_after-self.session.now()))
            self.wait(poll)

    def play(self,name):
        self.session.prepare_macro(name)
        actions,duration = self.macros[name]
        self.session.event("recording_start",recording=name,actions=len(actions),duration=duration)
        if name == "mission":
            self.mission_started = self.session.now()
        def action_report(action,dispatched):
            self.session.event("macro_action",recording=name,action=action,dispatched_seconds=dispatched,
                               lateness_ms=(dispatched-action["t"])*1000)
        self.measure("macro",replay,actions,duration,self.session.api,self.check,clock=self.session.now,sleep=self.session.sleep,
               on_action=action_report,verbose=self.verbose_actions)
        self.check()
        self.session.event("recording_complete",recording=name)

    def rotate_camera(self,key):
        policy = self.config["camera"]
        # Reuse the existing English-layout/focus guard without replaying or
        # changing either recording. C/Z are camera controls confirmed by user.
        self.session.prepare_macro("camera_calibration")
        self.check()
        try:
            self.session.api.keyDown(key,_pause=False)
            self._wait(policy["rotation_seconds"])
            self.check()
            self.session.api.keyUp(key,_pause=False)
        finally:
            # Release on F9, focus loss, timeout, or a backend exception too.
            if key in self.session.api.held_keys:
                self.session.api.release_all()

    def calibrate_camera(self):
        policy = self.config["camera"]
        count,rotations = 0,0
        direction = policy["rotation_key"]
        if self.session.api.held_keys or self.session.api.held_buttons:
            raise RuntimeError("Camera calibration requires all previous inputs released")
        while True:
            self.check()
            observation = self.measure("vision",self.vision.camera.observe,self.capture,scan_sides=(count == 0))
            self.check()
            count = count+1 if observation.aligned else 0
            if observation.side == "left":
                direction = policy["left_rotation_key"]
            elif observation.side == "right":
                direction = "z" if policy["left_rotation_key"] == "c" else "c"
            self.session.event("camera_observation",aligned=observation.aligned,
                               blue_pixels=observation.blue_pixels,blob_pixels=observation.blob_pixels,
                               min_blue_blob_pixels=policy["min_blue_blob_pixels"],point=observation.point,
                               roi=list(self.vision.camera.roi),confirmations=count,rotations=rotations,
                               side=observation.side,side_blob_pixels=observation.side_blob_pixels,
                               side_point=observation.side_point,search_key=direction)
            if count >= policy["confirmations"]:
                self.session.event("camera_calibrated",rotations=rotations,blue_pixels=observation.blue_pixels,
                                   blob_pixels=observation.blob_pixels,point=observation.point)
                return
            if not observation.aligned:
                if rotations >= policy["max_rotations"]:
                    raise RuntimeError(f"Camera calibration retry budget exhausted; rotations={rotations}; "
                                       f"blue_pixels={observation.blue_pixels}; blob_pixels={observation.blob_pixels}")
                self.session.event("retry",status="RETRY",reason="Camera landmark unconfirmed; rotate then recheck",
                                   attempt=rotations+1,key=direction,duration=policy["rotation_seconds"],side=observation.side)
                self.measure("rotation",self.rotate_camera,direction)
                rotations += 1
            # No capture/OpenCV while rotating. A positive match is confirmed
            # by a second independent observation with the camera stationary.
            self.wait(policy["settle_seconds"])

    def wait_room_list(self,initial_wait,poll_timeout,target="room_list"):
        # A failed poll never clicks. The complete polling window must expire
        # before the caller may enter the verified Talk-template fallback.
        self.wait(initial_wait)
        until = self.session.now()+poll_timeout
        while True:
            if self.match(target).found:
                return True
            remaining = until-self.session.now()
            if remaining <= 0:
                return False
            # One observation per interval, with a final check at the budget.
            self.wait(min(self.config["talk"].get("poll_interval",0.5),remaining))

    def move_toward_talk(self,seconds):
        # Use the original desktop anchor, just as replay does. Never replay
        # the start recording again or alter its timestamps for recovery.
        api = self.session.api
        if api.held_keys or api.held_buttons:
            raise RuntimeError("Talk recovery requires all previous inputs released")
        anchor = next((action for action in self.macros["start"][0]
                       if action["type"] == "mouse_down" and action["button"] == "right"),None)
        if anchor is None:
            raise RuntimeError("Talk recovery needs a right-mouse anchor in the start recording")
        self.session.prepare_macro("talk_recovery")
        self.check()
        policy = {**TALK_RECOVERY_DEFAULTS,**self.config["talk"].get("recovery",{})}
        try:
            api.mouseDown(x=anchor["x"],y=anchor["y"],button="right",_pause=False)
            for key in policy["keys"]:
                self.check()
                api.keyDown(key,_pause=False)
            self._wait(seconds)
        finally:
            # Includes partially dispatched input, F9, lost focus and timeout.
            # InputTrace attempts every release, even if one backend up fails.
            api.release_all()
        self.check()

    def recover_talk(self,missing):
        policy = {**TALK_RECOVERY_DEFAULTS,**self.config["talk"].get("recovery",{})}
        target = self.config["talk"].get("opened_target","room_list")
        attempts = 0
        while True:
            # Confirm absence on a separate stationary frame. Check Target
            # first: a delayed menu must never trigger compensating movement.
            self.wait(policy["confirm_wait"])
            if self.match(target).found:
                return None
            confirmed = self.match("talk")
            if confirmed.found:
                return confirmed
            if attempts >= policy["max_attempts"]:
                raise RuntimeError(f"Talk recovery retry budget exhausted; attempts={attempts}; "
                                   f"talk_score={confirmed.score:.6f}; no fallback click sent")
            direction = " + ".join(key.upper() for key in policy["keys"])
            self.session.event("retry",status="RETRY",reason=f"Talk absent in two checks; short right-mouse + {direction} recovery",
                               attempt=attempts+1,max_attempts=policy["max_attempts"],
                               duration=policy["hold_seconds"],keys=list(policy["keys"]),
                               previous_score=missing.score,score=confirmed.score)
            self.measure("recovery_move",self.move_toward_talk,policy["hold_seconds"])
            attempts += 1
            self.session.event("talk_recovery_move",attempt=attempts,duration=policy["hold_seconds"],
                               keys=list(policy["keys"]),
                               cumulative_hold_seconds=attempts*policy["hold_seconds"])
            # All inputs are released before settling or taking either ROI.
            self.wait(policy["settle_seconds"])
            if self.match(target).found:
                return None
            missing = self.match("talk")
            if missing.found:
                return missing

    def open_mission(self):
        spec = self.config["talk"]
        target = spec.get("opened_target","room_list")
        title = "Room List" if target == "room_list" else "Mission List"
        if self.match(target).found:
            return
        self.click(spec["point_client"],"talk")
        if self.wait_room_list(spec["first_wait"],spec.get("poll_timeout",5),target):
            return
        self.session.event("retry",status="RETRY",reason=f"{title} absent after Talk polling window; locating Talk template")
        talk = self.match("talk")
        if not talk.found:
            talk = self.recover_talk(talk)
        if talk is None:  # The configured destination appeared during recovery.
            return
        self.click(talk.point,"talk")
        # Preserve the verified-text fallback and its existing settle time.
        self.wait(spec["fallback_wait"])
        if not self.match(target).found:
            raise RuntimeError(f"Talk fallback did not open {title}")

    def advance(self,source,target,point=None,label=None):
        policy = self.config["ui"]
        last_source = last_target = None
        for attempt in range(policy["additional_retries"]+1):
            if attempt:
                self.wait(policy["retry_delay"])
            # A transition can hide Source before Target is recognized. Always
            # look for Target first, and never click an unconfirmed Source.
            last_target = self.match(target)
            if last_target.found:
                return
            last_source = self.match(source)
            if not last_source.found:
                self.session.event("retry",status="RETRY",attempt=attempt,
                                   reason=f"{source} and {target} unconfirmed; no click sent",
                                   source_score=last_source.score,target_score=last_target.score)
                continue
            self.click(last_source.point if point is None else point,label or source)
            self.wait(policy["settle"])
            last_target = self.match(target)
            if last_target.found:
                return
            self.session.event("retry",status="RETRY",attempt=attempt,
                               reason=f"{target} absent after {source}; recheck before another click",
                               source_score=last_source.score,target_score=last_target.score)
        raise RuntimeError(f"{source} -> {target}: UI retry budget exhausted; "
                           f"source_score={last_source.score:.6f}, target_score={last_target.score:.6f}")

    def wait_vote(self):
        if self.stage_clicked is None:
            raise RuntimeError("Vote polling requires a completed Start click")
        policy = self.config["vote"]
        self.wait(policy["first_wait"]-(self.session.now()-self.stage_clicked))
        while True:
            match = self.match("vote")
            if match.found:
                return match
            self.session.event("retry",status="RETRY",reason="Waiting for stage Vote button; no action repeated")
            self.wait(policy["poll"])

    def wait_result(self):
        # 80s includes the time already spent executing the mission recording.
        elapsed = self.session.now()-self.mission_started
        remaining = max(0,self.config["result"]["after_mission_start"]-elapsed)
        self.session.event("result_timer",mission_elapsed=elapsed,remaining=remaining)
        self.wait(remaining)
        count = 0
        while True:
            match = self.match("result")
            count = count+1 if match.found else 0
            if count >= self.config["result"]["confirmations"]:
                return
            self.session.event("retry",status="RETRY",reason="Waiting for stable Mission Result")
            self.wait(self.config["result"]["poll"])

    def reset_mission_counts(self):
        self.mission_counts = {"pass_with_reward":0,"mission_fail":0,"no_reward":0}
        self._counted_outcome_loop = None
        self._mission_outcome = None

    def record_mission_outcome(self,outcome,reason):
        if self._counted_outcome_loop == self.session.loop:
            if self._mission_outcome != outcome:
                raise RuntimeError("Conflicting mission outcomes in one loop")
            return
        key = {"PASS_WITH_REWARD":"pass_with_reward","FAIL":"mission_fail","NO_REWARD":"no_reward"}[outcome]
        self.mission_counts[key] += 1
        self._counted_outcome_loop,self._mission_outcome = self.session.loop,outcome
        self.session.event("mission_outcome",outcome=outcome,reason=reason,**self.mission_counts)

    def confirm_result_failure(self):
        policy = self.config["result"]
        previous,count = None,0
        names = {"result":"result","fail":policy["failure_target"]}
        while True:
            self.check()
            matches = self.measure("vision",self.vision.observe_result,self.capture)
            self.check()
            for role,name in names.items():
                match = matches[role]
                self.session.event("recognition",target=name,found=match.found,score=match.score,point=match.point,
                                   threshold=self.config["targets"][name]["threshold"])
            outcome = None
            if matches["result"].found:
                # A missing '-' keeps the existing pickup verification path;
                # it does not prove a win or require a particular letter grade.
                outcome = "FAIL" if matches["fail"].found else "CHECK_REWARDS"
            count = count+1 if outcome is not None and outcome == previous else (1 if outcome else 0)
            if count >= policy["confirmations"]:
                return outcome == "FAIL"
            previous = outcome
            self.session.event("retry",status="RETRY",reason="Waiting for stable Total Result",
                               outcome=outcome or "UNKNOWN",confirmations=count)
            self.wait(policy["poll"])

    def received(self,baseline):
        policy = self.config["reward"]
        for index in range(policy["confirmations"]):
            self.check()
            with self.session.capture(self.vision.reward_box) as after:
                changed,fraction = self.vision.change(baseline,after)
            self.check()
            self.session.event("reward_observation",changed=changed,fraction=fraction,confirmation=index+1)
            if not changed:
                return False
            if index+1 < policy["confirmations"]:
                self.wait(policy["verify_wait"])
        return True

    def collect(self):
        policy = self.config["reward"]
        self.wait(policy["settle"])
        if "failure_target" in self.config["result"]:
            if self.confirm_result_failure():
                self.record_mission_outcome("FAIL","Total Result '-' confirmed; skip rewards and return to Mission Camp")
                self.session.event("reward_skipped",reason="Mission defeated; no reward clicks sent")
                return
        elif not self.match("result").found:
            raise RuntimeError("Mission Result disappeared before reward collection")
        self.session.rest_pointer()
        self.wait(policy["verify_wait"])
        baseline = self.session.capture(self.vision.reward_box)
        try:
            if not self.vision.empty_slot(baseline):
                raise RuntimeError("First Your Items slot is not empty at entry; cannot verify a new pickup")
            for attempt in range(policy["additional_retries"]+1):
                if attempt:
                    self.wait(policy["retry_delay"])
                    # A delayed pickup may have arrived: do not click again then.
                    if self.received(baseline):
                        break
                    self.session.event("retry",status="RETRY",reason="First pickup missing; retry first slot only",attempt=attempt)
                if not self.match("result").found:
                    raise RuntimeError("Mission Result disappeared during reward retry")
                self.click(policy["slots_client"][0],"reward_1")
                self.session.rest_pointer()
                self.wait(policy["verify_wait"])
                if self.received(baseline):
                    break
            else:
                if not policy.get("return_if_missing",False):
                    raise RuntimeError("First pickup not detected after reward retry budget")
                # Allow the last click to finish without sending another one.
                self.wait(policy["retry_delay"])
                if not self.match("result").found:
                    raise RuntimeError("Mission Result disappeared after reward retry budget")
                if not self.received(baseline):
                    reason = "First pickup unconfirmed after reward retry budget; returning to Mission Camp"
                    self.record_mission_outcome("NO_REWARD",reason)
                    self.session.event("reward_skipped",reason=reason,retries=policy["additional_retries"],
                                       attempts=policy["additional_retries"]+1)
                    return
        finally:
            baseline.close()
        self.session.event("reward_verified",slot=1,verification="Your Items image changed in consecutive checks")
        self.record_mission_outcome("PASS_WITH_REWARD","First reward pickup verified in consecutive checks")
        for index,point in enumerate(policy["slots_client"][1:],start=2):
            self.wait(policy["between_slots"])
            if not self.match("result").found:
                raise RuntimeError("Mission Result disappeared before all configured slots were clicked")
            self.click(point,f"reward_{index}")
        self.wait(policy["between_slots"])
        self.session.event("reward_row_clicked",slots=len(policy["slots_client"]),verified_slots=[1])

    def execute_state(self):
        c,s = self.config,self.state
        if s not in ("VERIFY_START","VERIFY_RETURN_START"):
            self._return_confirmation = None
        if s == "VERIFY_START":
            proof = getattr(self,"_return_confirmation",None)
            self._return_confirmation = None  # Single use, even on failure or interruption.
            poll,confirmations = self.vision.start.config["poll_seconds"],self.vision.start.config["ready_confirmations"]
            if proof is not None and proof.get("kind") == "camp_checkpoint":
                self.reuse_camp_return(proof,poll,confirmations)
            else:
                self.wait_hud(poll,confirmations,proof)
        elif s == "PLAY_START":
            self.play("start")
        elif s == "OPEN_MISSION":
            self.open_mission()
        elif s == "OPEN_ROOM_LIST":
            target = "mission_list" if "mission_selection" in c else "create"
            self.advance("room_list",target)
        elif s == "SELECT_MISSION":
            self.advance("mission_list","mission_selected",point=c["mission_selection"]["point_client"],label="mission_selection")
        elif s == "CREATE_ROOM":
            self.advance("create","start")
        elif s == "START_STAGE":
            self.click_matched("start")
            self.stage_clicked = self.session.now()
        elif s == "WAIT_VOTE":
            self.wait_vote()
        elif s == "VOTE_STAGE":
            # Reconfirm the button in this state. A missing button polls safely;
            # a successful click is sent once, then only the settle wait runs.
            match = self.wait_vote()
            self.click(match.point,"vote")
            self.wait(c["vote"]["after_click_wait"])
        elif s == "WAIT_STAGE_LOADING":
            self.wait_loading(c["stage"]["loading_poll"])
        elif s == "VERIFY_STAGE":
            self.wait(c["stage"]["min_entry_wait"]-(self.session.now()-self.stage_clicked))
            self.wait_hud(c["stage"]["time_poll"],c["stage"]["confirmations"])
        elif s == "CALIBRATE_CAMERA":
            self.calibrate_camera()
        elif s == "PLAY_MISSION":
            self.play("mission")
        elif s == "WAIT_RESULT":
            self.wait_result()
        elif s == "COLLECT_REWARDS":
            self.collect()
        elif s == "RETURN_CAMP":
            self.reset_return_verification()
            self.click_matched("camp")
            self._return_clicked_loop = self.session.loop
        elif s == "WAIT_RETURN_LOADING":
            self.wait_loading(c["return"]["loading_poll"])
        elif s == "VERIFY_RETURN_START":
            self._return_confirmation = None
            pause = self.pause_marker()
            proof,first = (self.try_camp_return() if c["return"].get("camp_checkpoint",{}).get("enabled") else (None,None))
            if proof is not None:
                self._return_confirmation = proof
            else:
                self.wait_hud(c["return"]["time_poll"],c["return"]["confirmations"],
                              not_ready_poll=c["return"].get("not_ready_poll"),first_observation=first)
            loop = self.session.loop
            context = getattr(self.session,"capture_context",None)
            if (proof is None and self._return_clicked_loop == loop and self._return_loading_loop == loop and
                    self._return_loading_context == context and pause == self.pause_marker() and
                    not pause[1] and c["return"]["confirmations"] >= 2 and
                    not (self.session.api.held_keys or self.session.api.held_buttons)):
                self._return_confirmation = {"loop":loop,"at":self._last_hud_at,"pause":pause,
                                             "context":context,"confirmations":c["return"]["confirmations"],
                                             "poll":c["return"]["time_poll"]}
        else:
            raise ValueError("Unknown state: " + s)
        self.check()

    def run(self,through=None):
        states = mission_states(self.config)
        if through is not None and through not in states:
            raise ValueError("Stopping state is not enabled by this mission config")
        result = None
        self.reset_return_verification()
        self.reset_mission_counts()
        try:
            for loop in range(1,self.config["max_loops"]+1):
                self._mission_outcome = None
                for self.state in states:
                    self.session.enter(self.state,loop)
                    self.timings = {}
                    self.session.log.reset_timing()
                    started,wall = self.session.now(),time.perf_counter()
                    self.deadline = started+self.config["timeouts"][self.state]
                    self.session.event("state_entry",timeout=self.config["timeouts"][self.state])
                    try:
                        self.check()
                        self.execute_state()
                    except BaseException as exc:
                        self.session.event("state_result",status="FAIL",reason=f"{type(exc).__name__}: {exc}",
                                           elapsed=self.session.now()-started,wall_elapsed=time.perf_counter()-wall,timing=self.timing_summary())
                        raise
                    reason = "Expected state result satisfied"
                    if self.state == "COLLECT_REWARDS":
                        reason = {"FAIL":"Mission defeated; rewards skipped",
                                  "NO_REWARD":"Reward pickup unconfirmed; returning to Mission Camp"}.get(
                                      self._mission_outcome,reason)
                    self.session.event("state_result",status="SUCCESS",reason=reason,
                                       elapsed=self.session.now()-started,wall_elapsed=time.perf_counter()-wall,timing=self.timing_summary())
                    if through == self.state and self.state != states[-1]:
                        result = FlowResult("SUCCESS",f"Stopped after requested step: {through}",self.state,self.completed)
                        break
                if result is not None:
                    break
                self.completed += 1
            if result is None:
                result = FlowResult("SUCCESS","Loop limit reached; return verified",self.state,self.completed)
        except (Exception,KeyboardInterrupt) as exc:
            result = FlowResult("FAIL",f"{type(exc).__name__}: {exc}",self.state,self.completed,
                                isinstance(exc,(StopRequested,KeyboardInterrupt)))
        finally:
            self.reset_return_verification()
            try:
                self.session.api.release_all()
            except Exception as exc:
                reason = f"Input release failed: {exc}"
                if result is not None:
                    reason = result.reason+"; "+reason
                result = FlowResult("FAIL",reason,self.state,self.completed,
                                    result.stopped if result is not None else False)
            if result is not None and result.status == "FAIL":
                try:
                    self.session.debug("stopped" if result.stopped else "failure")
                except Exception as exc:
                    self.session.event("debug_error",reason=str(exc))
        self.session.event("run_result",status=result.status,reason=result.reason,loops_completed=result.loops_completed,
                           **self.mission_counts)
        return result
