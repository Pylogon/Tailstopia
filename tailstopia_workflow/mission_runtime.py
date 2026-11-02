"""Live/fixture sessions share the same controller and actual image recognition."""
from pathlib import Path
import string
import time

from PIL import Image, ImageGrab

from .logger import ExecutionLog


class InputTrace:
    def __init__(self,backend,event):
        self.backend,self.event = backend,event
        self.held_keys,self.held_buttons = set(),set()

    @property
    def KEYBOARD_KEYS(self):
        return self.backend.KEYBOARD_KEYS

    @property
    def FAILSAFE(self):
        return self.backend.FAILSAFE

    @FAILSAFE.setter
    def FAILSAFE(self,value):
        self.backend.FAILSAFE = value

    def failSafeCheck(self):
        self.backend.failSafeCheck()

    def keyDown(self,key,**kwargs):
        self.held_keys.add(key)
        self.backend.keyDown(key,**kwargs)
        self.event("action",method="keyDown",key=key)

    def keyUp(self,key,**kwargs):
        self.backend.keyUp(key,**kwargs)
        self.held_keys.discard(key)
        self.event("action",method="keyUp",key=key)

    def mouseDown(self,**kwargs):
        self.held_buttons.add(kwargs.get("button","left"))
        self.backend.mouseDown(**kwargs)
        self.event("action",method="mouseDown",params=kwargs)

    def mouseUp(self,**kwargs):
        self.backend.mouseUp(**kwargs)
        self.held_buttons.discard(kwargs.get("button","left"))
        self.event("action",method="mouseUp",params=kwargs)

    def moveTo(self,*args,**kwargs):
        self.backend.moveTo(*args,**kwargs)
        self.event("action",method="moveTo",args=args)

    def scroll(self,*args,**kwargs):
        self.backend.scroll(*args,**kwargs)
        self.event("action",method="scroll",args=args)

    def hscroll(self,*args,**kwargs):
        self.backend.hscroll(*args,**kwargs)
        self.event("action",method="hscroll",args=args)

    def release_all(self):
        errors = []
        enabled = self.FAILSAFE
        self.FAILSAFE = False
        try:
            for key in list(self.held_keys):
                try:
                    self.keyUp(key,_pause=False)
                except Exception as exc:
                    errors.append(str(exc))
            for button in list(self.held_buttons):
                try:
                    self.mouseUp(button=button,_pause=False)
                except Exception as exc:
                    errors.append(str(exc))
        finally:
            self.FAILSAFE = enabled
        if errors:
            raise RuntimeError("Release failed: " + "; ".join(errors))


class Session:
    def __init__(self,config,directory,backend):
        self.config = config
        self.log = ExecutionLog(directory)
        self.state,self.loop = "PREFLIGHT",0
        self.api = InputTrace(backend,self.event)
        self.last_image = None
        self.last_box = None
        self.capture_context = None

    def event(self,kind,**fields):
        self.log.emit(kind,state=self.state,loop=self.loop,active_at=self.now(),**fields)
        if kind in {"state_entry","state_result","retry","pause","resume","reward_verified","camera_calibrated","mission_outcome","run_result",
                    "camp_return_verified","camp_return_fallback"}:
            print(f"[{self.loop}:{self.state}] {kind} {fields}",flush=True)

    def prepare_macro(self,name):
        # Fixture sessions do not read or modify the OS keyboard layout.
        pass

    def enter(self,state,loop):
        self.state,self.loop = state,loop

    def capture(self,box):
        raise NotImplementedError

    def click(self,point,label):
        self.check()
        x,y = self.desktop(point)
        self.event("click",label=label,client_point=point,desktop_point=[x,y])
        try:
            self.api.mouseDown(x=x,y=y,button="left",_pause=False)
            until = self.now()+self.config["ui"]["click_hold"]
            while self.now() < until:
                self.check()
                self.sleep(min(.01,until-self.now()))
            self.check()
            self.api.mouseUp(button="left",_pause=False)
        finally:
            if "left" in self.api.held_buttons:
                self.api.release_all()
        self.click_effect(label)

    def click_effect(self,label):
        pass

    def rest_pointer(self):
        self.check()
        point = self.desktop(self.config["reward"]["pointer_rest_client"])
        self.api.moveTo(*point,_pause=False)

    def debug(self,label):
        if self.last_image is not None:
            path = self.log.directory/f"{self.loop}_{self.state}_{label}.png"
            self.last_image.save(path)
            self.event("debug_image",path=str(path),roi=self.last_box)
        else:
            self.event("debug_error",reason="No frame captured yet")


class LiveSession(Session):
    def __init__(self,config,directory,window):
        # GameWindow configures DPI awareness before PyAutoGUI is imported.
        import pyautogui
        pyautogui.FAILSAFE,pyautogui.PAUSE = True,0
        self.window = window
        self.paused_total = 0.0
        self.pause_started = None
        self.pause_requested = False
        self.was_f7 = self.window.pressed(0x76)
        super().__init__(config,directory,pyautogui)

    def prepare_macro(self,name):
        from .game_window import require_english_keyboard
        self.check()
        try:
            language = require_english_keyboard(self.window.api,self.window.hwnd)
        except RuntimeError as exc:
            self.event("keyboard_layout",status="FAIL",recording=name,reason=str(exc))
            raise
        self.event("keyboard_layout",status="SUCCESS",recording=name,language_id=f"0x{language:04X}")

    def now(self):
        real = time.monotonic()
        return real-self.paused_total-(real-self.pause_started if self.pause_started is not None else 0)

    def sleep(self,seconds):
        time.sleep(max(0,seconds))

    def check(self):
        self.window.check()
        self.api.failSafeCheck()
        f7 = self.window.pressed(0x76)
        if f7 and not self.was_f7:
            self.pause_requested = not self.pause_requested
            self.event("pause_request",pending=self.pause_requested)
            print("Pause requested; active holds finish before pause." if self.pause_requested else "Pause cancelled",flush=True)
        self.was_f7 = f7
        if self.pause_requested and not (self.api.held_keys or self.api.held_buttons):
            self.pause_started = time.monotonic()
            self.event("pause",reason="F7; F7 resumes, F9 stops")
            try:
                while self.pause_requested:
                    self.window.check()
                    f7 = self.window.pressed(0x76)
                    if f7 and not self.was_f7:
                        self.pause_requested = False
                    self.was_f7 = f7
                    time.sleep(.02)
            finally:
                self.paused_total += time.monotonic()-self.pause_started
                self.pause_started = None
            self.event("resume")

    def desktop(self,point):
        left,top,_,_ = self.window.geometry()
        return left+point[0],top+point[1]

    def capture(self,box):
        self.check()
        left,top,width,height = self.window.geometry()
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise RuntimeError("Capture ROI outside game client")
        image = ImageGrab.grab(bbox=(left+x1,top+y1,left+x2,top+y2),all_screens=True)
        if self.last_image is not None:
            self.last_image.close()
        self.last_image,self.last_box = image.copy(),list(box)
        # Reuse geometry already read for this capture; no extra OS query.
        self.capture_context = (self.window.hwnd,left,top,width,height)
        self.check()
        return image


    def debug(self,label):
        super().debug(label)
        # A fresh client screenshot is diagnostic only, after inputs are released.
        # Do not capture an unrelated foreground application after focus loss.
        try:
            if self.window.api.GetForegroundWindow() != self.window.hwnd or self.window.api.IsIconic(self.window.hwnd):
                self.event("debug_error",reason="Game not foreground; kept last ROI only")
                return
            left,top,width,height = self.window.geometry()
            with ImageGrab.grab(bbox=(left,top,left+width,top+height),all_screens=True) as image:
                path = self.log.directory/f"{self.loop}_{self.state}_{label}_client.png"
                image.save(path)
            self.event("debug_image",path=str(path),scope="fresh client")
        except Exception as exc:
            self.event("debug_error",reason=f"Fresh client capture failed: {exc}")


class SimulatedInput:
    FAILSAFE = True
    KEYBOARD_KEYS = list(string.ascii_lowercase+string.digits)+["space","enter","tab","esc","shift","ctrl","alt"]
    def failSafeCheck(self):
        pass
    def keyDown(self,*args,**kwargs):
        pass
    def keyUp(self,*args,**kwargs):
        pass
    def mouseDown(self,*args,**kwargs):
        pass
    def mouseUp(self,*args,**kwargs):
        pass
    def moveTo(self,*args,**kwargs):
        pass
    def scroll(self,*args,**kwargs):
        pass
    def hscroll(self,*args,**kwargs):
        pass


class FixtureSession(Session):
    """Fast, no OS input, real templates. Reward acquisition is SYNTHETIC."""
    def __init__(self,config,directory,resource):
        self.resource = Path(resource)
        self.virtual_time = 0.0
        self.frame = "intiate.png"
        self.received = False
        self.click_counts = {}
        super().__init__(config,directory,SimulatedInput())

    def now(self):
        return self.virtual_time

    def sleep(self,seconds):
        self.virtual_time += max(0,seconds)

    def check(self):
        pass

    def desktop(self,point):
        x,y = self.config["reference_client_origin"]
        return x+point[0],y+point[1]

    def enter(self,state,loop):
        super().enter(state,loop)
        frames = {"VERIFY_START":"intiate.png","OPEN_MISSION":"red panda2.png",
                  "OPEN_ROOM_LIST":"mission.png","CREATE_ROOM":"room list.png","START_STAGE":"startmission.png",
                  "WAIT_STAGE_LOADING":"loading.png","VERIFY_STAGE":"startinagme .png",
                  "CALIBRATE_CAMERA":"screenshot_18.png",
                  "PLAY_MISSION":"startinagme .png","WAIT_RESULT":"reward1.png",
                  "COLLECT_REWARDS":"reward1.png","RETURN_CAMP":"reward1.png",
                  "WAIT_RETURN_LOADING":"loading.png","VERIFY_RETURN_START":"intiate.png"}
        if "vote" in self.config:
            frames.update({"VERIFY_START":"95/talk.png","OPEN_MISSION":"95/talk.png",
                           "WAIT_VOTE":"95/vote.png","VOTE_STAGE":"95/vote.png",
                           "PLAY_MISSION":"95/vote.png","WAIT_RESULT":"95/reward.png",
                           "COLLECT_REWARDS":"95/reward.png","RETURN_CAMP":"95/reward.png",
                           "VERIFY_RETURN_START":"95/talk.png"})
        if "mission_selection" in self.config:
            frames.update({"SELECT_MISSION":"95/create.png","CREATE_ROOM":"95/create.png"})
        if state in frames:
            self.frame = frames[state]
        self.click_counts = {}
        if state == "COLLECT_REWARDS":
            self.received = False

    def click_effect(self,label):
        self.click_counts[label] = self.click_counts.get(label,0)+1
        transitions = {"talk":"mission.png","room_list":"room list.png","create":"startmission.png",
                       "start":"loading.png","camp":"loading.png"}
        if "mission_selection" in self.config:
            transitions.update({"room_list":"95/create.png","mission_selection":"95/create.png"})
        if self.config["talk"].get("opened_target","room_list") == "mission_list":
            transitions["talk"] = "95/create.png"
        self.frame = transitions.get(label,self.frame)
        if label == "reward_1":
            self.received = True

    def capture(self,box):
        with Image.open(self.resource/self.frame) as source:
            # Camera reference is client-only; older fixtures include borders.
            origin = (0,0) if self.frame == "screenshot_18.png" else self.config["reference_client_origin"]
            area = (box[0]+origin[0],box[1]+origin[1],box[2]+origin[0],box[3]+origin[1])
            image = source.crop(area).convert("RGB")
            if self.received and Path(self.frame).name.startswith("reward") and tuple(box) == tuple(self.config["reward"]["received_box_client"]):
                # Reference assets have empty Your Items slots only. Place a real
                # icon crop into that slot to exercise image-change verification.
                x,y = self.config["reward"]["slots_client"][0]
                w,h = image.size
                icon = source.crop((origin[0]+x-w//2,origin[1]+y-h//2,
                                    origin[0]+x-w//2+w,origin[1]+y-h//2+h)).convert("RGB")
                image.paste(icon,(0,0))
                icon.close()
        if self.last_image is not None:
            self.last_image.close()
        self.last_image,self.last_box = image.copy(),list(box)
        return image
