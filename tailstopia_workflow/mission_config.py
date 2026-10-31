import json
import math
from pathlib import Path

STATES = ("VERIFY_START","PLAY_START","OPEN_MISSION","OPEN_ROOM_LIST","CREATE_ROOM","START_STAGE",
          "WAIT_STAGE_LOADING","VERIFY_STAGE","CALIBRATE_CAMERA","PLAY_MISSION","WAIT_RESULT","COLLECT_REWARDS",
          "RETURN_CAMP","WAIT_RETURN_LOADING","VERIFY_RETURN_START")


VOTE_STATES = ("WAIT_VOTE","VOTE_STAGE")
ALL_STATES = STATES+VOTE_STATES+("SELECT_MISSION",)


def mission_states(config):
    states = STATES
    if "vote" in config:
        # Replace the desert HUD/camera gate with this stage's visual evidence.
        first = STATES.index("WAIT_STAGE_LOADING")
        mission = STATES.index("PLAY_MISSION")
        states = STATES[:first]+VOTE_STATES+STATES[mission:]
    if "mission_selection" in config:
        create = states.index("CREATE_ROOM")
        states = states[:create]+("SELECT_MISSION",)+states[create:]
    if config["talk"].get("opened_target","room_list") == "mission_list":
        states = tuple(state for state in states if state != "OPEN_ROOM_LIST")
    return states


TALK_RECOVERY_DEFAULTS = {"max_attempts":2,"hold_seconds":0.25,"confirm_wait":0.5,"settle_seconds":0.5}


def load_mission_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_mission_config(config)
    return config,path.parent


def validate_mission_config(config):
    def number(value,positive=False):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError("Invalid finite nonnegative number")
    def count(value,minimum=0):
        if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
            raise ValueError("Invalid count")
    def point(value):
        if len(value) != 2 or any(isinstance(v,bool) or not isinstance(v,int) or v < 0 for v in value):
            raise ValueError("Invalid client point")
    count(config["max_loops"],1)
    if not config["window_title"].strip():
        raise ValueError("Empty window title")
    point(config["reference_client_origin"])
    opened_target = config["talk"].get("opened_target","room_list")
    if opened_target not in ("room_list","mission_list"):
        raise ValueError("Talk opened_target must be room_list or mission_list")
    if opened_target not in config["targets"]:
        raise ValueError("Talk opened_target requires a configured target")
    for name in mission_states(config):
        number(config["timeouts"][name],True)
    if "vote" in config:
        vote = config["vote"]
        for key in ("first_wait","poll","after_click_wait"):
            number(vote[key],key == "poll")
        if "vote" not in config["targets"]:
            raise ValueError("Vote entry requires a vote target")
    if "mission_selection" in config:
        point(config["mission_selection"]["point_client"])
        if not {"mission_list","mission_selected"}.issubset(config["targets"]):
            raise ValueError("Mission selection requires list and selected-title targets")
    for section,keys in {"talk":["first_wait","fallback_wait"],
                         "ui":["settle","retry_delay","click_hold"],
                         "stage":["min_entry_wait","loading_poll","time_poll"],
                         "result":["after_mission_start","poll"],
                         "reward":["settle","retry_delay","verify_wait","between_slots","change_pixel_delta","empty_max_mean_error"],
                         "return":["loading_poll","time_poll"]}.items():
        for key in keys:
            number(config[section][key],key in {"poll","loading_poll","time_poll","verify_wait"})
    # Polling has its own window after first_wait; the state deadline still applies.
    number(config["talk"].get("poll_timeout",5),True)
    number(config["talk"].get("poll_interval",0.5),True)
    recovery = {**TALK_RECOVERY_DEFAULTS,**config["talk"].get("recovery",{})}
    count(recovery["max_attempts"])
    for key in ("hold_seconds","confirm_wait","settle_seconds"):
        number(recovery[key],True)
    if recovery["hold_seconds"] > 0.3:
        raise ValueError("Talk recovery hold_seconds must not exceed 0.3 seconds")
    for section in ("stage","result","reward","return"):
        count(config[section]["confirmations"],1)
    count(config["ui"]["additional_retries"])
    count(config["reward"]["additional_retries"])
    camera = config["camera"]
    number(camera["rotation_seconds"],True)
    number(camera["settle_seconds"],True)
    count(camera["max_rotations"])
    count(camera["confirmations"],2)
    count(camera["min_blue_blob_pixels"],1)
    for key in ("rotation_key","left_rotation_key"):
        if camera[key] not in ("c","z"):
            raise ValueError("Camera rotation keys must be c or z")
    box = camera["search_roi_client"]
    if len(box) != 4:
        raise ValueError("Invalid camera box")
    point(box[:2])
    point(box[2:])
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("Empty camera box")
    if camera["min_blue_blob_pixels"] > (box[2]-box[0])*(box[3]-box[1]):
        raise ValueError("Camera blue patch exceeds ROI area")
    direction_box = camera["direction_roi_client"]
    if len(direction_box) != 4:
        raise ValueError("Invalid camera direction ROI")
    point(direction_box[:2])
    point(direction_box[2:])
    if not (direction_box[0] < box[0] < box[2] < direction_box[2]
            and direction_box[1] == box[1] and direction_box[3] == box[3]):
        raise ValueError("Direction ROI must extend both sides of center at the same height")
    lower,upper = camera["blue_hsv_lower"],camera["blue_hsv_upper"]
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("Camera HSV bounds need three channels")
    for low,high,limit in zip(lower,upper,(179,255,255)):
        count(low)
        count(high)
        if not low <= high <= limit:
            raise ValueError("Invalid camera HSV bounds")
    for value in config["reward"]["slots_client"]:
        point(value)
    if not 1 <= len(config["reward"]["slots_client"]) <= 7:
        raise ValueError("One to seven reward slots required")
    point(config["reward"]["pointer_rest_client"])
    point(config["talk"]["point_client"])
    if not 0 < config["reward"]["change_fraction"] <= 1:
        raise ValueError("Invalid reward change fraction")
    targets = ("talk",opened_target,"create","start","result","camp")
    if "vote" in config:
        targets += ("vote",)
    if "mission_selection" in config:
        targets += ("mission_list","mission_selected")
    for name in targets:
        number(config["targets"][name]["threshold"],True)
        if not 0 < config["targets"][name]["threshold"] <= 1:
            raise ValueError("Invalid template threshold")
    talk = config["targets"]["talk"]
    if "template_scales" in talk:
        if not isinstance(talk["template_scales"],list) or not talk["template_scales"]:
            raise ValueError("Talk template_scales must be a nonempty list")
        for scale in talk["template_scales"]:
            number(scale,True)
    if "glyph_hsv_lower" in talk or "glyph_hsv_upper" in talk:
        lower,upper = talk.get("glyph_hsv_lower",[]),talk.get("glyph_hsv_upper",[])
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("Talk glyph HSV bounds need three channels")
        for low,high,limit in zip(lower,upper,(179,255,255)):
            count(low)
            count(high)
            if not low <= high <= limit:
                raise ValueError("Invalid Talk glyph HSV bounds")
    return config
