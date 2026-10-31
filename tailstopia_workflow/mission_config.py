import json
import math
from pathlib import Path

STATES = ("VERIFY_START","PLAY_START","OPEN_MISSION","OPEN_ROOM_LIST","CREATE_ROOM","START_STAGE",
          "WAIT_STAGE_LOADING","VERIFY_STAGE","CALIBRATE_CAMERA","PLAY_MISSION","WAIT_RESULT","COLLECT_REWARDS",
          "RETURN_CAMP","WAIT_RETURN_LOADING","VERIFY_RETURN_START")


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
    for name in STATES:
        number(config["timeouts"][name],True)
    for section,keys in {"talk":["first_wait","retry_delay","retry_wait","fallback_wait"],
                         "ui":["settle","retry_delay","click_hold"],
                         "stage":["min_entry_wait","loading_poll","time_poll"],
                         "result":["after_mission_start","poll"],
                         "reward":["settle","retry_delay","verify_wait","between_slots","change_pixel_delta","empty_max_mean_error"],
                         "return":["loading_poll","time_poll"]}.items():
        for key in keys:
            number(config[section][key],key in {"poll","loading_poll","time_poll","verify_wait"})
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
    if len(config["reward"]["slots_client"]) != 7:
        raise ValueError("Exactly seven reward slots required")
    point(config["reward"]["pointer_rest_client"])
    point(config["talk"]["point_client"])
    if not 0 < config["reward"]["change_fraction"] <= 1:
        raise ValueError("Invalid reward change fraction")
    for name in ("talk","room_list","create","start","result","camp"):
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
