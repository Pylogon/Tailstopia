def select_reward(items, priorities, fallback):
    rank = {name: index for index, name in enumerate(priorities)}
    known = [item for item in items if item["name"] in rank]
    if known:
        item = min(known, key=lambda item: (rank[item["name"]], item["y"], item["x"]))
        return {"type": "click", "x": item["x"], "y": item["y"]}, "priority: " + item["name"]
    if fallback is None:
        raise ValueError("Reward missing/unknown; no fallback configured")
    return {"type": "click", **fallback}, "configured fallback"
