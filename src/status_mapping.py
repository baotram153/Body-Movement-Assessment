import numpy as np

from .constants import ACTIVE_ACTIVITIES, ACTIVITY_ID_TO_NAME

def _validate_activity(activity: int) -> None:
    activity_name = ACTIVITY_ID_TO_NAME.get(int(activity))
    if activity_name not in ACTIVE_ACTIVITIES:
        raise ValueError(f"Invalid activity: {activity}. Must be one of {ACTIVE_ACTIVITIES}")

def map_activity_to_status(activity: int, target: int) -> str:
    _validate_activity(target)

    # logic-based activity mapping
    activity_id = int(activity)
    target_id = int(target)
    activity_name = ACTIVITY_ID_TO_NAME[activity_id]
    if activity_id == target_id:
        return "correct"
    elif activity_name in ACTIVE_ACTIVITIES:
        return "compensatory"
    else:
        return "rest"

def map_many_to_status(activities: np.ndarray, target: int) -> np.ndarray:
    _validate_activity(target)
    return np.array([map_activity_to_status(activity, target) for activity in activities])
