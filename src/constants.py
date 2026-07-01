ACTIVITY_ID_TO_NAME = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

ACTIVITY_NAME_TO_ID = {v: k for k, v in ACTIVITY_ID_TO_NAME.items()}

ACTIVE_ACTIVITIES = frozenset({
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
})

ACTIVE_ACTIVITY_IDS = frozenset({
    ACTIVITY_NAME_TO_ID[activity] for activity in ACTIVE_ACTIVITIES
})

REST_ACTIVITIES = frozenset({
    "SITTING",
    "STANDING",
    "LAYING",
})

REST_ACTIVITY_IDS = frozenset({
    ACTIVITY_NAME_TO_ID[activity] for activity in REST_ACTIVITIES
})

FOUR_CLASS_REST_ID = 4

STATUS_LABELS = ("correct", "compensatory", "rest")
