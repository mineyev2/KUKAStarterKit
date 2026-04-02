from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    WAITING_FOR_NEXT_SCAN = auto()
    PLANNING_MOVE_TO_PRESCAN = auto()
    COMPUTING_MOVE_TO_PRESCAN = auto()
    MOVING_TO_PRESCAN = auto()
    WAITING_TO_GO_TO_START = auto()
    PLANNING_MOVE_TO_START = auto()
    COMPUTING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    PLANNING_ALONG_ALTERNATE_PATH = auto()
    COMPUTING_ALONG_ALTERNATE_PATH = auto()
    MOVING_ALONG_ALTERNATE_PATH = auto()
    COMPUTING_IKS = auto()
    PAUSE = auto()
    DONE = auto()
