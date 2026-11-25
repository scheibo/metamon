from . import forward
from . import replay_state
from . import str_parsing
from . import checks
from . import exceptions


def __getattr__(name: str):
    if name == "backward":
        from . import backward

        return backward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
