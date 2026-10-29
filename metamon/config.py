import os

# officially supported Showdown rulesets.
# gen1-4 nu/uu/ubers were supported by the original metamon paper,
# but their rules change often and their datasets are rarely updated,
# so our models would be very out-of-sync with current metagame.
SUPPORTED_BATTLE_FORMATS = [
    "gen1ou",
    "gen1uu",
    "gen1nu",
    "gen1ubers",
    "gen2ou",
    "gen2uu",
    "gen2nu",
    "gen2ubers",
    "gen3ou",
    "gen3uu",
    "gen3nu",
    "gen3ubers",
    "gen4ou",
    "gen4uu",
    "gen4nu",
    "gen4ubers",
    "gen9ou",
]


# play format A without OOD inputs by telling the agent it is playing format B.
FORMAT_ALIASES = {
    "gen1oulongtimer": "gen1ou",
    "gen9oulongtimer": "gen9ou",
}


def format_for_agent(fmt: str) -> str:
    """Lets metamon play non-standard Showdown formats by pretending they're something else"""
    return FORMAT_ALIASES.get(fmt.lower(), fmt.lower())


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_dotenv_path = os.path.join(_repo_root, ".env")
if os.path.exists(_dotenv_path):
    with open(_dotenv_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

if "METAMON_CACHE_DIR" not in os.environ:
    os.environ["METAMON_CACHE_DIR"] = os.path.join(_repo_root, ".cache")
elif not os.path.isabs(os.environ["METAMON_CACHE_DIR"]):
    os.environ["METAMON_CACHE_DIR"] = os.path.join(
        _repo_root, os.environ["METAMON_CACHE_DIR"]
    )

METAMON_CACHE_DIR = os.environ.get("METAMON_CACHE_DIR", None)
if METAMON_CACHE_DIR is not None:
    os.makedirs(METAMON_CACHE_DIR, exist_ok=True)
