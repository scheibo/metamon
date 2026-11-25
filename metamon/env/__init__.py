__all__ = [
    "TeamSet",
    "PokeAgentTeamSet",
    "PokeEnvWrapper",
    "BattleAgainstBaseline",
    "QueueOnLocalLadder",
    "ChallengeByUsername",
    "METAMON_TEAM_SETS",
    "PokeAgentServerConfiguration",
    "PokeAgentLadder",
    "get_metamon_teams",
]


def __getattr__(name: str):
    if name in __all__:
        from metamon.env import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
