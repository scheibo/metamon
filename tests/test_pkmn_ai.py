"""Unit tests for the pkmn.ai Metamon TaurosV0 stdio adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pytest

from metamon.pkmn_ai import (
    AgentConfig,
    MAX_INFO_LINE_CHARS,
    MAX_INFO_POLICY_ENTRIES,
    NUM_ACTIONS,
    PkmnAiMetamonSession,
    fallback_command_from_request,
    format_info_line,
    value_to_win_prob,
)


class _FakeEngine:
    def __init__(self, probs: np.ndarray | None = None, v_value: float = 1000.0):
        self.cfg = AgentConfig(
            model_path=Path("/nonexistent.onnx"),
            sample_actions=False,
            emit_info=True,
            win_mode="linear",
        )
        self.probs = probs if probs is not None else np.full((NUM_ACTIONS,), 1.0 / NUM_ACTIONS)
        self.v_value = v_value
        self.reset_calls = 0
        self.infer_calls = 0
        self.last_selected: int | None = None

    def reset_episode(self) -> None:
        self.reset_calls += 1

    def infer(self, state, legal_indices: Set[int]) -> Tuple[np.ndarray, np.ndarray, float]:
        self.infer_calls += 1
        p = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        for idx in legal_indices:
            p[idx] = self.probs[idx]
        s = float(p.sum())
        if s > 0:
            p /= s
        q = np.full((NUM_ACTIONS,), self.v_value, dtype=np.float32)
        return p, q, self.v_value

    def select_action_index(self, probs: np.ndarray, idx_to_cmd: dict[int, str]) -> int:
        best = max(idx_to_cmd.keys(), key=lambda i: float(probs[i]))
        self.last_selected = best
        return best


def _sample_turn1_request() -> dict:
    return {
        "active": [
            {
                "moves": [
                    {"move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "disabled": False},
                    {"move": "Hyper Beam", "id": "hyperbeam", "pp": 8, "maxpp": 8, "disabled": False},
                    {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "disabled": False},
                    {"move": "Blizzard", "id": "blizzard", "pp": 8, "maxpp": 8, "disabled": False},
                ]
            }
        ],
        "side": {
            "name": "P1",
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Tauros",
                    "details": "Tauros",
                    "condition": "353/353",
                    "active": True,
                    "stats": {"atk": 298, "def": 288, "spa": 238, "spd": 238, "spe": 318},
                    "moves": ["bodyslam", "hyperbeam", "earthquake", "blizzard"],
                    "baseAbility": "noability",
                    "item": "",
                },
                {
                    "ident": "p1: Chansey",
                    "details": "Chansey",
                    "condition": "703/703",
                    "active": False,
                    "stats": {"atk": 108, "def": 108, "spa": 308, "spd": 308, "spe": 198},
                    "moves": ["icebeam", "softboiled", "thunderwave", "seismictoss"],
                    "baseAbility": "noability",
                    "item": "",
                },
                {
                    "ident": "p1: Snorlax",
                    "details": "Snorlax",
                    "condition": "523/523",
                    "active": False,
                    "stats": {"atk": 318, "def": 228, "spa": 228, "spd": 228, "spe": 158},
                    "moves": ["bodyslam", "earthquake", "selfdestruct", "hyperbeam"],
                    "baseAbility": "noability",
                    "item": "",
                },
            ],
        },
        "rqid": 1,
    }


def test_value_to_win_prob_linear_logistic_and_logit() -> None:
    cfg_lin = AgentConfig(model_path=Path("/x"), win_mode="linear", win_offset=0.0, win_scale=2000.0)
    assert value_to_win_prob(0.0, cfg_lin) == pytest.approx(0.0)
    assert value_to_win_prob(1000.0, cfg_lin) == pytest.approx(0.5)
    assert value_to_win_prob(2500.0, cfg_lin) == pytest.approx(1.0)
    assert value_to_win_prob(-200.0, cfg_lin) == pytest.approx(0.0)
    assert value_to_win_prob(float("nan"), cfg_lin) == pytest.approx(0.5)

    cfg_log = AgentConfig(model_path=Path("/x"), win_mode="logistic", win_slope=0.004, win_intercept=-5.6)
    assert value_to_win_prob(1400.0, cfg_log) == pytest.approx(0.5, abs=1e-6)
    assert 0.9 < value_to_win_prob(2000.0, cfg_log) < 1.0
    assert 0.0 < value_to_win_prob(200.0, cfg_log) < 0.1

    cfg_logit = AgentConfig(model_path=Path("/x"), win_mode="logit", win_scale=2000.0, win_slope=1.0697, win_intercept=-0.5143)
    assert 0.48 < value_to_win_prob(1235.0, cfg_logit) < 0.52
    assert value_to_win_prob(2000.0, cfg_logit) > 0.99
    assert value_to_win_prob(0.0, cfg_logit) < 0.01


def test_format_info_line_caps_policy_entries_and_validates() -> None:
    raw_policy = {f"move {i}": 0.1 * i for i in range(1, 5)}
    raw_policy.update({f"switch {i}": 0.05 * i for i in range(2, 7)})  # 9 entries total
    line = format_info_line(0.61234, raw_policy)
    assert line is not None
    assert line.startswith("/info ")
    assert len(line) <= MAX_INFO_LINE_CHARS
    payload = json.loads(line[6:])
    assert payload["win"] == pytest.approx(0.6123)
    assert len(payload["policy"]) == MAX_INFO_POLICY_ENTRIES
    assert all(0.0 < v <= 1.0 for v in payload["policy"].values())
    assert sum(payload["policy"].values()) == pytest.approx(1.0, abs=0.01)


def test_fallback_command_respects_force_switch_and_excluded() -> None:
    req = _sample_turn1_request()
    assert fallback_command_from_request(req) == "move 1"
    assert fallback_command_from_request(req, {"move 1", "move 2", "move 3", "move 4"}) == "switch 2"

    fs_req = {"forceSwitch": [True], "side": req["side"]}
    assert fallback_command_from_request(fs_req) == "switch 2"
    assert fallback_command_from_request(fs_req, {"switch 2"}) == "switch 3"


def test_session_defers_turn1_request_handles_utm_and_emits_info() -> None:
    probs = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    probs[1] = 0.7  # alphabetical sorted move[1]: Body Slam (Blizzard=0, Body Slam=1, Earthquake=2, Hyper Beam=3)
    probs[0] = 0.2
    probs[4] = 0.1  # switch[0]: Chansey
    engine = _FakeEngine(probs=probs, v_value=1200.0)
    infos: list[str] = []
    session = PkmnAiMetamonSession(engine=engine, info_sink=infos.append)

    req = _sample_turn1_request()
    assert session.handle_line("|utm|Tauros|||noability|bodyslam,hyperbeam,earthquake,blizzard|||||||") is None
    assert session.handle_line(f"|request|{json.dumps(req)}") is None
    assert engine.infer_calls == 0  # must wait until |turn|1 so opp active exists

    for line in (
        "|player|p1|P1||",
        "|player|p2|P2||",
        "|teamsize|p1|3",
        "|teamsize|p2|3",
        "|gametype|singles",
        "|gen|1",
        "|tier|[Gen 1] OU",
        "|start",
        "|switch|p1a: Tauros|Tauros|353/353",
        "|switch|p2a: Alakazam|Alakazam|100/100",
    ):
        assert session.handle_line(line) is None

    choice = session.handle_line("|turn|1")
    assert choice == "move 1"  # Body Slam is slot 1 in request
    assert engine.infer_calls == 1
    assert len(infos) == 1
    info = json.loads(infos[0][6:])
    assert info["win"] == pytest.approx(0.6)
    assert info["policy"]["move 1"] == pytest.approx(0.7, abs=0.01)


def test_invalid_choice_retry_picks_next_best_without_advancing_kv_cache() -> None:
    probs = np.zeros((NUM_ACTIONS,), dtype=np.float32)
    probs[1] = 0.7  # Body Slam -> move 1
    probs[0] = 0.2  # Blizzard -> move 4
    probs[4] = 0.1  # Chansey -> switch 2
    engine = _FakeEngine(probs=probs, v_value=1000.0)
    session = PkmnAiMetamonSession(engine=engine)

    req = _sample_turn1_request()
    for line in (
        f"|request|{json.dumps(req)}",
        "|player|p1|P1||",
        "|player|p2|P2||",
        "|teamsize|p1|3",
        "|teamsize|p2|3",
        "|gen|1",
        "|tier|[Gen 1] OU",
        "|start",
        "|switch|p1a: Tauros|Tauros|353/353",
        "|switch|p2a: Alakazam|Alakazam|100/100",
    ):
        session.handle_line(line)

    assert session.handle_line("|turn|1") == "move 1"
    assert engine.infer_calls == 1

    # Simulate arbiter [Invalid choice] + resent |request|
    assert session.handle_line("|error|[Invalid choice] Can't move: Body Slam is disabled") is None
    retry_choice = session.handle_line(f"|request|{json.dumps(req)}")
    assert retry_choice == "move 4"  # Blizzard (second highest probability)
    assert engine.infer_calls == 1  # KV cache NOT double-advanced


def test_session_resets_cleanly_on_new_battle() -> None:
    engine = _FakeEngine()
    session = PkmnAiMetamonSession(engine=engine)
    initial_resets = engine.reset_calls

    req = _sample_turn1_request()
    for line in (
        f"|request|{json.dumps(req)}",
        "|player|p1|P1||",
        "|player|p2|P2||",
        "|teamsize|p1|3",
        "|teamsize|p2|3",
        "|gen|1",
        "|tier|[Gen 1] OU",
        "|start",
        "|switch|p1a: Tauros|Tauros|353/353",
        "|switch|p2a: Alakazam|Alakazam|100/100",
        "|turn|1",
    ):
        session.handle_line(line)

    session.handle_line("|win|P1")
    assert session.battle is not None and session.battle.finished

    session.handle_line("|init|battle")
    assert engine.reset_calls == initial_resets + 1
    assert session.battle is not None and not session.battle.finished
    assert not session.saw_turn_1
