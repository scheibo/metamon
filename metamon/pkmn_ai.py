"""Standalone stdio arena agent for Metamon TaurosV0 (Gen 1 OU) on https://pkmn.ai.

Runs the exported ONNX graph (folded SigmaReparam + 32-step sliding-window KV cache +
masked actor + 3-critic NCriticsTwoHot value head) via ONNX Runtime CPUExecutionProvider
with zero PyTorch, AMAGO, WebSocket, or filesystem-write dependencies.

Emits `/ready` after model load & warmup, full `/info` telemetry (`win` + `policy`) on
every decision, and bare slot-indexed Showdown choices (`move N`, `switch N`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import warnings

import numpy as np
import onnxruntime as ort

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.backend.showdown_dex import Dex
from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.interface import (
    AggressiveShapedReward,
    DefaultActionSpace,
    GroupedObservationSpace,
    TokenizedObservationSpace,
    UniversalAction,
    UniversalState,
)
from metamon.tokenizer import get_tokenizer

ort.set_default_logger_severity(3)
if hasattr(ort, "disable_telemetry_events"):
    ort.disable_telemetry_events()

logger = logging.getLogger("metamon.pkmn_ai")

NUM_ACTIONS = 13
WINDOW_SIZE = 32
N_LAYERS = 4
N_HEADS = 8
HEAD_DIM = 88
MAX_INFO_POLICY_ENTRIES = 8
MAX_INFO_LINE_CHARS = 2048

_POKEMON_TEXT_KEYS = (
    "text_active_pokemon_tokens",
    "text_switch_0_tokens",
    "text_switch_1_tokens",
    "text_switch_2_tokens",
    "text_switch_3_tokens",
    "text_switch_4_tokens",
    "text_opponent_active_pokemon_tokens",
)
_POKEMON_NUM_KEYS = (
    "numbers_active_pokemon",
    "numbers_switch_0",
    "numbers_switch_1",
    "numbers_switch_2",
    "numbers_switch_3",
    "numbers_switch_4",
    "numbers_opponent_active_pokemon",
)

_ID_RE = re.compile(r"[^a-z0-9]+")


def _to_id(text: str) -> str:
    return _ID_RE.sub("", text.lower())


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _default_model_path() -> Path:
    env_path = os.environ.get("METAMON_MODEL_PATH")
    if env_path:
        return Path(env_path)
    repo_or_bundle = Path(__file__).resolve().parent.parent
    candidates = (
        repo_or_bundle / "model" / "tauros_v0_fp16.onnx",
        repo_or_bundle / "model" / "tauros_v0_fp32.onnx",
        repo_or_bundle / ".build-cache" / "onnx" / "tauros_v0_fp16.onnx",
        repo_or_bundle / ".build-cache" / "onnx" / "tauros_v0_fp32.onnx",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]


@dataclass
class AgentConfig:
    model_path: Path
    onnx_threads: int = 4
    sample_actions: bool = True
    temperature: float = 1.0
    emit_info: bool = True
    # Critic V(s) -> win probability calibration.
    # AggressiveShapedReward gives +200*10=+2000 for win and 0 for loss;
    # default logit-space Platt parameters calibrated on 100 M6 gen1ou self-play games
    # (13,784 turns, ECE 0.085 -> 0.024, Brier 0.188 -> 0.179).
    win_mode: str = "logit"  # "logit", "logistic", or "linear"
    win_offset: float = 0.0
    win_scale: float = 2000.0
    win_slope: float = 1.0697
    win_intercept: float = -0.5143
    seed: Optional[int] = None
    log_level: str = "WARNING"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        seed_raw = os.environ.get("METAMON_SEED")
        seed = int(seed_raw) if seed_raw and seed_raw.strip() else None
        return cls(
            model_path=_default_model_path(),
            onnx_threads=max(1, int(os.environ.get("METAMON_ONNX_THREADS", "4"))),
            sample_actions=_parse_bool_env("METAMON_SAMPLE", True),
            temperature=max(1e-4, float(os.environ.get("METAMON_TEMPERATURE", "1.0"))),
            emit_info=_parse_bool_env("METAMON_EMIT_INFO", True),
            win_mode=os.environ.get("METAMON_WIN_MODE", "logit").strip().lower(),
            win_offset=float(os.environ.get("METAMON_WIN_OFFSET", "0.0")),
            win_scale=float(os.environ.get("METAMON_WIN_SCALE", "2000.0")),
            win_slope=float(os.environ.get("METAMON_WIN_SLOPE", "1.0697")),
            win_intercept=float(os.environ.get("METAMON_WIN_INTERCEPT", "-0.5143")),
            seed=seed,
            log_level=os.environ.get("METAMON_LOG_LEVEL", "WARNING").upper(),
        )


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(z, 60.0)))
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def value_to_win_prob(v_value: float, cfg: AgentConfig) -> float:
    """Map critic expected return V(s) (gamma=0.999) to calibrated win probability [0, 1]."""
    if not math.isfinite(v_value):
        return 0.5
    scale = cfg.win_scale if abs(cfg.win_scale) > 1e-9 else 2000.0
    if cfg.win_mode == "logit":
        u = min(0.998, max(0.002, (v_value - cfg.win_offset) / scale))
        z = cfg.win_slope * math.log(u / (1.0 - u)) + cfg.win_intercept
        return _sigmoid(z)
    if cfg.win_mode == "logistic":
        return _sigmoid(cfg.win_slope * v_value + cfg.win_intercept)
    return min(1.0, max(0.0, (v_value - cfg.win_offset) / scale))


def valid_move_slots(req: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    if req.get("forceSwitch"):
        return []
    active = req.get("active")
    if not active or not isinstance(active, list):
        return []
    moves = active[0].get("moves") or []
    out: List[Tuple[int, Dict[str, Any]]] = []
    for idx, m in enumerate(moves, start=1):
        if not m.get("disabled"):
            out.append((idx, m))
    return out


def alive_switch_slots(req: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    active_list = req.get("active") or []
    if not req.get("forceSwitch") and active_list and active_list[0].get("trapped"):
        return []
    pokemon = (req.get("side") or {}).get("pokemon") or []
    out: List[Tuple[int, Dict[str, Any]]] = []
    for idx, p in enumerate(pokemon, start=1):
        if p.get("active"):
            continue
        cond = str(p.get("condition", "")).strip()
        if cond == "0 fnt" or cond.endswith(" fnt"):
            continue
        out.append((idx, p))
    return out


def fallback_command_from_request(
    req: Dict[str, Any], excluded: Optional[Set[str]] = None
) -> str:
    """Return a guaranteed-legal bare Showdown command directly from request JSON."""
    skip = excluded or set()
    if req.get("teamPreview"):
        team_len = len((req.get("side") or {}).get("pokemon") or []) or 6
        cmd = "team " + "".join(str(i) for i in range(1, team_len + 1))
        if cmd not in skip:
            return cmd

    for idx, _ in valid_move_slots(req):
        cmd = f"move {idx}"
        if cmd not in skip:
            return cmd
    for idx, _ in alive_switch_slots(req):
        cmd = f"switch {idx}"
        if cmd not in skip:
            return cmd
    return "default"


def map_legal_actions_to_commands(
    battle: MetamonBackendBattle,
    state: UniversalState,
    req: Dict[str, Any],
) -> Dict[int, str]:
    """Map every legal Metamon DefaultActionSpace index (0..12) to a bare Showdown slot command."""
    legal_universal = UniversalAction.definitely_valid_actions(state=state, battle=battle)
    req_moves = valid_move_slots(req)
    req_switches = alive_switch_slots(req)

    # Pre-index request moves and switches for exact 1-based slot lookup
    move_slot_by_id: Dict[str, int] = {}
    for slot, m in req_moves:
        mid = _to_id(str(m.get("id") or m.get("move") or ""))
        mname = _to_id(str(m.get("move") or ""))
        if mid and mid not in move_slot_by_id:
            move_slot_by_id[mid] = slot
        if mname and mname not in move_slot_by_id:
            move_slot_by_id[mname] = slot

    switch_slots_meta: List[Tuple[int, str, str, str]] = []
    for slot, p in req_switches:
        details_species = str(p.get("details", "")).split(",")[0].strip()
        ident_name = str(p.get("ident", "")).split(":", 1)[-1].strip()
        switch_slots_meta.append(
            (
                slot,
                _to_id(details_species),
                pokemon_name(details_species),
                _to_id(ident_name),
            )
        )

    idx_to_cmd: Dict[int, str] = {}
    for ua in sorted(legal_universal, key=lambda a: a.action_idx):
        idx = ua.action_idx
        if idx < 0 or idx >= NUM_ACTIONS:
            continue
        order = UniversalAction.action_idx_to_BattleOrder(battle, idx)
        if order is None or order.order is None:
            continue
        target = order.order

        if idx <= 3 or idx >= 9:
            # Move action (indices 9..12 unused in gen1ou, handled safely if ever legal)
            if not req_moves:
                continue
            target_id = _to_id(str(getattr(target, "id", "") or getattr(target, "name", "")))
            if target_id in ("recharge", "struggle", "fight"):
                slot = req_moves[0][0]
            else:
                slot = move_slot_by_id.get(target_id)
                if slot is None:
                    norm_m = _to_id(move_name(str(getattr(target, "name", ""))))
                    slot = move_slot_by_id.get(norm_m)
            if slot is not None:
                idx_to_cmd[idx] = f"move {slot}"
        else:
            # Switch action (indices 4..8)
            if not switch_slots_meta:
                continue
            sp_id = _to_id(str(getattr(target, "species", "") or getattr(target, "name", "")))
            sp_mm = pokemon_name(str(getattr(target, "species", "") or getattr(target, "name", "")))
            matched_slot: Optional[int] = None
            for slot, det_id, det_mm, ident_id in switch_slots_meta:
                if sp_id == det_id or sp_mm == det_mm or sp_id == ident_id:
                    matched_slot = slot
                    break
            if matched_slot is None:
                for slot, det_id, det_mm, _ in switch_slots_meta:
                    if det_id.startswith(sp_id) or sp_id.startswith(det_id):
                        matched_slot = slot
                        break
            if matched_slot is not None:
                idx_to_cmd[idx] = f"switch {matched_slot}"

    return idx_to_cmd


def format_info_line(
    win_prob: float,
    cmd_probs: Dict[str, float],
) -> Optional[str]:
    """Format a validated `/info <JSON>` line matching `server/protocol.ts`."""
    payload: Dict[str, Any] = {}
    if math.isfinite(win_prob):
        payload["win"] = round(min(1.0, max(0.0, float(win_prob))), 4)

    positive = [
        (cmd, float(p))
        for cmd, p in cmd_probs.items()
        if math.isfinite(p) and float(p) > 0.0
    ]
    if positive:
        positive.sort(key=lambda kv: kv[1], reverse=True)
        top = positive[:MAX_INFO_POLICY_ENTRIES]
        total = sum(p for _, p in top)
        if total > 0.0:
            policy = {
                cmd: round(min(1.0, max(0.0, p / total)), 4)
                for cmd, p in top
            }
            policy = {cmd: p for cmd, p in policy.items() if p > 0.0}
            if policy:
                payload["policy"] = policy

    if not payload:
        return None
    raw = json.dumps(payload, separators=(",", ":"))
    if len(raw) > MAX_INFO_LINE_CHARS and "policy" in payload:
        payload.pop("policy", None)
        raw = json.dumps(payload, separators=(",", ":"))
    if len(raw) > MAX_INFO_LINE_CHARS:
        return None
    return f"/info {raw}"


class TaurosOnnxEngine:
    """Stateful ONNX Runtime inference engine for Metamon TaurosV0."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = cfg.onnx_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress info/warning chatter on stderr

        if not cfg.model_path.is_file():
            raise FileNotFoundError(f"TaurosV0 ONNX model not found: {cfg.model_path}")

        self.session = ort.InferenceSession(
            str(cfg.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Prewarm static Gen 1 Dex and tokenizer before /ready
        Dex.from_gen(1)
        self.tokenizer = get_tokenizer("DefaultObservationSpace-v1")
        self.obs_space = TokenizedObservationSpace(
            base_obs_space=GroupedObservationSpace(),
            tokenizer=self.tokenizer,
        )
        self.action_space = DefaultActionSpace()
        self.reward_fn = AggressiveShapedReward()

        self.step_count = 0
        self.prev_state: Optional[UniversalState] = None
        self.prev_action_idx: Optional[int] = None
        self.k_cache = np.zeros((N_LAYERS, 1, WINDOW_SIZE, N_HEADS, HEAD_DIM), dtype=np.float32)
        self.v_cache = np.zeros((N_LAYERS, 1, WINDOW_SIZE, N_HEADS, HEAD_DIM), dtype=np.float32)

        self._warmup()

    def _warmup(self) -> None:
        dummy_feed = {
            "pokemon_text": np.zeros((1, 7, 12), dtype=np.int64),
            "pokemon_nums": np.zeros((1, 7, 31), dtype=np.float32),
            "global_text": np.zeros((1, 20), dtype=np.int64),
            "global_nums": np.zeros((1, 4), dtype=np.float32),
            "rl2": np.zeros((1, 14), dtype=np.float32),
            "illegal_actions": np.array(
                [[False, False, True, True, False, True, True, True, True, True, True, True, True]],
                dtype=bool,
            ),
            "time_idx": np.array([0], dtype=np.int64),
            "cache_len": np.array([0], dtype=np.int64),
            "k_cache": self.k_cache,
            "v_cache": self.v_cache,
        }
        for _ in range(2):
            self.session.run(None, dummy_feed)

    def reset_episode(self) -> None:
        self.obs_space.reset()
        self.step_count = 0
        self.prev_state = None
        self.prev_action_idx = None
        self.k_cache.fill(0.0)
        self.v_cache.fill(0.0)

    def infer(
        self,
        state: UniversalState,
        legal_indices: Set[int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run one forward step, updating KV cache and episode state.

        Returns `(probs[13], q_values[13], v_value)`.
        """
        obs = self.obs_space.state_to_obs(state)

        pokemon_text = np.stack(
            [np.asarray(obs[k], dtype=np.int64) for k in _POKEMON_TEXT_KEYS],
            axis=0,
        )[None, :, :]  # (1, 7, 12)
        pokemon_nums = np.stack(
            [np.asarray(obs[k], dtype=np.float32) for k in _POKEMON_NUM_KEYS],
            axis=0,
        )[None, :, :]  # (1, 7, 31)
        global_text = np.asarray(obs["text_misc_tokens"], dtype=np.int64)[None, :]  # (1, 20)
        global_nums = np.asarray(obs["numbers_misc"], dtype=np.float32)[None, :]  # (1, 4)

        illegal = np.ones((1, NUM_ACTIONS), dtype=bool)
        for idx in legal_indices:
            if 0 <= idx < NUM_ACTIONS:
                illegal[0, idx] = False

        if self.step_count == 0 or self.prev_state is None:
            reward_val = 0.0
            prev_act = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        else:
            reward_val = float(self.reward_fn(self.prev_state, state))
            prev_act = np.zeros((NUM_ACTIONS,), dtype=np.float32)
            if self.prev_action_idx is not None and 0 <= self.prev_action_idx < NUM_ACTIONS:
                prev_act[self.prev_action_idx] = 1.0

        rl2 = np.concatenate(([reward_val], prev_act), axis=-1).astype(np.float32)[None, :]
        time_idx = np.array([self.step_count], dtype=np.int64)
        cache_len = np.array([min(self.step_count, WINDOW_SIZE)], dtype=np.int64)

        probs_out, q_out, v_out, new_k, new_v = self.session.run(
            None,
            {
                "pokemon_text": pokemon_text,
                "pokemon_nums": pokemon_nums,
                "global_text": global_text,
                "global_nums": global_nums,
                "rl2": rl2,
                "illegal_actions": illegal,
                "time_idx": time_idx,
                "cache_len": cache_len,
                "k_cache": self.k_cache,
                "v_cache": self.v_cache,
            },
        )
        self.k_cache = new_k
        self.v_cache = new_v
        self.prev_state = state
        self.step_count += 1

        return probs_out[0], q_out[0], float(v_out[0])

    def select_action_index(
        self,
        probs: np.ndarray,
        idx_to_cmd: Dict[int, str],
    ) -> int:
        valid_idxs = sorted(idx_to_cmd.keys())
        p = np.array([max(0.0, float(probs[i])) for i in valid_idxs], dtype=np.float64)
        if self.cfg.temperature != 1.0 and p.sum() > 0.0:
            # Apply temperature in log-probability space over legal actions
            log_p = np.log(np.maximum(p, 1e-30)) / self.cfg.temperature
            log_p -= np.max(log_p)
            p = np.exp(log_p)
        total = float(p.sum())
        if total <= 0.0 or not math.isfinite(total):
            p = np.full((len(valid_idxs),), 1.0 / len(valid_idxs), dtype=np.float64)
        else:
            p /= total

        if self.cfg.sample_actions:
            choice_pos = int(self.rng.choice(len(valid_idxs), p=p))
        else:
            choice_pos = int(np.argmax(p))
        chosen_idx = valid_idxs[choice_pos]
        self.prev_action_idx = chosen_idx
        return chosen_idx


class PkmnAiMetamonSession:
    """Line-oriented pkmn.ai stdio protocol driver for Metamon TaurosV0 (gen1ou)."""

    def __init__(
        self,
        engine: TaurosOnnxEngine,
        info_sink: Optional[Callable[[str], None]] = None,
    ):
        self.engine = engine
        self.cfg = engine.cfg
        self._info_sink = info_sink
        self.battle_index = 0
        self.battle: Optional[MetamonBackendBattle] = None
        self.saw_turn_1 = False
        self.pending_request: Optional[Dict[str, Any]] = None
        self.last_request: Optional[Dict[str, Any]] = None
        self.retrying_error = False
        self.tried_cmds: Set[str] = set()
        self.last_cmd_probs: Dict[str, float] = {}
        self._reset_battle()

    def _reset_battle(self) -> None:
        self.battle_index += 1
        self.battle = MetamonBackendBattle(
            battle_tag=f"battle-gen1ou-{self.battle_index}",
            username="Metamon",
            logger=logger,
            save_replays=False,
            gen=1,
        )
        self.engine.reset_episode()
        self.saw_turn_1 = False
        self.pending_request = None
        self.last_request = None
        self.retrying_error = False
        self.tried_cmds = set()
        self.last_cmd_probs = {}

    def _emit_info(self, line: Optional[str]) -> None:
        if line and self.cfg.emit_info and self._info_sink is not None:
            self._info_sink(line)

    def _can_run_model(self) -> bool:
        if self.battle is None or self.battle.player_role not in ("p1", "p2"):
            return False
        try:
            p1 = self.battle.player_role == "p1"
            turn = self.battle._current_turn
            my_active = turn.get_active_pokemon(p1)[0]
            opp_active = turn.get_active_pokemon(not p1)[0]
            return my_active is not None and opp_active is not None
        except Exception:
            return False

    def _choose_for_request(self, req: Dict[str, Any]) -> str:
        assert self.battle is not None
        self.last_request = req

        if not self.retrying_error:
            self.tried_cmds = set()
            self.last_cmd_probs = {}
        else:
            # On [Invalid choice] / [Unavailable choice] retry, pick the highest-prob
            # untried legal command without double-advancing the transformer KV cache.
            self.retrying_error = False
            self.battle.parse_request(req)
            ranked = sorted(
                self.last_cmd_probs.items(), key=lambda kv: kv[1], reverse=True
            )
            for cmd, _ in ranked:
                if cmd not in self.tried_cmds:
                    self.tried_cmds.add(cmd)
                    return cmd
            cmd = fallback_command_from_request(req, self.tried_cmds)
            self.tried_cmds.add(cmd)
            return cmd

        self.battle.parse_request(req)

        if req.get("teamPreview"):
            cmd = fallback_command_from_request(req, self.tried_cmds)
            self.tried_cmds.add(cmd)
            return cmd

        if not self._can_run_model():
            logger.warning("Battle active slots not initialized; using legal fallback")
            cmd = fallback_command_from_request(req, self.tried_cmds)
            self.tried_cmds.add(cmd)
            return cmd

        try:
            state = UniversalState.from_Battle(self.battle)
            idx_to_cmd = map_legal_actions_to_commands(self.battle, state, req)
            if not idx_to_cmd:
                logger.warning("No mapped legal actions on turn %d; using fallback", self.battle.turn)
                cmd = fallback_command_from_request(req, self.tried_cmds)
                self.tried_cmds.add(cmd)
                return cmd

            probs, _q_vals, v_val = self.engine.infer(state, set(idx_to_cmd.keys()))

            cmd_probs: Dict[str, float] = {}
            for idx, cmd in idx_to_cmd.items():
                p = float(probs[idx])
                if math.isfinite(p) and p >= 0.0:
                    cmd_probs[cmd] = cmd_probs.get(cmd, 0.0) + p
            self.last_cmd_probs = cmd_probs

            chosen_idx = self.engine.select_action_index(probs, idx_to_cmd)
            chosen_cmd = idx_to_cmd[chosen_idx]
            self.tried_cmds.add(chosen_cmd)

            win_prob = value_to_win_prob(v_val, self.cfg)
            self._emit_info(format_info_line(win_prob, cmd_probs))
            return chosen_cmd
        except Exception:
            logger.exception("TaurosV0 inference failed on turn %d; using legal fallback", self.battle.turn)
            cmd = fallback_command_from_request(req, self.tried_cmds)
            self.tried_cmds.add(cmd)
            return cmd

    def handle_line(self, raw_line: str) -> Optional[str]:
        line = raw_line.rstrip("\r\n")
        if not line or not line.startswith("|"):
            return None

        # Intercept arena-only team packet before Metamon's SimProtocol
        if line.startswith("|utm|"):
            return None

        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""

        if tag == "init" and len(parts) > 2 and parts[2] == "battle":
            if self.saw_turn_1 or (self.battle is not None and self.battle.finished):
                self._reset_battle()
            return None

        if tag == "tier":
            if self.battle is not None and self.battle.finished:
                self._reset_battle()

        if tag == "request":
            payload = line[len("|request|") :].strip()
            if not payload:
                return None
            try:
                req = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Malformed |request| JSON ignored")
                return None
            if req.get("wait"):
                self.pending_request = None
                return None
            if not self.saw_turn_1:
                # Turn 1 |request| arrives before |switch|p1a / |switch|p2a / |turn|1.
                # Defer parsing and choosing until |turn|1 so both active slots exist.
                self.pending_request = req
                return None
            return self._choose_for_request(req)

        if tag == "error":
            err_msg = parts[2] if len(parts) > 2 else ""
            if "[Invalid choice]" in err_msg or "[Unavailable choice]" in err_msg:
                logger.warning("Showdown rejected choice: %s", err_msg)
                self.retrying_error = True
            return None

        # Feed standard Showdown protocol line into MetamonBackendBattle
        assert self.battle is not None
        try:
            self.battle.parse_message(parts)
        except Exception as exc:
            logger.debug("Ignored MetamonBackendBattle parse error on %r: %s", line[:120], exc)

        if tag == "turn":
            self.saw_turn_1 = True
            if self.pending_request is not None:
                req = self.pending_request
                self.pending_request = None
                return self._choose_for_request(req)

        return None


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="metamon.env.metamon_battle")
    cfg = AgentConfig.from_env()
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, cfg.log_level, logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = TaurosOnnxEngine(cfg)

    def write_line(text: str) -> None:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    session = PkmnAiMetamonSession(engine=engine, info_sink=write_line)

    # Signal readiness only after ONNX session load, Dex/tokenizer init, and warmup passes.
    write_line("/ready")

    for raw_line in sys.stdin:
        cmd = session.handle_line(raw_line)
        if cmd is not None:
            write_line(cmd)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
