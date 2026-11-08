"""
Connect Metamon agents (e.g. TaurosV0 for Gen 1 OU + Kakuna for Gen 3 / Gen 9 OU)
to https://pkmn.bot.

Supports:
  1. Unrated practice challenges against a specific bot (e.g. `--challenge Cocoa`)
     without entering the rated matchmaking pool.
  2. Ranked matchmaking (`--ranked`) across multiple ladders (`gen1ou`, `gen3ou`,
     `gen9ou`) on a single WebSocket connection with automatic reconnection,
     per-battle team rotation from `/Users/kjs/Code/src/github.com/pkmn/teams`,
     and automatic blocked-format handling/signaling to `pkmn.bot`.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
from dataclasses import dataclass, field
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import gin
import numpy as np
import orjson
import torch
import websockets.exceptions

import amago
import amago.nets.transformer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.concurrency import POKE_LOOP
from poke_env.environment import AbstractBattle
from poke_env.exceptions import ShowdownException
from poke_env.player import BattleOrder
from poke_env.ps_client import PSClient
from poke_env.teambuilder import Teambuilder

from metamon.backend.showdown_dex import Dex
from metamon.backend.team_preview.preview import TeamPreviewModel
from metamon.config import FORMAT_ALIASES, format_for_agent
from metamon.env import PokeAgentTeamSet, TeamSet, get_metamon_teams
from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.env.metamon_player import MetamonPlayer
from metamon.interface import (
    ActionSpace,
    ObservationSpace,
    RewardFunction,
    TokenizedObservationSpace,
    UniversalAction,
    UniversalState,
)
from metamon.rl.pretrained import PretrainedModel, get_pretrained_model

# Map random battle format to gen9ou for agent observation/dex lookup
FORMAT_ALIASES.setdefault("gen9randombattle", "gen9ou")

PKMN_BOT_SERVER = ServerConfiguration(
    "wss://pkmn.bot/showdown/websocket",
    "https://pkmn.bot/action.php?",
)

DEFAULT_PKMN_TEAMS_DIR = "/Users/kjs/Code/src/github.com/pkmn/teams"
DEFAULT_VGC_BOT_DIR = "/Users/kjs/Code/src/github.com/scheibo/vgc.bot"

# Single-worker thread pool for PyTorch/MPS forward passes so inference never blocks
# POKE_LOOP WebSocket I/O and concurrent battles never contend on Metal command queues.
_INFERENCE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="metamon-mps-infer"
)
_MPS_LOCK = threading.Lock()

# Map display format names in pkmn.bot popups back to canonical format IDs
_FORMAT_DISPLAY_TO_ID = {
    "[gen 1] ou": "gen1ou",
    "gen 1 ou": "gen1ou",
    "gen1ou": "gen1ou",
    "[gen 2] ou": "gen2ou",
    "gen 2 ou": "gen2ou",
    "gen2ou": "gen2ou",
    "[gen 3] ou": "gen3ou",
    "gen 3 ou": "gen3ou",
    "gen3ou": "gen3ou",
    "[gen 4] ou": "gen4ou",
    "gen 4 ou": "gen4ou",
    "gen4ou": "gen4ou",
    "[gen 9] ou": "gen9ou",
    "gen 9 ou": "gen9ou",
    "gen9ou": "gen9ou",
    "[gen 9] random battle": "gen9randombattle",
    "gen 9 random battle": "gen9randombattle",
    "gen9randombattle": "gen9randombattle",
}


class _DropRawFramesFilter(logging.Filter):
    """Suppress poke-env's huge `<<<` and `>>>` raw WebSocket frame dumps on stdout/file logs
    so multi-day runs do not generate multi-GB logs or trigger swap thrashing."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        head = msg[:30]
        if "<<<" in head or ">>>" in head:
            # Keep redacted login confirmation only
            return "|/trn" in head
        return True


def _configure_bounded_logging(repo_root: str) -> None:
    log_dir = os.path.join(repo_root, ".cache")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pkmn_bot.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    frame_filter = _DropRawFramesFilter()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(frame_filter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(frame_filter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [stream_handler, file_handler]


def clean_and_pack_showdown_team(raw: str) -> str:
    """Clean trailing spaces from Showdown export format and pack without poke-env's
    buggy HP_TO_IVS table (allowing Showdown's TeamValidator to fill generation-correct
    Hidden Power IVs and Gen 1 EVs)."""
    cleaned = "\n".join(line.strip() for line in raw.splitlines())
    mons = Teambuilder.parse_showdown_team(cleaned)
    for mon in mons:
        if all(iv == 31 for iv in mon.ivs):
            for move in mon.moves:
                m = re.match(r"Hidden Power \[([A-Za-z]+)\]", move.strip(), re.I)
                if m:
                    mon.hiddenpowertype = m.group(1).capitalize()
    return "]".join(
        "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s%s"
        % (
            mon.nickname or "",
            re.sub(r"[^a-z0-9]+", "", mon.species.lower()) if mon.species else "",
            re.sub(r"[^a-z0-9]+", "", mon.item.lower()) if mon.item else "",
            re.sub(r"[^a-z0-9]+", "", mon.ability.lower()) if mon.ability else "",
            mon.formatted_moves or "",
            mon.nature or "",
            mon.formatted_evs or "",
            mon.gender or "",
            mon.formatted_ivs or "",
            "S" if mon.shiny else "",
            mon.level or "",
            mon.happiness or "",
            mon.formatted_endstring,
        )
        for mon in mons
    )


class PkmnTeamsSet(Teambuilder):
    """In-memory, pre-validated team set loaded from `/Users/kjs/Code/src/github.com/pkmn/teams`.

    - Filters to teams marked valid (`✓`) in `<teams_dir>/<format>/README.md`.
    - Validates and canonicalizes packed teams against `vgc.bot`'s `validateTeam` engine
      (cached in `.cache/pkmn_teams_validated.json`), falling back to `clean_and_pack_showdown_team`.
    - Keeps all packed strings in RAM so `yield_team()` performs zero disk I/O during live battles.
    - Supports runtime blacklisting (`mark_team_rejected`) if the server ever rejects a team.
    """

    def __init__(
        self,
        teams_dir: str,
        battle_format: str,
        cache_dir: Optional[str] = None,
        vgc_bot_dir: str = DEFAULT_VGC_BOT_DIR,
    ):
        super().__init__()
        self.teams_dir = teams_dir
        self.battle_format = format_for_agent(battle_format)
        self.format_dir = os.path.join(teams_dir, self.battle_format)
        if not os.path.isdir(self.format_dir):
            raise ValueError(
                f"Format directory not found at {self.format_dir}"
            )

        readme_path = os.path.join(self.format_dir, "README.md")
        valid_ids: List[str] = []
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.search(
                        r"\]\(([0-9a-f]{12})\)\s*\|\s*[^|]+\|\s*✓\s*\|", line
                    )
                    if m:
                        tid = m.group(1)
                        if os.path.isfile(os.path.join(self.format_dir, tid)):
                            valid_ids.append(tid)

        if not valid_ids:
            for entry in sorted(os.listdir(self.format_dir)):
                if re.fullmatch(r"[0-9a-f]{12}", entry) and os.path.isfile(
                    os.path.join(self.format_dir, entry)
                ):
                    valid_ids.append(entry)

        self._rejected_packed: Set[str] = set()
        self._most_recent_team_file: Optional[str] = None
        self._most_recent_packed: Optional[str] = None
        self.team_files: List[str] = [
            os.path.join(self.format_dir, tid) for tid in valid_ids
        ]
        self._packed_by_file: Dict[str, str] = self._load_and_validate_teams(
            valid_ids, cache_dir=cache_dir, vgc_bot_dir=vgc_bot_dir
        )
        self.team_files = [
            f for f in self.team_files if f in self._packed_by_file
        ]

    def _load_and_validate_teams(
        self,
        valid_ids: List[str],
        cache_dir: Optional[str],
        vgc_bot_dir: str,
    ) -> Dict[str, str]:
        cache_path = (
            os.path.join(
                cache_dir, f"pkmn_teams_{self.battle_format}_validated.json"
            )
            if cache_dir
            else None
        )
        readme_path = os.path.join(self.format_dir, "README.md")
        readme_mtime = (
            os.path.getmtime(readme_path) if os.path.exists(readme_path) else 0.0
        )

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if (
                    cached.get("readme_mtime") == readme_mtime
                    and cached.get("count") == len(valid_ids)
                    and isinstance(cached.get("packed_by_id"), dict)
                ):
                    return {
                        os.path.join(self.format_dir, tid): packed
                        for tid, packed in cached["packed_by_id"].items()
                    }
            except Exception:
                pass

        raw_by_id: Dict[str, str] = {}
        for tid in valid_ids:
            fpath = os.path.join(self.format_dir, tid)
            with open(fpath, "r", encoding="utf-8") as f:
                raw_by_id[tid] = f.read()

        packed_by_id: Dict[str, str] = {}
        bun_bin = shutil.which("bun")
        engine_ts = os.path.join(vgc_bot_dir, "worker", "engine", "index.ts")
        if bun_bin and os.path.exists(engine_ts):
            try:
                proc = subprocess.run(
                    [
                        bun_bin,
                        "-e",
                        (
                            'import { validateTeam } from "./worker/engine/index.ts";\n'
                            "const { fmt, teams } = JSON.parse(await Bun.stdin.text());\n"
                            "const out = {};\n"
                            "for (const [id, raw] of Object.entries(teams)) {\n"
                            "  const v = validateTeam(fmt, raw, { keepNicknames: false });\n"
                            "  if (v.ok) out[id] = v.packedTeam;\n"
                            "}\n"
                            "console.log(JSON.stringify(out));\n"
                        ),
                    ],
                    input=json.dumps(
                        {"fmt": self.battle_format, "teams": raw_by_id}
                    ),
                    text=True,
                    cwd=vgc_bot_dir,
                    capture_output=True,
                    timeout=30,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    packed_by_id = json.loads(proc.stdout.strip())
            except Exception:
                packed_by_id = {}

        if not packed_by_id:
            for tid, raw in raw_by_id.items():
                packed = clean_and_pack_showdown_team(raw)
                if not self.block_team(packed):
                    packed_by_id[tid] = packed
        else:
            packed_by_id = {
                tid: packed
                for tid, packed in packed_by_id.items()
                if not self.block_team(packed)
            }

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "readme_mtime": readme_mtime,
                            "count": len(valid_ids),
                            "packed_by_id": packed_by_id,
                        },
                        f,
                    )
            except Exception:
                pass

        return {
            os.path.join(self.format_dir, tid): packed
            for tid, packed in packed_by_id.items()
        }

    def block_team(self, packed_team: str) -> bool:
        if packed_team in self._rejected_packed:
            return True
        if self.battle_format == "gen9ou":
            norm = re.sub(r"[^a-z0-9]+", "", packed_team.lower())
            for banned in ("roaringmoon", "zoroark", "zorua", "revivalblessing"):
                if banned in norm:
                    return True
        return False

    def mark_team_rejected(self, packed_team: Optional[str] = None) -> None:
        target = packed_team or self._most_recent_packed
        if target:
            self._rejected_packed.add(target)
            self.team_files = [
                f
                for f in self.team_files
                if self._packed_by_file.get(f) not in self._rejected_packed
            ]

    def has_valid_teams(self) -> bool:
        return len(self.team_files) > 0

    @property
    def most_recent_team_file(self) -> Optional[str]:
        return self._most_recent_team_file

    def yield_team(self) -> str:
        if not self.team_files:
            raise RuntimeError(
                f"No valid teams remaining for format {self.battle_format}"
            )
        for _ in range(min(100, len(self.team_files) * 2)):
            file = random.choice(self.team_files)
            packed = self._packed_by_file.get(file)
            if packed and not self.block_team(packed):
                self._most_recent_team_file = file
                self._most_recent_packed = packed
                return packed
        raise RuntimeError(
            f"Could not find valid team for {self.battle_format}"
        )


def _patch_flash_attention_fallback() -> None:
    """Provide an eager sliding-window attention fallback when CUDA flash_attn is absent,
    disable TorchInductor Metal kernel overflow on MPS, and fix Python 3.12 float save_every."""
    import torch._dynamo
    from amago.envs.amago_env import SequenceWrapper

    torch._dynamo.config.disable = True
    SequenceWrapper.random_traj_length = (
        lambda self: random.randint(int(self.save_every[0]), int(self.save_every[1]))
        if self.save_every
        else None
    )
    if amago.nets.transformer.flash_attn is not None:
        return

    def _init_fallback(
        self,
        causal: bool = True,
        dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
    ):
        amago.nets.transformer.SelfAttention.__init__(
            self, causal=causal, dropout=dropout
        )
        self.window_size = window_size

    @torch.compiler.disable
    def _forward_fallback(
        self, qkv, key_cache=None, val_cache=None, cache_seqlens=None
    ):
        orig_dtype = qkv.dtype
        queries, keys, values = torch.unbind(qkv.float(), dim=2)
        B, L, H, E = queries.shape
        scale = 1.0 / math.sqrt(E)

        if key_cache is None or val_cache is None or cache_seqlens is None:
            scores = scale * torch.einsum("blhe,bshe->bhls", queries, keys)
            if self.causal:
                q_idx = torch.arange(L, device=qkv.device)[:, None]
                k_idx = torch.arange(L, device=qkv.device)[None, :]
                invalid = k_idx > q_idx
                if self.window_size[0] >= 0:
                    invalid = invalid | ((q_idx - k_idx) > self.window_size[0])
                scores = scores.masked_fill(invalid[None, None, :, :], -torch.inf)
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhls,bshd->blhd", attn, values)
            return out.to(orig_dtype)
        else:
            assert not self.training and L == 1
            cache_idxs = torch.arange(B, device=key_cache.device)
            key_cache[cache_idxs, cache_seqlens] = keys[:, 0].to(key_cache.dtype)
            val_cache[cache_idxs, cache_seqlens] = values[:, 0].to(val_cache.dtype)
            end = cache_seqlens + 1
            max_len = int(end.max().item())
            k_cache = torch.nan_to_num(key_cache[:, :max_len].float())
            v_cache = torch.nan_to_num(val_cache[:, :max_len].float())
            scores = scale * torch.einsum("blhe,bshe->bhls", queries, k_cache)
            s_idxs = torch.arange(max_len, device=cache_seqlens.device)[None, :]
            invalid = s_idxs >= end[:, None]
            if self.window_size[0] >= 0:
                start = torch.clamp(cache_seqlens - self.window_size[0], min=0)
                invalid = invalid | (s_idxs < start[:, None])
            scores = scores.masked_fill(invalid[:, None, None, :], -torch.inf)
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhls,bshd->blhd", attn, v_cache)
            return out.to(orig_dtype)

    amago.nets.transformer.FlashAttention.__init__ = _init_fallback
    amago.nets.transformer.FlashAttention.forward = _forward_fallback


@dataclass
class _BattleEpisodeState:
    obs_space: ObservationSpace
    action_space: ActionSpace
    reward_fn: RewardFunction
    step_count: int
    prev_state: Optional[UniversalState]
    prev_action_idx: Optional[int]
    hidden_state: Any
    last_updated: float = field(default_factory=time.monotonic)


class ModelPolicyRunner:
    """Wraps a PretrainedModel for direct per-battle turn inference without OpenAIGymEnv."""

    def __init__(
        self,
        model_name: str,
        checkpoint: Optional[int] = None,
        action_temperature: float = 1.0,
        sample_actions: bool = True,
    ):
        _patch_flash_attention_fallback()
        gin.clear_config()
        self.model_name = model_name
        self.pretrained_model: PretrainedModel = get_pretrained_model(model_name)
        self.checkpoint = (
            checkpoint
            if checkpoint is not None
            else self.pretrained_model.default_checkpoint
        )
        self.action_temperature = action_temperature
        self.sample_actions = sample_actions
        self.experiment = self.pretrained_model.initialize_agent(
            checkpoint=self.checkpoint,
            log=False,
            action_temperature=action_temperature,
        )
        self.policy = getattr(self.experiment, "policy_aclr", self.experiment.policy)
        self.policy.eval()
        self.device = self.experiment.DEVICE
        self.action_size = self.pretrained_model.action_space.gym_space.n
        self._battle_states: Dict[str, _BattleEpisodeState] = {}
        self._warmup()

    def _warmup(self) -> None:
        """Run a single dummy forward pass so MPS/CPU kernels are warmed up before live battles."""
        hs = self.policy.traj_encoder.init_hidden_state(1, self.device)
        obs_space = self.pretrained_model.observation_space.gym_space
        obs = {
            k: torch.zeros(
                (1, 1, *v.shape),
                dtype=torch.from_numpy(np.zeros(v.shape, dtype=v.dtype)).dtype,
                device=self.device,
            )
            for k, v in obs_space.items()
        }
        obs["illegal_actions"] = torch.zeros(
            (1, 1, self.action_size), dtype=torch.bool, device=self.device
        )
        rl2 = torch.zeros(
            (1, 1, self.action_size + 1), dtype=torch.float32, device=self.device
        )
        t_idx = torch.tensor([[[0]]], dtype=torch.int64, device=self.device)
        with torch.no_grad():
            self.policy.get_actions(
                obs=obs, rl2s=rl2, time_idxs=t_idx, hidden_state=hs, sample=True
            )

    def _clone_obs_space(self) -> ObservationSpace:
        """Clone only the lightweight stateful base observation space without deepcopying
        PokemonTokenizer's entire vocabulary dict on every battle."""
        src = self.pretrained_model.observation_space
        if isinstance(src, TokenizedObservationSpace):
            obs_space = TokenizedObservationSpace(
                base_obs_space=copy.deepcopy(src.base_obs_space),
                tokenizer=src.tokenizer,
            )
        else:
            obs_space = copy.deepcopy(src)
        obs_space.reset()
        return obs_space

    def _get_or_create_episode(self, battle_tag: str) -> _BattleEpisodeState:
        if battle_tag not in self._battle_states:
            obs_space = self._clone_obs_space()
            action_space = self.pretrained_model.action_space
            reward_fn = self.pretrained_model.reward_function
            hidden_state = self.policy.traj_encoder.init_hidden_state(
                batch_size=1, device=self.device
            )
            self._battle_states[battle_tag] = _BattleEpisodeState(
                obs_space=obs_space,
                action_space=action_space,
                reward_fn=reward_fn,
                step_count=0,
                prev_state=None,
                prev_action_idx=None,
                hidden_state=hidden_state,
                last_updated=time.monotonic(),
            )
        ep = self._battle_states[battle_tag]
        ep.last_updated = time.monotonic()
        return ep

    def clear_battle(self, battle_tag: str) -> None:
        with _MPS_LOCK:
            self._battle_states.pop(battle_tag, None)

    def prune_stale_battles(
        self, active_tags: Set[str], max_age_s: float = 600.0
    ) -> int:
        with _MPS_LOCK:
            now = time.monotonic()
            stale = [
                tag
                for tag, ep in list(self._battle_states.items())
                if tag not in active_tags and (now - ep.last_updated) > max_age_s
            ]
            for tag in stale:
                self._battle_states.pop(tag, None)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return len(stale)

    def clear_all_battles(self) -> None:
        with _MPS_LOCK:
            self._battle_states.clear()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def infer_universal_action(
        self,
        battle_tag: str,
        state: UniversalState,
        legal_universal: List[UniversalAction],
    ) -> UniversalAction:
        """Thread-safe forward pass that depends only on `UniversalState` and `legal_universal`
        so it can run in `_INFERENCE_EXECUTOR` off `POKE_LOOP`."""
        with _MPS_LOCK:
            ep = self._get_or_create_episode(battle_tag)
            obs = ep.obs_space.state_to_obs(state)

            legal_actions = [
                ep.action_space.action_to_agent_output(state=state, action=act)
                for act in legal_universal
            ]
            illegal_mask = np.ones((self.action_size,), dtype=bool)
            for act_idx in legal_actions:
                if 0 <= act_idx < self.action_size:
                    illegal_mask[act_idx] = False
            obs["illegal_actions"] = illegal_mask

            if ep.step_count == 0 or ep.prev_state is None:
                reward_val = 0.0
                prev_action_onehot = np.zeros((self.action_size,), dtype=np.float32)
            else:
                reward_val = float(ep.reward_fn(ep.prev_state, state))
                prev_action_onehot = np.zeros((self.action_size,), dtype=np.float32)
                if (
                    ep.prev_action_idx is not None
                    and 0 <= ep.prev_action_idx < self.action_size
                ):
                    prev_action_onehot[ep.prev_action_idx] = 1.0

            rl2 = np.concatenate(([reward_val], prev_action_onehot), axis=-1).astype(
                np.float32
            )
            obs_tensors = {
                k: torch.from_numpy(np.asarray(v))
                .to(self.device)
                .view(1, 1, *np.asarray(v).shape)
                for k, v in obs.items()
            }
            rl2_tensor = torch.from_numpy(rl2).to(self.device).view(1, 1, -1)
            time_idx_tensor = torch.tensor(
                [[[ep.step_count]]], dtype=torch.int64, device=self.device
            )

            with torch.no_grad():
                actions, ep.hidden_state = self.policy.get_actions(
                    obs=obs_tensors,
                    rl2s=rl2_tensor,
                    time_idxs=time_idx_tensor,
                    hidden_state=ep.hidden_state,
                    sample=self.sample_actions,
                )
            action_idx = int(actions.squeeze().item())
            ep.prev_state = state
            ep.prev_action_idx = action_idx
            ep.step_count += 1

            return ep.action_space.agent_output_to_action(
                state=state, agent_output=action_idx
            )

    def choose_action(
        self, battle_tag: str, state: UniversalState, battle: AbstractBattle
    ) -> Optional[BattleOrder]:
        legal_universal = UniversalAction.definitely_valid_actions(
            state=state, battle=battle
        )
        universal_action = self.infer_universal_action(
            battle_tag=battle_tag, state=state, legal_universal=legal_universal
        )
        return universal_action.to_BattleOrder(battle)


class PkmnBotPSClient(PSClient):
    """PSClient tailored for pkmn.bot protocol (direct /trn token login, no /avatar)."""

    async def log_in(self, split_message: List[str]):
        if self.account_configuration.password:
            self.logger.info(">>> |/trn %s,0,[REDACTED]", self.username)
            await self.websocket.send(
                f"|/trn {self.username},0,{self.account_configuration.password}"
            )
        else:
            await self.send_message(f"/trn {self.username},0,")


def _extract_format_from_popup(
    popup_text: str, candidate_formats: List[str]
) -> Optional[str]:
    lower = popup_text.lower()
    for disp, fmt_id in _FORMAT_DISPLAY_TO_ID.items():
        if disp in lower:
            return fmt_id
    for fmt in candidate_formats:
        if fmt.lower() in lower:
            return fmt.lower()
    return None


class PkmnBotPlayer(MetamonPlayer):
    """Multi-format Metamon player for pkmn.bot routing by battle format."""

    def __init__(
        self,
        model_runners: Dict[str, ModelPolicyRunner],
        default_runner: ModelPolicyRunner,
        team_sets: Dict[str, TeamSet],
        formats: Optional[List[str]] = None,
        blocked_formats: Optional[Set[str]] = None,
        rotate_teams_on_battle_end: bool = True,
        team_preview_model: Optional[TeamPreviewModel] = None,
        *args,
        **kwargs,
    ):
        self.model_runners = model_runners
        self.default_runner = default_runner
        self.team_sets = team_sets
        self.rotate_teams_on_battle_end = rotate_teams_on_battle_end
        self.blocked_formats: Set[str] = (
            blocked_formats if blocked_formats is not None else set()
        )
        initial_formats = [
            f.replace("-", "").lower()
            for f in (formats or list(team_sets.keys()))
        ]
        self.active_formats: List[str] = [
            f for f in initial_formats if f not in self.blocked_formats
        ]
        self.last_uploaded_team_by_format: Dict[str, str] = {}
        self.pending_upload_format: Optional[str] = None
        self.consecutive_rejections_by_format: Dict[str, int] = {}
        self._team_upload_lock = asyncio.Lock()

        self.completed_battles: List[Dict[str, Any]] = []
        self.total_completed_count: int = 0
        self.active_battles: Set[str] = set()
        self.server_games: Set[str] = set()
        self._battle_locks: Dict[str, asyncio.Lock] = {}
        self._retried_rqids: Dict[str, int] = {}
        self.random_format_games_completed: Dict[str, int] = {}
        self.draining: bool = False
        self.popups: List[str] = []
        self.battle_finished_event = asyncio.Event()
        super().__init__(
            *args,
            team_preview_model=team_preview_model,
            start_timer_on_battle_start=False,
            **kwargs,
        )
        # Prevent duplicate StreamHandlers and suppress raw WebSocket frame dumps
        frame_filter = _DropRawFramesFilter()
        for lg in (self.logger, getattr(self.ps_client, "logger", None)):
            if lg is not None:
                lg.propagate = True
                lg.handlers = []
                lg.addFilter(frame_filter)
        # Replace PSClient log_in with direct pkmn.bot login (avoids /avatar popup)
        self.ps_client.log_in = PkmnBotPSClient.log_in.__get__(
            self.ps_client, PSClient
        )
        orig_handle_message = self.ps_client._handle_message

        async def _wrapped_handle_message(message: str):
            if message.startswith("|popup|"):
                popup_text = message[len("|popup|") :].strip()
                self.popups.append(popup_text)
                if len(self.popups) > 100:
                    self.popups = self.popups[-100:]
                self.logger.warning("Arena popup: %s", popup_text)
                await self._handle_arena_popup(popup_text)
                if any(
                    err in popup_text
                    for err in (
                        "does not play on",
                        "was rejected",
                        "Send a team legal",
                        "There is no bot named",
                        "cannot challenge itself",
                        "Try again in",
                        "could not start your battle",
                        "not taking new battles",
                        "is not open right now",
                    )
                ):
                    self.battle_finished_event.set()
            elif message.startswith("|updatesearch|"):
                try:
                    search_data = orjson.loads(
                        message[len("|updatesearch|") :].strip()
                    )
                    games = search_data.get("games") or {}
                    self.server_games = (
                        set(games.keys()) if isinstance(games, dict) else set()
                    )
                    if (
                        self.draining
                        and not games
                        and not self.active_battles
                        and not self._is_mid_random_pair()
                    ):
                        self.logger.info(
                            "[DRAIN] Server reported no active games in |updatesearch|. Closing connection."
                        )
                        await self.close_connection()
                except Exception:
                    pass
            await orig_handle_message(message)

        self.ps_client._handle_message = _wrapped_handle_message

    def _is_mid_random_pair(self) -> bool:
        return any(
            count % 2 == 1
            for count in self.random_format_games_completed.values()
        )

    async def close_connection(self) -> None:
        ws = getattr(self.ps_client, "websocket", None)
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _safe_send_global(self, command: str) -> bool:
        ws = getattr(self.ps_client, "websocket", None)
        if not self.ps_client.logged_in.is_set() or ws is None:
            return False
        try:
            await self.ps_client.send_message(command)
            return True
        except Exception as exc:
            self.logger.warning("Failed to send %s: %s", command, exc)
            return False

    async def _safe_set_team(self, fmt: str, packed: str) -> bool:
        norm_fmt = fmt.replace("-", "").lower()
        if norm_fmt in self.blocked_formats:
            return False
        ws = getattr(self.ps_client, "websocket", None)
        if not self.ps_client.logged_in.is_set() or ws is None:
            return False
        async with self._team_upload_lock:
            try:
                self.pending_upload_format = norm_fmt
                self.last_uploaded_team_by_format[norm_fmt] = packed
                await self.ps_client.set_team(packed)
                return True
            except Exception as exc:
                self.logger.warning(
                    "Failed to upload team for %s via /utm: %s", norm_fmt, exc
                )
                return False

    async def block_format(self, fmt: str, reason: str) -> None:
        """Stop playing `fmt` and signal to `pkmn.bot` that we cannot play it by
        clearing stored server teams (`/utm null`) and re-uploading teams only for
        the remaining active formats (and `/cancelsearch` if no formats remain)."""
        norm_fmt = fmt.replace("-", "").lower()
        if norm_fmt in self.blocked_formats:
            return
        self.blocked_formats.add(norm_fmt)
        self.active_formats = [
            f for f in self.active_formats if f != norm_fmt
        ]
        self.last_uploaded_team_by_format.pop(norm_fmt, None)
        self.logger.error(
            "[FORMAT BLOCKED] Stopping format '%s' (reason: %s). Active formats remaining: %s",
            norm_fmt,
            reason,
            self.active_formats,
        )
        # Signal to pkmn.bot by clearing all stored format teams via |/utm null
        # and re-uploading teams ONLY for remaining active formats.
        await self._safe_send_global("/utm null")
        await asyncio.sleep(0.1)
        for active_fmt in list(self.active_formats):
            if "random" in active_fmt:
                continue
            packed = self.last_uploaded_team_by_format.get(
                active_fmt
            ) or self.sample_team_for_format(active_fmt)
            if packed:
                await self._safe_set_team(active_fmt, packed)
                await asyncio.sleep(0.15)
        if not self.active_formats:
            self.logger.error(
                "[FORMAT BLOCKED] All formats are blocked; sending /cancelsearch to pkmn.bot."
            )
            await self._safe_send_global("/cancelsearch")

    async def _handle_arena_popup(self, popup_text: str) -> None:
        candidates = list(
            set(self.active_formats) | set(self.team_sets.keys())
        )
        if "was rejected" in popup_text:
            fmt = self.pending_upload_format or _extract_format_from_popup(
                popup_text, candidates
            )
            if fmt:
                ts = self.team_sets.get(fmt)
                rejected_packed = self.last_uploaded_team_by_format.get(fmt)
                if isinstance(ts, PkmnTeamsSet):
                    ts.mark_team_rejected(rejected_packed)
                count = self.consecutive_rejections_by_format.get(fmt, 0) + 1
                self.consecutive_rejections_by_format[fmt] = count
                if count >= 5 or (
                    isinstance(ts, PkmnTeamsSet) and not ts.has_valid_teams()
                ):
                    await self.block_format(
                        fmt, f"team rejected {count} times: {popup_text}"
                    )
                else:
                    new_packed = self.sample_team_for_format(fmt)
                    if new_packed:
                        self.logger.info(
                            "Resampling and uploading replacement team for %s (attempt %d)...",
                            fmt,
                            count,
                        )
                        await self._safe_set_team(fmt, new_packed)
                    else:
                        await self.block_format(
                            fmt, f"no valid teams left after rejection: {popup_text}"
                        )
        elif (
            "This bot does not play on" in popup_text
            or "is not open right now" in popup_text
        ):
            fmt = _extract_format_from_popup(popup_text, candidates)
            if fmt:
                await self.block_format(fmt, popup_text)
        elif "Send a team legal in" in popup_text:
            fmt = _extract_format_from_popup(popup_text, candidates)
            if fmt and fmt not in self.blocked_formats:
                packed = self.sample_team_for_format(fmt)
                if packed:
                    await self._safe_set_team(fmt, packed)
                else:
                    await self.block_format(fmt, popup_text)

    async def request_drain(self, reason: str = "SIGHUP") -> None:
        self.draining = True
        self.rotate_teams_on_battle_end = False
        await self._safe_send_global("/cancelsearch")
        if (
            not self.active_battles
            and not self.server_games
            and not self._is_mid_random_pair()
        ):
            self.logger.info(
                "[DRAIN] Received %s with no active battles. Closing WebSocket immediately...",
                reason,
            )
            await self.close_connection()
        else:
            self.logger.info(
                "[DRAIN] Received %s while battle(s) active (local=%s, server=%s). Sent /cancelsearch; will disconnect once all finish.",
                reason,
                sorted(self.active_battles),
                sorted(self.server_games),
            )

    def _runner_for_format(self, fmt: str) -> ModelPolicyRunner:
        norm_fmt = fmt.replace("-", "").lower()
        if norm_fmt in self.model_runners:
            return self.model_runners[norm_fmt]
        agent_fmt = format_for_agent(norm_fmt)
        return self.model_runners.get(agent_fmt, self.default_runner)

    def sample_team_for_format(self, fmt: str) -> Optional[str]:
        norm_fmt = fmt.replace("-", "").lower()
        if norm_fmt in self.blocked_formats or "random" in norm_fmt:
            return None
        ts = self.team_sets.get(norm_fmt)
        if ts is None:
            return None
        try:
            return ts.yield_team()
        except Exception as exc:
            self.logger.error("Could not sample team for %s: %s", norm_fmt, exc)
            return None

    async def wait_for_login_or_disconnect(
        self, login_timeout: float = 30.0
    ) -> None:
        login_task = asyncio.create_task(self.ps_client.logged_in.wait())
        listen_fut = getattr(self.ps_client, "_listening_coroutine", None)
        waitables = [login_task]
        if listen_fut is not None:
            waitables.append(asyncio.wrap_future(listen_fut))
        await asyncio.wait(
            waitables,
            timeout=login_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not self.ps_client.logged_in.is_set():
            login_task.cancel()
            await self.close_connection()
            raise TimeoutError(
                "WebSocket disconnected or timed out before login completed"
            )

    async def upload_teams_for_formats(
        self, formats: List[str], login_timeout: float = 30.0
    ) -> None:
        await self.wait_for_login_or_disconnect(login_timeout=login_timeout)
        # Clear any stale teams on the server first so only active_formats have stored teams
        await self._safe_send_global("/utm null")
        await asyncio.sleep(0.1)
        for fmt in list(formats):
            norm_fmt = fmt.replace("-", "").lower()
            if norm_fmt in self.blocked_formats or "random" in norm_fmt:
                continue
            packed = self.sample_team_for_format(norm_fmt)
            if packed:
                if await self._safe_set_team(norm_fmt, packed):
                    self.logger.info(
                        "Uploaded validated team for %s via /utm", norm_fmt
                    )
                    await asyncio.sleep(0.15)
            else:
                await self.block_format(
                    norm_fmt, "no valid teams available at startup"
                )

    def create_metamon_battle_for_format(
        self, battle_tag: str, battle_format: str
    ) -> MetamonBackendBattle:
        agent_fmt = format_for_agent(battle_format)
        gen = Dex.from_format(agent_fmt).gen
        return MetamonBackendBattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=gen,
        )

    def teampreview(self, battle: AbstractBattle) -> str:
        # Ensure self._format matches this specific battle's format even when multiple
        # battles across different formats are running concurrently.
        parts = battle.battle_tag.split("-")
        if len(parts) >= 2:
            self._format = parts[1].replace("-", "").lower()
        return super().teampreview(battle)

    async def _create_battle(self, split_message: List[str]) -> AbstractBattle:
        if len(split_message) < 2:
            raise ShowdownException(f"Invalid battle init message: {split_message}")
        battle_format = split_message[1].replace("-", "").lower()
        battle_tag = "-".join(split_message)[1:]

        if battle_tag in self._battles:
            return self._battles[battle_tag]

        self._format = battle_format
        battle = self.create_metamon_battle_for_format(battle_tag, battle_format)
        # Never block on _battle_count_queue if the arena starts extra concurrent battles
        if not self._battle_count_queue.full():
            await self._battle_count_queue.put(None)
        if battle_tag in self._battles:
            if not self._battle_count_queue.empty():
                await self._battle_count_queue.get()
                self._battle_count_queue.task_done()
            return self._battles[battle_tag]
        async with self._battle_start_condition:
            self._battle_semaphore.release()
            self._battle_start_condition.notify_all()
            self._battles[battle_tag] = battle
            self.active_battles.add(battle_tag)
        self.consecutive_rejections_by_format[battle_format] = 0
        self.logger.info(
            "Started battle %s (format=%s, model=%s, active_battles=%d)",
            battle_tag,
            battle_format,
            self._runner_for_format(battle_format).model_name,
            len(self.active_battles),
        )
        # Pre-rotate the stored team for this format safely so a socket error never
        # aborts _create_battle before |player| and |switch| messages are parsed.
        if (
            self.rotate_teams_on_battle_end
            and not self.draining
            and battle_format not in self.blocked_formats
            and "random" not in battle_format
        ):
            packed = self.sample_team_for_format(battle_format)
            if packed:
                await self._safe_set_team(battle_format, packed)
        return battle

    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        maybe_default_order: bool = False,
    ):
        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
        elif battle.teampreview:
            message = self.teampreview(battle)
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = await self.choose_move_async(battle)
            message = choice.message

        # Append rqid per pkmn.bot protocol specification
        rqid = getattr(battle, "_last_request", {}).get("rqid")
        if rqid is not None and not str(message).endswith(f"|{rqid}"):
            message = f"{message}|{rqid}"

        await self.ps_client.send_message(message, battle.battle_tag)

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        raw_room = split_messages[0][0]
        battle_tag = raw_room[1:] if raw_room.startswith(">") else raw_room
        lock = self._battle_locks.setdefault(battle_tag, asyncio.Lock())
        async with lock:
            await self._handle_battle_message_locked(split_messages)

    async def _handle_battle_message_locked(self, split_messages: List[List[str]]):
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = await self._create_battle(battle_info)
        else:
            battle = await self._get_battle(split_messages[0][0])

        for split_message in split_messages[1:]:
            if len(split_message) <= 1:
                continue
            elif split_message[0] == "":
                battle.parse_message(split_message)

            if split_message[1] in self.MESSAGES_TO_IGNORE:
                pass
            elif split_message[1] == "request":
                if split_message[2]:
                    request = orjson.loads(split_message[2])
                    battle.parse_request(request)
                    if battle._wait:
                        self._waiting.set()
                    else:
                        await self._handle_battle_request(battle)
            elif split_message[1] in ("win", "tie"):
                winner = split_message[2] if split_message[1] == "win" else "TIE"
                battle_format = battle.battle_tag.split("-")[1]
                runner = self._runner_for_format(battle_format)
                runner.clear_battle(battle.battle_tag)
                self.active_battles.discard(battle.battle_tag)
                self.server_games.discard(battle.battle_tag)
                self._battles.pop(battle.battle_tag, None)
                self._battle_locks.pop(battle.battle_tag, None)
                self._retried_rqids.pop(battle.battle_tag, None)
                if "random" in battle_format:
                    self.random_format_games_completed[battle_format] = (
                        self.random_format_games_completed.get(battle_format, 0) + 1
                    )
                won = (
                    split_message[1] == "win"
                    and winner.strip().lower() == self.username.strip().lower()
                )
                result_str = (
                    "WIN" if won else ("TIE" if split_message[1] == "tie" else "LOSS")
                )
                self.total_completed_count += 1
                self.completed_battles.append(
                    {
                        "battle_tag": battle.battle_tag,
                        "format": battle_format,
                        "model": runner.model_name,
                        "winner": winner,
                        "result": result_str,
                        "opponent": getattr(battle, "opponent_username", None),
                    }
                )
                if len(self.completed_battles) > 500:
                    self.completed_battles = self.completed_battles[-500:]
                self.logger.info(
                    "Finished %s | format=%s | model=%s | result=%s (winner=%s) | active_remaining=%d | total_done=%d",
                    battle.battle_tag,
                    battle_format,
                    runner.model_name,
                    result_str,
                    winner,
                    len(self.active_battles),
                    self.total_completed_count,
                )
                if not self._battle_count_queue.empty():
                    await self._battle_count_queue.get()
                    self._battle_count_queue.task_done()
                self._battle_finished_callback(battle)
                async with self._battle_end_condition:
                    self._battle_end_condition.notify_all()
                self.battle_finished_event.set()

                # Periodic cleanup of any abandoned battle states and MPS cache
                if self.total_completed_count % 25 == 0:
                    for r in set(self.model_runners.values()):
                        r.prune_stale_battles(self.active_battles)

                # Note: Do NOT send `/leave` on pkmn.bot (unsupported command popup).
                if self.draining:
                    if (
                        not self.active_battles
                        and not self.server_games
                        and not self._is_mid_random_pair()
                    ):
                        self.logger.info(
                            "[DRAIN] All active battles finished (%d completed in session). Closing WebSocket and shutting down.",
                            self.total_completed_count,
                        )
                        await self.close_connection()
                        return
                elif (
                    self.rotate_teams_on_battle_end
                    and battle_format not in self.blocked_formats
                    and "random" not in battle_format
                ):
                    packed = self.sample_team_for_format(battle_format)
                    if packed:
                        await self._safe_set_team(battle_format, packed)
            elif split_message[1] == "error":
                err_text = split_message[2] if len(split_message) > 2 else ""
                self.logger.warning(
                    "Battle error in %s: %s",
                    battle.battle_tag,
                    "|".join(split_message),
                )
                if "There is nothing for you to choose right now" in err_text:
                    pass
                elif err_text.startswith("[Invalid choice] Sorry, too late"):
                    if battle.trapped:
                        self._trying_again.set()
                elif err_text.startswith("[Unavailable choice]"):
                    # Per pkmn.bot protocol, [Unavailable choice] comes with a new |request|
                    # showing what was just revealed (e.g. a disabled move).
                    if err_text.endswith("is disabled"):
                        battle.move_on_next_request = True
                elif err_text.startswith("[Invalid choice]"):
                    if err_text.endswith("is disabled"):
                        battle.move_on_next_request = True
                    rqid = getattr(battle, "_last_request", {}).get("rqid")
                    if (
                        rqid is not None
                        and self._retried_rqids.get(battle.battle_tag) == rqid
                    ):
                        # Already retried this rqid once; fall back to `default` so we never time out
                        self._retried_rqids.pop(battle.battle_tag, None)
                        await self.ps_client.send_message(
                            f"/choose default|{rqid}", battle.battle_tag
                        )
                    else:
                        if rqid is not None:
                            self._retried_rqids[battle.battle_tag] = rqid
                        await self._handle_battle_request(
                            battle, maybe_default_order=True
                        )
            elif split_message[1] == "turn":
                battle._mm_battle.turnlist = battle._mm_battle.turnlist[-2:]
            elif split_message[1] == "bigerror":
                self.logger.warning("Received 'bigerror' message: %s", split_message)
            elif split_message[1] == "uhtml" and split_message[2] == "otsrequest":
                await self._handle_ots_request(battle.battle_tag)

    async def choose_move_async(self, battle: AbstractBattle) -> BattleOrder:
        """Non-blocking turn action selection: extracts state on POKE_LOOP, offloads
        MPS transformer inference to `_INFERENCE_EXECUTOR` with a 15s timeout, and
        falls back to a legal random move if anything fails or stalls."""
        battle_format = battle.battle_tag.split("-")[1]
        runner = self._runner_for_format(battle_format)
        try:
            if (
                battle.active_pokemon is None
                or battle.opponent_active_pokemon is None
            ):
                return self.choose_random_move(battle)
            state = UniversalState.from_Battle(battle)
            legal_universal = UniversalAction.definitely_valid_actions(
                state=state, battle=battle
            )
            if not legal_universal:
                return self.choose_random_move(battle)
            loop = asyncio.get_running_loop()
            universal_action = await asyncio.wait_for(
                loop.run_in_executor(
                    _INFERENCE_EXECUTOR,
                    runner.infer_universal_action,
                    battle.battle_tag,
                    state,
                    legal_universal,
                ),
                timeout=15.0,
            )
            order = universal_action.to_BattleOrder(battle)
            if order is not None:
                return order
        except asyncio.TimeoutError:
            self.logger.error(
                "Inference timed out (>15s) in %s (%s); falling back to fast legal move.",
                battle.battle_tag,
                runner.model_name,
            )
        except Exception as exc:
            self.logger.exception(
                "Exception in %s choose_move_async (%s): %s",
                battle.battle_tag,
                runner.model_name,
                exc,
            )
        return self.choose_random_move(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        battle_format = battle.battle_tag.split("-")[1]
        runner = self._runner_for_format(battle_format)
        try:
            if (
                battle.active_pokemon is None
                or battle.opponent_active_pokemon is None
            ):
                return self.choose_random_move(battle)
            state = UniversalState.from_Battle(battle)
            order = runner.choose_action(battle.battle_tag, state, battle)
            if order is not None:
                return order
        except Exception as exc:
            self.logger.exception(
                "Exception in %s choose_move (%s): %s",
                battle.battle_tag,
                runner.model_name,
                exc,
            )
        return self.choose_random_move(battle)


def fetch_live_arena() -> Dict[str, Any]:
    req = urllib.request.Request(
        "https://pkmn.bot/api/live",
        headers={"User-Agent": "metamon-pkmn-bot/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bot_registered_ladders(bot_name: str) -> Optional[List[str]]:
    name_id = re.sub(r"[^a-z0-9]+", "", bot_name.lower())
    url = f"https://pkmn.bot/api/bots/{name_id}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "metamon-pkmn-bot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ladders = data.get("bot", {}).get("ladders")
        if isinstance(ladders, list):
            return [str(l).replace("-", "").lower() for l in ladders]
    except Exception:
        pass
    return None


def load_dotenv_file(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def build_models_and_teams(
    formats: List[str],
    teams_dir: Optional[str] = DEFAULT_PKMN_TEAMS_DIR,
    team_set_name: str = "competitive",
    single_model: Optional[str] = None,
    temperature: float = 1.0,
    team_preview_checkpoint: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    team_sets: Dict[str, TeamSet] = {}
    for fmt in formats:
        if "random" in fmt:
            continue
        norm_fmt = format_for_agent(fmt)
        if teams_dir and os.path.isdir(os.path.join(teams_dir, norm_fmt)):
            print(
                f"Loading validated pkmn/teams for '{fmt}' from {os.path.join(teams_dir, norm_fmt)}..."
            )
            team_sets[fmt] = PkmnTeamsSet(
                teams_dir=teams_dir,
                battle_format=fmt,
                cache_dir=cache_dir,
            )
        else:
            print(f"Loading Metamon '{team_set_name}' team set for '{fmt}'...")
            team_sets[fmt] = get_metamon_teams(
                fmt, team_set_name, set_type=PokeAgentTeamSet
            )
        print(f"  {fmt}: {len(team_sets[fmt].team_files):,} valid teams ready in memory")

    model_runners: Dict[str, ModelPolicyRunner] = {}
    if single_model:
        print(f"Initializing single model '{single_model}' for all formats...")
        runner = ModelPolicyRunner(
            model_name=single_model, action_temperature=temperature
        )
        for fmt in formats:
            model_runners[fmt] = runner
        default_runner = runner
    else:
        need_tauros = any(fmt.startswith("gen1") for fmt in formats)
        need_kakuna = (
            any(not fmt.startswith("gen1") for fmt in formats) or not need_tauros
        )
        tauros_runner = None
        kakuna_runner = None
        if need_tauros:
            print("Initializing TaurosV0 for Gen 1 (gen1ou)...")
            tauros_runner = ModelPolicyRunner(
                model_name="TaurosV0", action_temperature=temperature
            )
        if need_kakuna:
            print("Initializing Kakuna for Gen 3 / Gen 9 (gen3ou, gen9ou)...")
            kakuna_runner = ModelPolicyRunner(
                model_name="Kakuna", action_temperature=temperature
            )
        default_runner = kakuna_runner or tauros_runner
        for fmt in formats:
            if fmt.startswith("gen1") and tauros_runner is not None:
                model_runners[fmt] = tauros_runner
            else:
                model_runners[fmt] = default_runner

    team_preview_model = None
    if team_preview_checkpoint and os.path.exists(team_preview_checkpoint):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Gen 9 TeamPreviewModel from {team_preview_checkpoint}...")
        team_preview_model = TeamPreviewModel.load_from_checkpoint(
            checkpoint_path=team_preview_checkpoint,
            device=device,
            use_argmax=True,
        )

    return model_runners, default_runner, team_sets, team_preview_model


async def _run_unrated_challenges(
    bot_name: str,
    bot_token: str,
    opponent: str,
    formats: List[str],
    n_battles_per_format: int,
    model_runners: Dict[str, ModelPolicyRunner],
    default_runner: ModelPolicyRunner,
    team_sets: Dict[str, TeamSet],
    team_preview_model: Optional[TeamPreviewModel],
) -> List[Dict[str, Any]]:
    """Run unrated practice challenges against `opponent` without entering rated matchmaking."""
    player = PkmnBotPlayer(
        model_runners=model_runners,
        default_runner=default_runner,
        team_sets=team_sets,
        formats=formats,
        rotate_teams_on_battle_end=False,
        team_preview_model=team_preview_model,
        account_configuration=AccountConfiguration(bot_name, bot_token),
        server_configuration=PKMN_BOT_SERVER,
        battle_format=formats[0],
        team=None,
        avatar=None,
        max_concurrent_battles=1,
        log_level=logging.INFO,
        ping_interval=25,
        ping_timeout=25,
    )

    try:
        await player.ps_client.logged_in.wait()
        print(
            f"Logged in to pkmn.bot as '{bot_name}' (unrated challenge mode vs '{opponent}')."
        )

        for fmt in formats:
            if fmt in player.blocked_formats:
                continue
            battles_per_challenge = 2 if "random" in fmt else 1
            for round_idx in range(n_battles_per_format):
                target_count = len(player.completed_battles) + battles_per_challenge
                packed = player.sample_team_for_format(fmt)
                if packed:
                    await player._safe_set_team(fmt, packed)
                await player.ps_client.send_message(f"/challenge {opponent}, {fmt}")
                print(
                    f"Sent unrated challenge to {opponent} in {fmt} "
                    f"(round {round_idx + 1}/{n_battles_per_format}, model={player._runner_for_format(fmt).model_name})..."
                )
                while len(player.completed_battles) < target_count:
                    player.battle_finished_event.clear()
                    prev_popups = len(player.popups)
                    await player.battle_finished_event.wait()
                    if (
                        len(player.completed_battles) < target_count
                        and len(player.popups) > prev_popups
                    ):
                        print(
                            f"Challenge in {fmt} aborted due to arena popup: {player.popups[-1]}"
                        )
                        break
    finally:
        await player.close_connection()

    return player.completed_battles


class _DrainState:
    def __init__(self) -> None:
        self.draining: bool = False
        self.reason: str = ""
        self.current_player: Optional[PkmnBotPlayer] = None

    def trigger(self, reason: str) -> None:
        self.draining = True
        self.reason = reason
        player = self.current_player
        if player is not None:
            asyncio.run_coroutine_threadsafe(
                player.request_drain(reason), POKE_LOOP
            )


async def _stay_online_ranked(
    bot_name: str,
    bot_token: str,
    formats: List[str],
    model_runners: Dict[str, ModelPolicyRunner],
    default_runner: ModelPolicyRunner,
    team_sets: Dict[str, TeamSet],
    team_preview_model: Optional[TeamPreviewModel],
    max_concurrent_battles: int = 5,
    drain_state: Optional[_DrainState] = None,
) -> None:
    """Stay online continuously on pkmn.bot for rated matchmaking with auto-reconnect,
    blocked-format detection/signaling, and SIGHUP drain."""
    if drain_state is None:
        drain_state = _DrainState()

    blocked_formats: Set[str] = set()
    registered_ladders = fetch_bot_registered_ladders(bot_name)
    if registered_ladders is not None:
        for fmt in formats:
            norm_fmt = fmt.replace("-", "").lower()
            if norm_fmt not in registered_ladders:
                print(
                    f"[FORMAT BLOCKED] '{bot_name}' is not registered on ladder '{norm_fmt}' on pkmn.bot (registered={registered_ladders}); skipping '{norm_fmt}'."
                )
                blocked_formats.add(norm_fmt)

    while True:
        # Clear any stale episode states across reconnects
        for r in set(model_runners.values()):
            r.clear_all_battles()
        gc.collect()

        active_formats = [
            f.replace("-", "").lower()
            for f in formats
            if f.replace("-", "").lower() not in blocked_formats
        ]
        if not active_formats:
            print(
                f"[STOPPED] All requested formats ({formats}) are blocked ({sorted(blocked_formats)}). Exiting."
            )
            break

        player = PkmnBotPlayer(
            model_runners=model_runners,
            default_runner=default_runner,
            team_sets=team_sets,
            formats=active_formats,
            blocked_formats=blocked_formats,
            rotate_teams_on_battle_end=not drain_state.draining,
            team_preview_model=team_preview_model,
            account_configuration=AccountConfiguration(bot_name, bot_token),
            server_configuration=PKMN_BOT_SERVER,
            battle_format=active_formats[0],
            team=None,
            avatar=None,
            max_concurrent_battles=max(max_concurrent_battles, 20),
            log_level=logging.INFO,
            ping_interval=25,
            ping_timeout=25,
        )
        player.draining = drain_state.draining
        drain_state.current_player = player
        try:
            if drain_state.draining:
                await player.wait_for_login_or_disconnect()
                await player._safe_send_global("/cancelsearch")
                print(
                    f"'{bot_name}' reconnected in drain mode (/cancelsearch sent) to finish any remaining battles..."
                )
            else:
                await player.upload_teams_for_formats(active_formats)
                print(
                    f"'{bot_name}' is online on pkmn.bot with validated pkmn/teams uploaded for {player.active_formats} "
                    f"(blocked={sorted(player.blocked_formats)}, max_concurrent_battles={max_concurrent_battles}). "
                    f"Waiting for matches... (Send SIGHUP to pid {os.getpid()} to drain)"
                )
            if hasattr(player.ps_client, "_listening_coroutine"):
                await asyncio.wrap_future(player.ps_client._listening_coroutine)
            else:
                await player.ps_client.wait_for_disconnect()
        except Exception as exc:
            if not drain_state.draining:
                print(f"Connection interrupted ({exc}), reconnecting in 3s...")

        if (
            drain_state.draining
            and not player.active_battles
            and not player.server_games
            and not player._is_mid_random_pair()
        ):
            print(
                f"[DRAIN COMPLETE] '{bot_name}' disconnected cleanly after completing {player.total_completed_count} battle(s)."
            )
            break

        await asyncio.sleep(3)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    load_dotenv_file(os.path.join(repo_root, ".env"))

    parser = argparse.ArgumentParser(description="Run Metamon on https://pkmn.bot")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["gen1ou", "gen3ou", "gen9ou"],
        help="Formats to play (default: gen1ou gen3ou gen9ou)",
    )
    parser.add_argument(
        "--teams_dir",
        type=str,
        default=DEFAULT_PKMN_TEAMS_DIR,
        help=f"Path to pkmn/teams repository (default: {DEFAULT_PKMN_TEAMS_DIR})",
    )
    parser.add_argument(
        "--team_set",
        type=str,
        default="competitive",
        help="Fallback Metamon team set name if --teams_dir does not contain format (default: competitive)",
    )
    parser.add_argument(
        "--single_model",
        type=str,
        default=None,
        help="Optional single model name (e.g. Kakuna). If omitted, uses TaurosV0 for Gen 1 and Kakuna for Gen 3/9.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Action sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--team_preview_checkpoint",
        type=str,
        default=os.path.join(
            repo_root,
            "metamon/backend/team_preview/gen9ou_high_elo_v4/best_model.pt",
        ),
        help="Path to Gen 9 team preview checkpoint",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        default=None,
        help="Challenge a specific online bot (e.g. Cocoa) for unrated practice battles and exit.",
    )
    parser.add_argument(
        "--n_battles",
        type=int,
        default=1,
        help="Number of unrated challenge battles per format when --challenge is set (default: 1).",
    )
    parser.add_argument(
        "--ranked",
        action="store_true",
        help="Stay online continuously in the pkmn.bot ranked matchmaking pool.",
    )
    parser.add_argument(
        "--max_concurrent_battles",
        type=int,
        default=5,
        help="Maximum concurrent battles to support (default: 5, matching pkmn.bot 'Battles at once' limit).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Load models and teams offline and exit without connecting to pkmn.bot.",
    )
    args = parser.parse_args()

    _configure_bounded_logging(repo_root)
    cache_dir = os.path.join(repo_root, ".cache")

    model_runners, default_runner, team_sets, team_preview_model = (
        build_models_and_teams(
            formats=args.formats,
            teams_dir=args.teams_dir,
            team_set_name=args.team_set,
            single_model=args.single_model,
            temperature=args.temperature,
            team_preview_checkpoint=args.team_preview_checkpoint,
            cache_dir=cache_dir,
        )
    )

    if args.smoke_test:
        for fmt in args.formats:
            runner = model_runners[fmt]
            sample_team = team_sets[fmt].yield_team() if fmt in team_sets else None
            print(
                f"[SMOKE TEST OK] format={fmt} -> model={runner.model_name}@{runner.checkpoint} "
                f"| valid_teams={len(team_sets[fmt].team_files) if fmt in team_sets else 0} "
                f"| sample_team_len={len(sample_team) if sample_team else 0}"
            )
        return

    bot_name = os.environ.get("ARENA_BOT_NAME")
    bot_token = os.environ.get("ARENA_BOT_TOKEN")
    if not bot_name or not bot_token:
        raise ValueError(
            "ARENA_BOT_NAME and ARENA_BOT_TOKEN must be set in environment or .env"
        )

    if args.challenge:
        results = asyncio.run_coroutine_threadsafe(
            _run_unrated_challenges(
                bot_name=bot_name,
                bot_token=bot_token,
                opponent=args.challenge,
                formats=args.formats,
                n_battles_per_format=args.n_battles,
                model_runners=model_runners,
                default_runner=default_runner,
                team_sets=team_sets,
                team_preview_model=team_preview_model,
            ),
            POKE_LOOP,
        ).result()
        print("\n=== Unrated Challenge Summary ===")
        print(json.dumps(results, indent=2))
    elif args.ranked:
        drain_state = _DrainState()
        pid_file = os.path.join(repo_root, ".cache", "pkmn_bot.pid")
        os.makedirs(os.path.dirname(pid_file), exist_ok=True)
        with open(pid_file, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")

        def _on_drain_signal(signum, _frame):
            sig_name = signal.Signals(signum).name
            print(
                f"\n[SIGNAL] Caught {sig_name} — initiating graceful drain (will exit after any in-progress battle finishes)...",
                flush=True,
            )
            drain_state.trigger(sig_name)

        signal.signal(signal.SIGHUP, _on_drain_signal)
        signal.signal(signal.SIGUSR1, _on_drain_signal)

        fut = asyncio.run_coroutine_threadsafe(
            _stay_online_ranked(
                bot_name=bot_name,
                bot_token=bot_token,
                formats=args.formats,
                model_runners=model_runners,
                default_runner=default_runner,
                team_sets=team_sets,
                team_preview_model=team_preview_model,
                max_concurrent_battles=args.max_concurrent_battles,
                drain_state=drain_state,
            ),
            POKE_LOOP,
        )
        try:
            while not fut.done():
                try:
                    fut.result(timeout=0.5)
                except concurrent.futures.TimeoutError:
                    continue
        finally:
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except OSError:
                    pass
    else:
        raise ValueError(
            "Specify either --smoke_test, --challenge <BotName>, or --ranked."
        )


if __name__ == "__main__":
    main()
