"""
Connect Metamon agents (e.g. TaurosV0 for Gen 1 OU + Kakuna for Gen 3 / Gen 9 OU)
to https://pkmn.bot.

Supports:
  1. Unrated practice challenges against a specific bot (e.g. `--challenge Cocoa`)
     without entering the rated matchmaking pool.
  2. Ranked matchmaking (`--ranked`) across multiple ladders (`gen1ou`, `gen3ou`,
     `gen9ou`, `gen9randombattle`) on a single WebSocket connection with automatic
     reconnection and per-battle team rotation from Metamon's `"competitive"` TeamSet.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import dataclass
import json
import logging
import math
import os
import random
import time
import urllib.request
from typing import Any, Dict, List, Optional

import gin
import numpy as np
import orjson
import torch

import amago
import amago.nets.transformer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines
from poke_env.environment import AbstractBattle
from poke_env.exceptions import ShowdownException
from poke_env.player import BattleOrder, DefaultBattleOrder
from poke_env.ps_client import PSClient

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

    def _get_or_create_episode(self, battle_tag: str) -> _BattleEpisodeState:
        if battle_tag not in self._battle_states:
            obs_space = copy.deepcopy(self.pretrained_model.observation_space)
            obs_space.reset()
            action_space = copy.deepcopy(self.pretrained_model.action_space)
            reward_fn = copy.deepcopy(self.pretrained_model.reward_function)
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
            )
        return self._battle_states[battle_tag]

    def clear_battle(self, battle_tag: str) -> None:
        self._battle_states.pop(battle_tag, None)

    def choose_action(
        self, battle_tag: str, state: UniversalState, battle: AbstractBattle
    ) -> Optional[BattleOrder]:
        ep = self._get_or_create_episode(battle_tag)
        obs = ep.obs_space.state_to_obs(state)

        legal_universal = UniversalAction.definitely_valid_actions(
            state=state, battle=battle
        )
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

        universal_action = ep.action_space.agent_output_to_action(
            state=state, agent_output=action_idx
        )
        return universal_action.to_BattleOrder(battle)


class PkmnBotPSClient(PSClient):
    """PSClient tailored for pkmn.bot protocol (direct /trn token login, no /avatar)."""

    async def log_in(self, split_message: List[str]):
        if self.account_configuration.password:
            await self.send_message(
                f"/trn {self.username},0,{self.account_configuration.password}"
            )
        else:
            await self.send_message(f"/trn {self.username},0,")


class PkmnBotPlayer(MetamonPlayer):
    """Multi-format Metamon player for pkmn.bot routing by battle format."""

    def __init__(
        self,
        model_runners: Dict[str, ModelPolicyRunner],
        default_runner: ModelPolicyRunner,
        team_sets: Dict[str, TeamSet],
        rotate_teams_on_battle_end: bool = True,
        team_preview_model: Optional[TeamPreviewModel] = None,
        *args,
        **kwargs,
    ):
        self.model_runners = model_runners
        self.default_runner = default_runner
        self.team_sets = team_sets
        self.rotate_teams_on_battle_end = rotate_teams_on_battle_end
        self.completed_battles: List[Dict[str, Any]] = []
        self.popups: List[str] = []
        self.battle_finished_event = asyncio.Event()
        super().__init__(
            *args,
            team_preview_model=team_preview_model,
            start_timer_on_battle_start=False,
            **kwargs,
        )
        # Replace PSClient log_in with direct pkmn.bot login (avoids /avatar popup)
        self.ps_client.log_in = PkmnBotPSClient.log_in.__get__(
            self.ps_client, PSClient
        )
        orig_handle_message = self.ps_client._handle_message

        async def _wrapped_handle_message(message: str):
            if message.startswith("|popup|"):
                popup_text = message[len("|popup|") :].strip()
                self.popups.append(popup_text)
                self.logger.warning("Arena popup: %s", popup_text)
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
                    )
                ):
                    self.battle_finished_event.set()
            await orig_handle_message(message)

        self.ps_client._handle_message = _wrapped_handle_message

    def _runner_for_format(self, fmt: str) -> ModelPolicyRunner:
        norm_fmt = fmt.replace("-", "").lower()
        if norm_fmt in self.model_runners:
            return self.model_runners[norm_fmt]
        agent_fmt = format_for_agent(norm_fmt)
        return self.model_runners.get(agent_fmt, self.default_runner)

    def sample_team_for_format(self, fmt: str) -> Optional[str]:
        norm_fmt = fmt.replace("-", "").lower()
        if "random" in norm_fmt:
            return None
        ts = self.team_sets.get(norm_fmt)
        if ts is None:
            return None
        return ts.yield_team()

    async def upload_teams_for_formats(self, formats: List[str]) -> None:
        await self.ps_client.logged_in.wait()
        for fmt in formats:
            packed = self.sample_team_for_format(fmt)
            if packed:
                await self.ps_client.set_team(packed)
                self.logger.info("Uploaded competitive team for %s via /utm", fmt)

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

    async def _create_battle(self, split_message: List[str]) -> AbstractBattle:
        if len(split_message) < 2:
            raise ShowdownException(f"Invalid battle init message: {split_message}")
        battle_format = split_message[1].replace("-", "").lower()
        battle_tag = "-".join(split_message)[1:]

        if battle_tag in self._battles:
            return self._battles[battle_tag]

        # Update self._format to the active battle's format so teampreview() uses it
        self._format = battle_format
        battle = self.create_metamon_battle_for_format(battle_tag, battle_format)
        await self._battle_count_queue.put(None)
        if battle_tag in self._battles:
            await self._battle_count_queue.get()
            return self._battles[battle_tag]
        async with self._battle_start_condition:
            self._battle_semaphore.release()
            self._battle_start_condition.notify_all()
            self._battles[battle_tag] = battle
        self.logger.info(
            "Started battle %s (format=%s, model=%s)",
            battle_tag,
            battle_format,
            self._runner_for_format(battle_format).model_name,
        )
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
            choice = self.choose_move(battle)
            if isinstance(choice, asyncio.Future) or asyncio.iscoroutine(choice):
                choice = await choice
            message = choice.message

        # Append rqid per pkmn.bot protocol specification
        rqid = getattr(battle, "_last_request", {}).get("rqid")
        if rqid is not None and not str(message).endswith(f"|{rqid}"):
            message = f"{message}|{rqid}"

        await self.ps_client.send_message(message, battle.battle_tag)

    async def _handle_battle_message(self, split_messages: List[List[str]]):
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
                won = split_message[1] == "win" and winner.strip().lower() == self.username.strip().lower()
                result_str = "WIN" if won else ("TIE" if split_message[1] == "tie" else "LOSS")
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
                self.logger.info(
                    "Finished %s | format=%s | model=%s | result=%s (winner=%s)",
                    battle.battle_tag,
                    battle_format,
                    runner.model_name,
                    result_str,
                    winner,
                )
                await self._battle_count_queue.get()
                self._battle_count_queue.task_done()
                self._battle_finished_callback(battle)
                async with self._battle_end_condition:
                    self._battle_end_condition.notify_all()
                self.battle_finished_event.set()
                # Note: Do NOT send `/leave` on pkmn.bot (unsupported command popup).
                if self.rotate_teams_on_battle_end and "random" not in battle_format:
                    packed = self.sample_team_for_format(battle_format)
                    if packed:
                        await self.ps_client.set_team(packed)
            elif split_message[1] == "error":
                err_text = split_message[2] if len(split_message) > 2 else ""
                self.logger.warning(
                    "Battle error in %s: %s",
                    battle.battle_tag,
                    "|".join(split_message),
                )
                if "There is nothing for you to choose right now" in err_text:
                    pass
                elif err_text.startswith(
                    "[Invalid choice] Sorry, too late to make a different move"
                ):
                    if battle.trapped:
                        self._trying_again.set()
                elif err_text.endswith("is disabled"):
                    battle.move_on_next_request = True
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif err_text.startswith("[Invalid choice]") or err_text.startswith(
                    "[Unavailable choice]"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[1] == "turn":
                battle._mm_battle.turnlist = battle._mm_battle.turnlist[-2:]
            elif split_message[1] == "bigerror":
                self.logger.warning("Received 'bigerror' message: %s", split_message)
            elif split_message[1] == "uhtml" and split_message[2] == "otsrequest":
                await self._handle_ots_request(battle.battle_tag)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        battle_format = battle.battle_tag.split("-")[1]
        runner = self._runner_for_format(battle_format)
        try:
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
    team_set_name: str = "competitive",
    single_model: Optional[str] = None,
    temperature: float = 1.0,
    team_preview_checkpoint: Optional[str] = None,
):
    print(f"Loading '{team_set_name}' team sets for formats: {formats}...")
    team_sets: Dict[str, TeamSet] = {}
    for fmt in formats:
        if "random" in fmt:
            continue
        team_sets[fmt] = get_metamon_teams(
            fmt, team_set_name, set_type=PokeAgentTeamSet
        )
        print(f"  {fmt}: {len(team_sets[fmt].team_files)} teams ready")

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
        need_kakuna = any(not fmt.startswith("gen1") for fmt in formats) or not need_tauros
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
        print(f"Logged in to pkmn.bot as '{bot_name}' (unrated challenge mode vs '{opponent}').")

        for fmt in formats:
            battles_per_challenge = 2 if "random" in fmt else 1
            for round_idx in range(n_battles_per_format):
                target_count = len(player.completed_battles) + battles_per_challenge
                packed = player.sample_team_for_format(fmt)
                if packed:
                    await player.ps_client.set_team(packed)
                await player.ps_client.send_message(f"/challenge {opponent}, {fmt}")
                print(
                    f"Sent unrated challenge to {opponent} in {fmt} "
                    f"(round {round_idx + 1}/{n_battles_per_format}, model={player._runner_for_format(fmt).model_name})..."
                )
                while len(player.completed_battles) < target_count:
                    player.battle_finished_event.clear()
                    prev_popups = len(player.popups)
                    await player.battle_finished_event.wait()
                    if len(player.completed_battles) < target_count and len(player.popups) > prev_popups:
                        print(f"Challenge in {fmt} aborted due to arena popup: {player.popups[-1]}")
                        break
    finally:
        if hasattr(player.ps_client, "websocket") and player.ps_client.websocket:
            await player.ps_client.websocket.close()

    return player.completed_battles


async def _stay_online_ranked(
    bot_name: str,
    bot_token: str,
    formats: List[str],
    model_runners: Dict[str, ModelPolicyRunner],
    default_runner: ModelPolicyRunner,
    team_sets: Dict[str, TeamSet],
    team_preview_model: Optional[TeamPreviewModel],
) -> None:
    """Stay online continuously on pkmn.bot for rated matchmaking with auto-reconnect."""
    while True:
        player = PkmnBotPlayer(
            model_runners=model_runners,
            default_runner=default_runner,
            team_sets=team_sets,
            rotate_teams_on_battle_end=True,
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
            await player.upload_teams_for_formats(formats)
            print(
                f"'{bot_name}' is online on pkmn.bot with competitive teams uploaded for {formats}. "
                f"Waiting for matches..."
            )
            await player.ps_client.wait_for_disconnect()
        except AttributeError:
            # Fallback if poke-env version uses _listening_coroutine
            if hasattr(player.ps_client, "_listening_coroutine"):
                await player.ps_client._listening_coroutine
        except Exception as exc:
            print(f"Connection interrupted ({exc}), reconnecting in 3s...")
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
        "--team_set",
        type=str,
        default="competitive",
        help="Metamon team set name (default: competitive)",
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
        "--smoke_test",
        action="store_true",
        help="Load models and teams offline and exit without connecting to pkmn.bot.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_runners, default_runner, team_sets, team_preview_model = (
        build_models_and_teams(
            formats=args.formats,
            team_set_name=args.team_set,
            single_model=args.single_model,
            temperature=args.temperature,
            team_preview_checkpoint=args.team_preview_checkpoint,
        )
    )

    if args.smoke_test:
        for fmt in args.formats:
            runner = model_runners[fmt]
            sample_team = team_sets[fmt].yield_team() if fmt in team_sets else None
            print(
                f"[SMOKE TEST OK] format={fmt} -> model={runner.model_name}@{runner.checkpoint} "
                f"| sample_team_len={len(sample_team) if sample_team else 0}"
            )
        return

    bot_name = os.environ.get("ARENA_BOT_NAME")
    bot_token = os.environ.get("ARENA_BOT_TOKEN")
    if not bot_name or not bot_token:
        raise ValueError("ARENA_BOT_NAME and ARENA_BOT_TOKEN must be set in environment or .env")

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
        asyncio.run_coroutine_threadsafe(
            _stay_online_ranked(
                bot_name=bot_name,
                bot_token=bot_token,
                formats=args.formats,
                model_runners=model_runners,
                default_runner=default_runner,
                team_sets=team_sets,
                team_preview_model=team_preview_model,
            ),
            POKE_LOOP,
        ).result()
    else:
        raise ValueError("Specify either --smoke_test, --challenge <BotName>, or --ranked.")


if __name__ == "__main__":
    main()
