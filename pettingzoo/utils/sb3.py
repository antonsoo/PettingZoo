"""Optional Stable-Baselines3 integration for Parallel environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
from gymnasium.utils.env_checker import data_equivalence
from gymnasium.vector.utils import concatenate, create_empty_array
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from typing_extensions import override

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class SB3ParallelVecEnv(VecEnv):
    """Train one SB3 policy across agents in one or more Parallel environments.

    Each factory creates an independent world. Worlds run sequentially in this
    process; within a world, all agents act together. Vector slots are ordered
    first by factory, then by that world's ``possible_agents``. Therefore
    ``num_envs`` is the total number of agents, not the number of worlds.

    All agents must have equal observation and action spaces supported by SB3,
    and remain present until their world ends. Wrap environments with early
    agent departures in :class:`BlackDeathParallelV4` first. Agents appearing
    after reset are not supported. A world resets as a unit when all its agents
    finish; terminal observations and time-limit flags follow SB3's VecEnv API.

    Seeds and reset options apply to worlds, since their agents share a reset.
    ``seed(s)`` assigns ``s + world_index`` and returns that seed for each slot
    in the world. A list passed to ``set_options`` contains one dict per agent
    slot, as in SB3; options for agents sharing a world must agree.
    Attribute/method indices still refer to agent slots. Mutating an attribute
    or calling a method affects the whole world, once per selected world.

    Stable-Baselines3 is an optional dependency: install it separately before
    importing this module. This adapter does not change any environment spaces
    or preprocess observations.

    Args:
        env_fns: Factories that each create a fresh Parallel environment.
    """

    def __init__(self, env_fns: Sequence[Callable[[], ParallelEnv[Any, Any, Any]]]):
        if not env_fns:
            raise ValueError("At least one environment factory is required.")
        self.envs: list[ParallelEnv[Any, Any, Any]] = []
        self._agents: list[list[Any]] = []
        self._slots: list[int] = []
        self._actions: np.ndarray | None = None
        self._closed = False
        try:
            for fn in env_fns:
                env = fn()
                if not isinstance(env, ParallelEnv):
                    raise TypeError("Each factory must return a ParallelEnv.")
                self.envs.append(env)
                if not env.possible_agents:
                    raise ValueError(
                        "Each world must have at least one possible agent."
                    )
                if len(set(env.possible_agents)) != len(env.possible_agents):
                    raise ValueError("possible_agents must contain unique agent IDs.")
                self._agents.append(list(env.possible_agents))
                self._slots.extend([len(self.envs) - 1] * len(env.possible_agents))

            if len({id(env.unwrapped) for env in self.envs}) != len(self.envs):
                raise ValueError("Each factory must create a distinct environment.")

            first, agent = self.envs[0], self._agents[0][0]
            observation_space = first.observation_space(agent)
            action_space = first.action_space(agent)
            for env, agents in zip(self.envs, self._agents):
                for agent in agents:
                    if env.observation_space(agent) != observation_space:
                        raise ValueError(
                            "All agents must have equal observation spaces."
                        )
                    if env.action_space(agent) != action_space:
                        raise ValueError("All agents must have equal action spaces.")

            super().__init__(len(self._slots), observation_space, action_space)
            self._world_seeds: list[int | None] = [None] * len(self.envs)
            self._world_options: list[dict[str, Any]] = [{} for _ in self.envs]
        except Exception:
            self.close()
            raise

    def _check_agents(self, world: int) -> None:
        if set(self.envs[world].agents) != set(self._agents[world]):
            raise ValueError(
                "Agent sets must stay fixed until the world ends. Use "
                "BlackDeathParallelV4 for environments with early agent departures."
            )

    def _batch(self, observations: list[Any]) -> VecEnvObs:
        return concatenate(
            self.observation_space,
            observations,
            create_empty_array(self.observation_space, self.num_envs),
        )

    def _reset_world(self, world: int, **kwargs: Any) -> list[Any]:
        env, agents = self.envs[world], self._agents[world]
        observations, infos = env.reset(**kwargs)
        self._check_agents(world)
        start = sum(len(group) for group in self._agents[:world])
        for offset, agent in enumerate(agents):
            self.reset_infos[start + offset] = deepcopy(infos.get(agent, {}))
        return [observations[agent] for agent in agents]

    @override
    def reset(self) -> VecEnvObs:
        """Reset worlds, consuming scheduled seeds and options exactly once."""
        self._actions = None
        observations = []
        for world in range(len(self.envs)):
            observations.extend(
                self._reset_world(
                    world,
                    seed=self._world_seeds[world],
                    options=deepcopy(self._world_options[world]),
                )
            )
        self._world_seeds = [None] * len(self.envs)
        self._world_options = [{} for _ in self.envs]
        return self._batch(observations)

    @override
    def step_async(self, actions: np.ndarray) -> None:
        """Store one action per agent slot for the next joint step."""
        if self._actions is not None:
            raise RuntimeError("A step is already pending; call step_wait first.")
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}.")
        self._actions = actions

    @override
    def step_wait(self) -> VecEnvStepReturn:
        """Step each world once and reset completed worlds on the same call."""
        if self._actions is None:
            raise RuntimeError("Call step_async before step_wait.")
        actions, self._actions = self._actions, None
        observations, rewards, dones, infos = [], [], [], []
        start = 0
        for world, (env, agents) in enumerate(zip(self.envs, self._agents)):
            self._check_agents(world)
            joint_action = {
                agent: actions[start + offset] for offset, agent in enumerate(agents)
            }
            obs, rew, term, trunc, info = env.step(joint_action)
            ended = [bool(term[agent] or trunc[agent]) for agent in agents]
            if any(ended) and not all(ended):
                raise ValueError(
                    "Agents must finish together. Wrap early departures with "
                    "BlackDeathParallelV4 before creating the vector environment."
                )
            world_infos = [deepcopy(info.get(agent, {})) for agent in agents]
            for agent, agent_info in zip(agents, world_infos):
                agent_info["TimeLimit.truncated"] = bool(
                    trunc[agent] and not term[agent]
                )
                if all(ended):
                    # Reset may reuse the same observation arrays; keep the old values.
                    agent_info["terminal_observation"] = deepcopy(obs[agent])
            rewards.extend(rew[agent] for agent in agents)
            dones.extend(ended)
            infos.extend(world_infos)
            if all(ended):
                observations.extend(self._reset_world(world))
            else:
                self._check_agents(world)
                observations.extend(obs[agent] for agent in agents)
            start += len(agents)
        return (
            self._batch(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    @override
    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        """Schedule a distinct seed per world for the next explicit reset."""
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
        self._world_seeds = [seed + world for world in range(len(self.envs))]
        return [self._world_seeds[world] for world in self._slots]

    @override
    def set_options(
        self, options: list[dict[str, Any]] | dict[str, Any] | None = None
    ) -> None:
        """Schedule reset options, requiring agreement within each world."""
        if options is None or isinstance(options, dict):
            world_options = [options or {} for _ in self.envs]
        else:
            if len(options) != self.num_envs:
                raise ValueError("Pass one options dict per agent slot.")
            world_options = []
            start = 0
            for agents in self._agents:
                group = options[start : start + len(agents)]
                if not all(
                    data_equivalence(group[0], item, exact=True) for item in group
                ):
                    raise ValueError(
                        "Reset options must agree for agents in the same world."
                    )
                world_options.append(group[0])
                start += len(agents)
        self._world_options = deepcopy(world_options)

    @override
    def close(self) -> None:
        """Close every world once."""
        if not self._closed:
            self._closed = True
            seen = set()
            for env in self.envs:
                if id(env) not in seen:
                    env.close()
                    seen.add(id(env))

    def _selected_worlds(self, indices: VecEnvIndices) -> list[int]:
        return [self._slots[index] for index in self._get_indices(indices)]

    def _world_attr(self, world: int, name: str) -> Any:
        try:
            return getattr(self.envs[world], name)
        except AttributeError:
            return getattr(self.envs[world].unwrapped, name)

    @override
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return each selected agent slot's world attribute."""
        return [
            self._world_attr(world, attr_name)
            for world in self._selected_worlds(indices)
        ]

    @override
    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set the attribute once in each selected world."""
        for world in dict.fromkeys(self._selected_worlds(indices)):
            setattr(self.envs[world], attr_name, value)

    @override
    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call a method once per selected world, returning one result per slot."""
        worlds = self._selected_worlds(indices)
        results = {
            world: self._world_attr(world, method_name)(*method_args, **method_kwargs)
            for world in dict.fromkeys(worlds)
        }
        return [results[world] for world in worlds]

    @override
    def env_is_wrapped(
        self, wrapper_class: type, indices: VecEnvIndices = None
    ) -> list[bool]:
        """Check the Parallel wrapper chain for each selected agent slot."""
        results = []
        for world in self._selected_worlds(indices):
            env = self.envs[world]
            while not isinstance(env, wrapper_class) and isinstance(
                env, BaseParallelWrapper
            ):
                env = env.env
            results.append(isinstance(env, wrapper_class))
        return results

    @override
    def get_images(self) -> Sequence[np.ndarray | None]:
        """Render each world once and expose its image for each agent slot."""
        if self.render_mode != "rgb_array":
            return [None] * self.num_envs
        images = [cast(np.ndarray | None, env.render()) for env in self.envs]
        return [images[world] for world in self._slots]
