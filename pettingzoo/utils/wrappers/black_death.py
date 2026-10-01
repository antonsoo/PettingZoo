"""Wrapper that keeps agents in a parallel environment after they have left it."""

from __future__ import annotations

from typing import Any

import gymnasium.spaces
import numpy as np
from gymnasium.spaces import Box
from typing_extensions import override

from pettingzoo.utils.env import ActionType, AgentID, ParallelEnv
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


def _check_space(space: gymnasium.spaces.Space[Any], agent: Any) -> Box:
    """Returns ``space`` if a zero observation can stand in for it, else raises."""
    if not isinstance(space, Box):
        raise TypeError(
            "BlackDeathParallelV4 only works with Box observation spaces, "
            f"got {space} for agent {agent!r}."
        )
    return space


class BlackDeathParallelV4(BaseParallelWrapper[AgentID, np.ndarray, ActionType]):
    """Keeps every agent in the environment until the whole episode ends.

    A parallel environment stops returning values for an agent once it has
    terminated or been truncated. With this wrapper, such an agent stays in
    ``agents`` and in every dictionary returned by ``step``: its observation is
    an all-zeros array with the shape and dtype of its observation space, its
    reward is ``0.0`` and its info is empty. Any action supplied for it is
    ignored; only the actions of agents that are still in the underlying
    environment are forwarded. An agent that joins during the episode is added
    to ``agents`` and kept in the same way.

    While any agent is still active, ``terminations`` and ``truncations`` are
    ``False`` for every agent, including the ones that have already left. On the
    step that ends the episode, each agent reports the flags it left with, so an
    agent that terminated early in an episode that was later truncated reports
    ``terminations[agent] = True`` and ``truncations[agent] = False``. ``agents``
    is then cleared.

    Only ``Box`` observation spaces are supported; any other observation space
    raises a ``TypeError``. The observation and action spaces are not modified.

    :param env: The parallel environment to wrap.
    """

    def __init__(self, env: ParallelEnv[AgentID, np.ndarray, ActionType]):
        super().__init__(env)
        for agent in getattr(env, "possible_agents", []):
            _check_space(self.env.observation_space(agent), agent)
        self.agents: list[AgentID] = []
        self._terminated: dict[AgentID, bool] = {}
        self._truncated: dict[AgentID, bool] = {}

    def _add_agents(self, agents: list[AgentID]) -> None:
        for agent in agents:
            if agent not in self._terminated:
                _check_space(self.env.observation_space(agent), agent)
                self.agents.append(agent)
                self._terminated[agent] = False
                self._truncated[agent] = False

    @override
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentID, np.ndarray], dict[AgentID, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = []
        self._terminated = {}
        self._truncated = {}
        self._add_agents(self.env.agents)
        return observations, infos

    @override
    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, np.ndarray],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict[str, Any]],
    ]:
        active = set(self.env.agents)
        observations, rewards, terminations, truncations, infos = self.env.step(
            {agent: action for agent, action in actions.items() if agent in active}
        )
        self._add_agents(self.env.agents)

        for agent in self.agents:
            if agent in terminations or agent in truncations:
                self._terminated[agent] = bool(terminations.get(agent, False))
                self._truncated[agent] = bool(truncations.get(agent, False))

        episode_over = all(
            self._terminated[agent] or self._truncated[agent]
            for agent in self.env.agents
        )

        all_observations = {}
        all_rewards = {}
        all_infos = {}
        for agent in self.agents:
            if agent in observations:
                all_observations[agent] = observations[agent]
            else:
                space = _check_space(self.env.observation_space(agent), agent)
                all_observations[agent] = np.zeros(space.shape, dtype=space.dtype)
            all_rewards[agent] = rewards.get(agent, 0.0)
            all_infos[agent] = infos.get(agent, {})

        if episode_over:
            # An agent the environment dropped without a flag left for good as well
            all_terminations = {
                agent: self._terminated[agent] or not self._truncated[agent]
                for agent in self.agents
            }
            all_truncations = dict(self._truncated)
            self.agents = []
        else:
            all_terminations = dict.fromkeys(self.agents, False)
            all_truncations = dict.fromkeys(self.agents, False)

        return (
            all_observations,
            all_rewards,
            all_terminations,
            all_truncations,
            all_infos,
        )

    @override
    def __str__(self) -> str:
        return f"BlackDeathParallelV4<{self.env!s}>"
