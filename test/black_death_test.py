from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from pettingzoo.sisl import multiwalker_v9
from pettingzoo.test import parallel_api_test
from pettingzoo.test.example_envs import generated_agents_parallel_v0
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import BlackDeathParallelV4

AGENTS = ["agent_0", "agent_1", "agent_2"]

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
IMAGE_SPACE = Box(low=0, high=255, shape=(2, 2, 3), dtype=np.uint8)
ACTION_SPACE = Discrete(3)


class ScriptedParallel(ParallelEnv):
    """Agents leave on the step that ``leaves`` names, the way it says.

    ``leaves`` maps an agent to ``(step, "terminate" | "truncate")``. ``joins``
    maps an agent to the step after which it enters the environment.
    """

    metadata = {"render_modes": [], "name": "scripted_parallel"}

    def __init__(self, leaves, joins=None, observation_spaces=None):
        super().__init__()
        self.leaves = dict(leaves)
        self.joins = dict(joins or {})
        self.possible_agents = list(AGENTS)
        self._observation_spaces = observation_spaces or {}
        self.received = []
        self.reset_calls = []

    def observation_space(self, agent):
        return self._observation_spaces.get(agent, OBS_SPACE)

    def action_space(self, agent):
        return ACTION_SPACE

    def _observe(self, agent):
        space = self.observation_space(agent)
        return np.full(space.shape, 1, dtype=space.dtype)

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))
        self.agents = [a for a in self.possible_agents if a not in self.joins]
        self.received = []
        self._step = 0
        return (
            {a: self._observe(a) for a in self.agents},
            {a: {"step": 0} for a in self.agents},
        )

    def step(self, actions):
        self.received.append(dict(actions))
        self._step += 1
        acting = list(self.agents)
        leaving = {
            a: self.leaves[a][1] for a in acting if self.leaves[a][0] == self._step
        }
        observations = {a: self._observe(a) for a in acting}
        rewards = dict.fromkeys(acting, 1.0)
        terminations = {a: leaving.get(a) == "terminate" for a in acting}
        truncations = {a: leaving.get(a) == "truncate" for a in acting}
        infos = {a: {"step": self._step} for a in acting}
        self.agents = [a for a in acting if a not in leaving]
        for agent, step in self.joins.items():
            if step == self._step:
                self.agents.append(agent)
                observations[agent] = self._observe(agent)
                rewards[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {"step": self._step}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        pass

    def state(self):
        return np.zeros(1, dtype=np.float32)


def all_actions(env):
    return dict.fromkeys(env.possible_agents, 1)


def test_departed_agent_stays_with_zero_observation_and_reward():
    env = BlackDeathParallelV4(
        ScriptedParallel(
            {
                "agent_0": (1, "terminate"),
                "agent_1": (5, "terminate"),
                "agent_2": (5, "terminate"),
            },
            observation_spaces={"agent_0": IMAGE_SPACE},
        )
    )
    observations, infos = env.reset()
    assert env.agents == AGENTS
    assert set(observations) == set(infos) == set(AGENTS)

    # The step on which agent_0 leaves still carries what the environment returned for it
    observations, rewards, terminations, truncations, infos = env.step(all_actions(env))
    np.testing.assert_array_equal(observations["agent_0"], np.ones((2, 2, 3)))
    assert rewards["agent_0"] == 1.0
    assert infos["agent_0"] == {"step": 1}
    assert terminations == dict.fromkeys(AGENTS, False)
    assert truncations == dict.fromkeys(AGENTS, False)
    assert env.agents == AGENTS
    assert env.unwrapped.agents == ["agent_1", "agent_2"]

    for step in (2, 3):
        observations, rewards, terminations, truncations, infos = env.step(
            all_actions(env)
        )
        assert env.agents == AGENTS
        for returned in (observations, rewards, terminations, truncations, infos):
            assert list(returned) == AGENTS
        np.testing.assert_array_equal(
            observations["agent_0"], np.zeros((2, 2, 3), dtype=np.uint8)
        )
        assert observations["agent_0"].dtype == np.uint8
        assert env.observation_space("agent_0").contains(observations["agent_0"])
        assert rewards["agent_0"] == 0.0
        assert infos["agent_0"] == {}
        np.testing.assert_array_equal(observations["agent_1"], np.ones(2))
        assert rewards["agent_1"] == 1.0
        assert infos["agent_1"] == {"step": step}
        assert terminations == dict.fromkeys(AGENTS, False)
        assert truncations == dict.fromkeys(AGENTS, False)


def test_actions_of_departed_agents_are_not_forwarded():
    inner = ScriptedParallel(
        {
            "agent_0": (1, "terminate"),
            "agent_1": (2, "truncate"),
            "agent_2": (4, "terminate"),
        }
    )
    env = BlackDeathParallelV4(inner)
    env.reset()
    for _ in range(3):
        env.step({"agent_0": 0, "agent_1": 1, "agent_2": 2})

    assert inner.received == [
        {"agent_0": 0, "agent_1": 1, "agent_2": 2},
        {"agent_1": 1, "agent_2": 2},
        {"agent_2": 2},
    ]


def test_episode_ending_by_termination():
    env = BlackDeathParallelV4(
        ScriptedParallel(
            {
                "agent_0": (1, "terminate"),
                "agent_1": (3, "terminate"),
                "agent_2": (3, "terminate"),
            }
        )
    )
    env.reset()
    env.step(all_actions(env))
    env.step(all_actions(env))
    observations, rewards, terminations, truncations, infos = env.step(all_actions(env))

    assert terminations == dict.fromkeys(AGENTS, True)
    assert truncations == dict.fromkeys(AGENTS, False)
    np.testing.assert_array_equal(observations["agent_0"], np.zeros(2))
    assert rewards == {"agent_0": 0.0, "agent_1": 1.0, "agent_2": 1.0}
    assert infos == {"agent_0": {}, "agent_1": {"step": 3}, "agent_2": {"step": 3}}
    assert env.agents == []


def test_episode_ending_by_truncation_keeps_how_each_agent_left():
    env = BlackDeathParallelV4(
        ScriptedParallel(
            {
                "agent_0": (1, "terminate"),
                "agent_1": (2, "truncate"),
                "agent_2": (3, "truncate"),
            }
        )
    )
    env.reset()
    for _ in range(2):
        *_, terminations, truncations, _ = env.step(all_actions(env))
        assert terminations == dict.fromkeys(AGENTS, False)
        assert truncations == dict.fromkeys(AGENTS, False)
        assert env.agents == AGENTS
    *_, terminations, truncations, _ = env.step(all_actions(env))

    assert terminations == {"agent_0": True, "agent_1": False, "agent_2": False}
    assert truncations == {"agent_0": False, "agent_1": True, "agent_2": True}
    assert env.agents == []


def test_reset_starts_a_fresh_agent_set():
    inner = ScriptedParallel(
        {
            "agent_0": (1, "terminate"),
            "agent_1": (2, "terminate"),
            "agent_2": (2, "terminate"),
        }
    )
    env = BlackDeathParallelV4(inner)
    env.reset()
    env.step(all_actions(env))
    env.step(all_actions(env))
    assert env.agents == []

    inner.leaves = dict.fromkeys(AGENTS, (2, "truncate"))
    observations, infos = env.reset(seed=7, options={"key": "value"})
    assert inner.reset_calls[-1] == (7, {"key": "value"})
    assert env.agents == AGENTS
    assert set(observations) == set(infos) == set(AGENTS)

    observations, rewards, *_ = env.step(all_actions(env))
    np.testing.assert_array_equal(observations["agent_0"], np.ones(2))
    assert rewards["agent_0"] == 1.0
    *_, terminations, truncations, _ = env.step(all_actions(env))
    assert terminations == dict.fromkeys(AGENTS, False)
    assert truncations == dict.fromkeys(AGENTS, True)


def test_agent_that_joins_later_is_kept_too():
    env = BlackDeathParallelV4(
        ScriptedParallel(
            {
                "agent_0": (4, "terminate"),
                "agent_1": (4, "terminate"),
                "agent_2": (2, "terminate"),
            },
            joins={"agent_2": 1},
        )
    )
    observations, _ = env.reset()
    assert env.agents == ["agent_0", "agent_1"]
    assert "agent_2" not in observations

    observations, *_ = env.step(all_actions(env))
    assert env.agents == AGENTS
    np.testing.assert_array_equal(observations["agent_2"], np.ones(2))

    env.step(all_actions(env))
    observations, rewards, terminations, truncations, infos = env.step(all_actions(env))
    assert env.agents == AGENTS
    np.testing.assert_array_equal(observations["agent_2"], np.zeros(2))
    assert rewards["agent_2"] == 0.0
    assert not terminations["agent_2"] and not truncations["agent_2"]


@pytest.mark.parametrize(
    "space", [Discrete(3), Dict({"observation": OBS_SPACE})], ids=["discrete", "dict"]
)
def test_rejects_observation_spaces_that_are_not_box(space):
    inner = ScriptedParallel(
        dict.fromkeys(AGENTS, (2, "terminate")), observation_spaces={"agent_1": space}
    )
    with pytest.raises(TypeError, match="Box observation spaces"):
        BlackDeathParallelV4(inner)


def test_str():
    inner = ScriptedParallel(dict.fromkeys(AGENTS, (2, "terminate")))
    assert str(BlackDeathParallelV4(inner)) == f"BlackDeathParallelV4<{inner!s}>"


def test_parallel_api_scripted():
    env = BlackDeathParallelV4(
        ScriptedParallel(
            {
                "agent_0": (2, "terminate"),
                "agent_1": (4, "truncate"),
                "agent_2": (6, "terminate"),
            }
        )
    )
    parallel_api_test(env, num_cycles=20)


def test_parallel_api_agents_come_and_go():
    parallel_api_test(
        BlackDeathParallelV4(generated_agents_parallel_v0.parallel_env()),
        num_cycles=200,
    )


def test_parallel_api_multiwalker():
    parallel_api_test(
        BlackDeathParallelV4(multiwalker_v9.parallel_env(terminate_on_fall=False)),
        num_cycles=300,
    )
