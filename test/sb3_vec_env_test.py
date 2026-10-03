from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from pettingzoo import make
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.sb3 import SB3ParallelVecEnv
from pettingzoo.utils.wrappers import BlackDeathParallelV4
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class World(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"]}
    render_mode = "rgb_array"

    def __init__(self, number=0, horizon=2, structured=False, agents=2):
        self.number = number
        self.horizon = horizon
        self.structured = structured
        self.possible_agents = [f"agent_{index}" for index in range(agents)]
        self.agents = []
        box = spaces.Box(-10000, 10000, shape=(4,), dtype=np.float32)
        self.obs_space = (
            spaces.Dict({"state": box, "id": spaces.Discrete(agents)})
            if structured
            else box
        )
        self.act_space = spaces.Discrete(3)
        self.buffers = {
            agent: np.zeros(4, dtype=np.float32) for agent in self.possible_agents
        }
        self.reset_calls = []
        self.joint_actions = []
        self.closes = 0
        self.method_calls = 0
        self.render_calls = 0
        self.step_info = {"nested": {"step": 0}}

    def observation_space(self, agent):
        return self.obs_space

    def action_space(self, agent):
        return self.act_space

    def observations(self):
        for index, agent in enumerate(self.possible_agents):
            self.buffers[agent][:] = (
                self.number,
                len(self.reset_calls),
                self.steps,
                index,
            )
        if self.structured:
            return {
                agent: {"state": self.buffers[agent], "id": index}
                for index, agent in reversed(list(enumerate(self.possible_agents)))
            }
        # Dict insertion order deliberately disagrees with possible_agents.
        return dict(reversed(list(self.buffers.items())))

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))
        self.agents = self.possible_agents[::-1]
        self.steps = 0
        self.step_info["nested"]["step"] = 0
        return self.observations(), {
            agent: {"reset": len(self.reset_calls)} for agent in self.agents
        }

    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.joint_actions.append(actions)
        self.steps += 1
        self.step_info["nested"]["step"] = self.steps
        finished = self.steps == self.horizon
        # In one world, agent_0 terminates and the others reach the time limit.
        terms = {
            agent: finished and index == 0
            for index, agent in enumerate(self.possible_agents)
        }
        truncs = {
            agent: finished and index != 0
            for index, agent in enumerate(self.possible_agents)
        }
        if finished:
            self.agents = []
        return (
            self.observations(),
            {
                agent: self.number * 10 + index + actions[agent]
                for index, agent in enumerate(self.possible_agents)
            },
            terms,
            truncs,
            dict.fromkeys(self.possible_agents, self.step_info),
        )

    def mark(self, increment=1):
        self.method_calls += increment
        return self.method_calls

    def render(self):
        self.render_calls += 1
        return np.full((2, 3, 3), self.number, dtype=np.uint8)

    def close(self):
        self.closes += 1


@pytest.mark.parametrize("structured", [False, True])
def test_slot_order_joint_steps_and_same_step_resets(structured):
    vec = SB3ParallelVecEnv(
        [partial(World, 4, 1, structured), partial(World, 9, 2, structured)]
    )
    try:
        assert vec.num_envs == 4
        obs = vec.reset()
        np.testing.assert_array_equal(
            obs["state"] if structured else obs,
            [[4, 1, 0, 0], [4, 1, 0, 1], [9, 1, 0, 0], [9, 1, 0, 1]],
        )
        if structured:
            np.testing.assert_array_equal(obs["id"], [0, 1, 0, 1])
        next_obs, rewards, dones, infos = vec.step(np.array([0, 2, 1, 0]))
        np.testing.assert_array_equal(rewards, [40, 43, 91, 91])
        np.testing.assert_array_equal(dones, [True, True, False, False])
        assert vec.envs[0].joint_actions == [{"agent_0": 0, "agent_1": 2}]
        assert vec.envs[1].joint_actions == [{"agent_0": 1, "agent_1": 0}]
        assert infos[0]["TimeLimit.truncated"] is False
        assert infos[1]["TimeLimit.truncated"] is True
        assert all(info["TimeLimit.truncated"] is False for info in infos[2:])
        terminal = infos[1]["terminal_observation"]
        np.testing.assert_array_equal(
            terminal["state"] if structured else terminal, [4, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            (next_obs["state"] if structured else next_obs)[1], [4, 2, 0, 1]
        )
        assert [info["reset"] for info in vec.reset_infos] == [2, 2, 1, 1]
        assert all(info["nested"]["step"] == 1 for info in infos)
        assert all("reset" not in info for info in infos)
        assert all("terminal_observation" not in info for info in infos[2:])
        vec.step(np.zeros(4, dtype=int))
        # Returned arrays, nested infos, and terminal observations survive reuse
        # of the environment's buffers on later steps and resets.
        np.testing.assert_array_equal(
            terminal["state"] if structured else terminal, [4, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            (next_obs["state"] if structured else next_obs)[1], [4, 2, 0, 1]
        )
        assert infos[1]["nested"]["step"] == 1
        assert [len(env.reset_calls) for env in vec.envs] == [3, 2]
    finally:
        vec.close()


def test_world_seed_options_and_automatic_reset_are_separate():
    vec = SB3ParallelVecEnv(
        [partial(World, horizon=1), partial(World, horizon=1, agents=3)]
    )
    try:
        assert vec.seed(42) == [42, 42, 43, 43, 43]
        options = [{"difficulty": [1]} for _ in range(2)] + [
            {"difficulty": [2]} for _ in range(3)
        ]
        vec.set_options(options)
        options[0]["difficulty"][0] = 99
        vec.reset()
        assert vec.envs[0].reset_calls == [(42, {"difficulty": [1]})]
        assert vec.envs[1].reset_calls == [(43, {"difficulty": [2]})]
        vec.seed(8)
        vec.set_options({"difficulty": [5]})
        vec.step(np.zeros(5, dtype=int))
        assert vec.envs[0].reset_calls[-1] == (None, None)
        vec.reset()
        assert vec.envs[0].reset_calls[-1] == (8, {"difficulty": [5]})
        assert vec.envs[1].reset_calls[-1] == (9, {"difficulty": [5]})
        vec.reset()
        assert vec.envs[0].reset_calls[-1] == (None, {})
        assert vec.envs[1].reset_calls[-1] == (None, {})
        random_seeds = vec.seed()
        assert random_seeds[0] == random_seeds[1]
        assert random_seeds[2] == random_seeds[0] + 1
        with pytest.raises(ValueError, match="must agree"):
            vec.set_options([{"value": i} for i in range(5)])
        vec.set_options([{"array": np.array([1, 2])} for _ in range(5)])
        vec.set_options(None)
        with pytest.raises(ValueError, match="one options dict per agent slot"):
            vec.set_options([{}] * 2)
    finally:
        vec.close()


def test_slot_indices_select_worlds_without_repeating_method_side_effects():
    vec = SB3ParallelVecEnv([partial(World, number=4), partial(World, number=9)])
    try:
        assert vec.get_attr("number", indices=[3, 0, 2]) == [9, 4, 9]
        assert vec.env_method("mark", increment=2, indices=[3, 0, 2]) == [2, 2, 2]
        assert [env.method_calls for env in vec.envs] == [2, 2]
        vec.set_attr("horizon", 7, indices=1)
        assert vec.get_attr("horizon") == [7, 7, 2, 2]
        assert vec.env_is_wrapped(BlackDeathParallelV4) == [False] * 4
        images = vec.get_images()
        assert [env.render_calls for env in vec.envs] == [1, 1]
        assert [int(image[0, 0, 0]) for image in images] == [4, 4, 9, 9]
    finally:
        vec.close()
        vec.close()
    assert [env.closes for env in vec.envs] == [1, 1]


def test_reset_cancels_a_pending_step():
    vec = SB3ParallelVecEnv([World])
    try:
        vec.reset()
        with pytest.raises(RuntimeError, match="step_async"):
            vec.step_wait()
        with pytest.raises(ValueError, match="Expected 2 actions"):
            vec.step_async(np.zeros(3))
        vec.step_async(np.zeros(2))
        with pytest.raises(RuntimeError, match="already pending"):
            vec.step_async(np.zeros(2))
        vec.reset()
        assert vec.envs[0].joint_actions == []
        with pytest.raises(RuntimeError, match="step_async"):
            vec.step_wait()
    finally:
        vec.close()


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate",
        "nonparallel",
        "noagents",
        "agentids",
        "observations",
        "actions",
        "factory",
    ],
)
def test_invalid_factories_close_created_worlds(kind):
    env = World()
    if kind == "empty":
        factories = []
    elif kind == "duplicate":
        factories = [lambda: env] * 2
    elif kind == "nonparallel":
        factories = [lambda: env, lambda: object()]
    elif kind == "noagents":
        env.possible_agents = []
        factories = [lambda: env]
    elif kind == "agentids":
        env.possible_agents = ["a", "a"]
        factories = [lambda: env]
    elif kind in ("observations", "actions"):
        other = World()
        if kind == "observations":
            other.obs_space = spaces.Box(0, 1, shape=(1,))
        else:
            other.act_space = spaces.Discrete(7)
        factories = [lambda: env, lambda: other]
    else:

        def failing_factory():
            raise RuntimeError("creation failed")

        factories = [lambda: env, failing_factory]
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        SB3ParallelVecEnv(factories)
    assert env.closes == (0 if kind == "empty" else 1)


class DepartingWorld(World):
    def step(self, actions):
        assert set(actions) == set(self.agents)
        self.steps += 1
        departed = self.agents.pop()
        agents = self.agents + [departed]
        return (
            {agent: np.ones(4, dtype=np.float32) for agent in agents},
            dict.fromkeys(agents, 1.0),
            {agent: agent == departed for agent in agents},
            dict.fromkeys(agents, False),
            {agent: {} for agent in agents},
        )


def test_early_departures_need_black_death_and_keep_the_original_end_cause():
    vec = SB3ParallelVecEnv([DepartingWorld])
    try:
        vec.reset()
        with pytest.raises(ValueError, match="BlackDeathParallelV4"):
            vec.step(np.zeros(2, dtype=int))
    finally:
        vec.close()

    vec = SB3ParallelVecEnv(
        [lambda: BaseParallelWrapper(BlackDeathParallelV4(DepartingWorld()))]
    )
    try:
        assert vec.env_is_wrapped(BlackDeathParallelV4) == [True, True]
        vec.reset()
        _, rewards, dones, _ = vec.step(np.zeros(2, dtype=int))
        assert rewards.tolist() == [1.0, 1.0]
        assert dones.tolist() == [False, False]
        _, rewards, dones, infos = vec.step(np.zeros(2, dtype=int))
        assert rewards.tolist() == [0.0, 1.0]
        assert dones.tolist() == [True, True]
        assert all(info["TimeLimit.truncated"] is False for info in infos)
        np.testing.assert_array_equal(infos[0]["terminal_observation"], np.zeros(4))
    finally:
        vec.close()


@pytest.mark.parametrize("when", ["reset", "step"])
def test_unannounced_agent_changes_are_rejected(when):
    class ChangingWorld(World):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            if when == "reset":
                self.agents.pop()
            return result

        def step(self, actions):
            result = super().step(actions)
            self.agents.append("newborn")
            return result

    vec = SB3ParallelVecEnv([ChangingWorld])
    try:
        with pytest.raises(ValueError, match="Agent sets must stay fixed"):
            vec.reset()
            vec.step(np.zeros(2, dtype=int))
    finally:
        vec.close()


@pytest.mark.parametrize("structured", [False, True])
def test_short_ppo_update_with_terminal_bootstrapping(structured):
    vec = SB3ParallelVecEnv([partial(World, structured=structured)])
    try:
        model = PPO(
            "MultiInputPolicy" if structured else "MlpPolicy",
            vec,
            n_steps=4,
            batch_size=8,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
            seed=0,
        )
        before = [parameter.detach().clone() for parameter in model.policy.parameters()]
        model.learn(total_timesteps=16)
        assert model.num_timesteps == 16
        assert any(
            not old.equal(new) for old, new in zip(before, model.policy.parameters())
        )
        assert np.isfinite(model.rollout_buffer.returns).all()
    finally:
        vec.close()


@pytest.mark.parametrize("environment", ["multiwalker", "kaz"])
def test_ppo_with_real_parallel_environments(environment):
    if environment == "multiwalker":
        pytest.importorskip("Box2D")
        factory = partial(make, "parallel", "sisl/multiwalker-v9", max_cycles=2)
        slots = 6
    else:
        pytest.importorskip("pygame")

        def factory():
            return BlackDeathParallelV4(
                make(
                    "parallel",
                    "butterfly/knights_archers_zombies-v11",
                    obs_method="vector",
                    max_cycles=2,
                )
            )

        slots = 8

    vec = SB3ParallelVecEnv([factory, factory])
    try:
        assert vec.num_envs == slots
        assert vec.get_images() == [None] * slots
        if environment == "multiwalker":
            # AEC-to-Parallel conversion does not forward custom raw attributes.
            assert vec.get_attr("agent_name_mapping", indices=4) == [
                {"walker_0": 0, "walker_1": 1, "walker_2": 2}
            ]
        vec.seed(42)
        first = vec.reset()
        vec.seed(42)
        np.testing.assert_array_equal(vec.reset(), first)
        for _ in range(2):
            _, _, dones, infos = vec.step(
                np.asarray([vec.action_space.sample() for _ in range(slots)])
            )
        assert dones.all()
        assert all("terminal_observation" in info for info in infos)
        model = PPO(
            "MlpPolicy",
            vec,
            n_steps=4,
            batch_size=slots * 4,
            n_epochs=1,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
            seed=0,
        )
        model.learn(total_timesteps=slots * 4)
        assert np.isfinite(model.rollout_buffer.returns).all()
    finally:
        vec.close()
