---
title: "SB3: Parallel environment vectorization"
---

# SB3: Parallel environment vectorization

`pettingzoo.utils.sb3.SB3ParallelVecEnv` adapts one or more Parallel environments to Stable-Baselines3's [VecEnv API](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html). One SB3 policy controls every agent. The adapter lives in an optional integration module: it constructs an SB3 vector environment, rather than an AEC or Parallel wrapper. Importing PettingZoo itself does not require SB3.

## Environment Setup

The Multiwalker training and evaluation scripts use these dependencies:

```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/multiwalker/requirements.txt
   :language: text
```

The adapter requires the development version of PettingZoo until it is released. From a checkout containing this module, install the dependencies and then the local source:

```bash
python -m pip install -r tutorials/SB3/multiwalker/requirements.txt
python -m pip install -e ".[sisl]"
```

## Worlds and agent slots

Pass factories that each create a fresh Parallel environment. Worlds execute sequentially in the current process. Each agent is a vector slot, ordered by factory and then by `possible_agents`. For example, two three-agent Multiwalker worlds produce six slots:

```text
world 0 -> walker_0, walker_1, walker_2 -> slots 0, 1, 2
world 1 -> walker_0, walker_1, walker_2 -> slots 3, 4, 5
```

All agents must have equal observation and action spaces supported by SB3. The adapter preserves those spaces, including supported dictionary observations, and does no preprocessing. SB3's `num_envs` and timestep count refer to agent slots. Agents in one world interact and reset together; they are not independent copies of a single-agent task.

```python
from stable_baselines3 import PPO

from pettingzoo import make
from pettingzoo.utils.sb3 import SB3ParallelVecEnv

env = SB3ParallelVecEnv(
    [lambda: make("parallel", "sisl/multiwalker-v9", max_cycles=500) for _ in range(2)]
)
try:
    model = PPO("MlpPolicy", env, seed=0, device="cpu")
    model.learn(total_timesteps=12_288)
finally:
    env.close()
```

`seed(s)` schedules seed `s + world_index` for the next explicit `reset()`. It returns one seed per agent slot, repeating the world's seed. `set_options(dict)` schedules options for every world; a list must contain one dictionary per slot, with equal options for agents sharing a world. Automatic episode resets do not consume scheduled seeds or options. Attribute and method indices also select slots; setting an attribute or calling a method affects each selected world once.

## Episode endings and early agent departures

When all agents finish, the adapter resets their world on the same step. The returned observations belong to the next episode. Each agent's final observation is in `infos[i]["terminal_observation"]`. `infos[i]["TimeLimit.truncated"]` is true only for truncation without termination, so SB3 can bootstrap time limits. Reset information is separate, in `env.reset_infos`.

All possible agents must be present after reset. If some agents can leave before others, apply the native `BlackDeathParallelV4` wrapper inside each factory. It keeps their slots present with zero observations and rewards until the world ends. For KAZ with vector observations:

```python
from pettingzoo import make
from pettingzoo.utils.sb3 import SB3ParallelVecEnv
from pettingzoo.utils.wrappers import BlackDeathParallelV4


def make_world():
    return BlackDeathParallelV4(
        make("parallel", "butterfly/knights_archers_zombies-v11", obs_method="vector")
    )


env = SB3ParallelVecEnv([make_world, make_world])
```

Without this wrapper, early departures raise an error. Environments where agents appear after reset are unsupported.

## Replacing SuperSuit vector functions

| Previous workflow | Replacement in this integration |
| --- | --- |
| `pettingzoo_env_to_vec_env_v1(env)` | `SB3ParallelVecEnv([make_env])` for one fresh Parallel world |
| `concat_vec_envs_v1(env, n, num_cpus=1, base_class="stable_baselines3")` | `SB3ParallelVecEnv([make_env for _ in range(n)])` |
| `stable_baselines3_vec_env_v0(vec)` | No extra adapter: the result already implements SB3's VecEnv API |

The maintained examples call the first two functions. They do not call `stable_baselines3_vec_env_v0` directly. Multiwalker now uses this replacement and no longer imports or depends on SuperSuit.

This is a first part of [the vectorization migration](https://github.com/Farama-Foundation/PettingZoo/issues/1480). The KAZ training script still uses SuperSuit for visual preprocessing and its existing vectorization path. Pistonball also uses subprocess vectorization, which this sequential adapter does not provide. CleanRL expects Gymnasium's vector API, which differs from SB3's. Those workflows and their dependencies still need separate migrations before SuperSuit can be retired.

## Code

The complete Multiwalker script trains with the environment's default rewards and can render a saved model. Its evaluation script reports package displacement to distinguish forward movement from a policy that merely stands still. A short training run checks the integration; it does not establish policy quality.

```{eval-rst}
.. literalinclude:: ../../../tutorials/SB3/multiwalker/train_multiwalker_policy.py
   :language: python
```
