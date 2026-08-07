#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Input noise uncertainty analysis for three DQN models.

This script is intentionally self-contained:
- It does not import helper functions from sibling project files.
- It re-implements environment loading, model loading, noise sampling,
  Monte Carlo simulation, raw-data saving, and plotting inside this file.

Noise rule used here:
    noisy_state_t = X_t * state_t
where each element of X_t is sampled independently from
U(1 - delta, 1 + delta) for every state dimension at every timestep.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"

import numpy as np
import tensorflow as tf
import yaml
from pyswmm import Links, Nodes, RainGages, Simulation, SystemStats
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import TIMESERIES
from swmm_api.input_file.sections.others import TimeseriesData
from tensorflow import keras
from tensorflow.keras import layers


SCRIPT_TYPE = "input_noise"
MODEL_SPECS = [
    {"key": "reward5_gci", "label": "DQN reward5 GCI", "model_dir": "Results_DQN_reward5/model"},
    {"key": "reward3_base", "label": "DQN reward3 Base", "model_dir": "Results_DQN_reward3/model"},
    {"key": "reward3_gienv", "label": "DQN reward3 GIenv", "model_dir": "Results_DQN_reward3_train_GI/model"},
]
MODEL_DIR = None
ACTIVE_MODEL_KEY = None
ACTIVE_MODEL_LABEL = None
SWMM_INP = "SWMM_GR/chaohu_noHC"
STATE_CONFIG = "states_yaml/chaohu"
ACTION_TABLE = "SWMM_GR/action_table.csv"
TRAINING_RAIN_PATH = "rainfall/training_raindata.npy"
REAL_RAIN_PATH = "rainfall/real_raindata.npy"
REAL_RAIN_SCALE = 5.0

NOISE_MEAN = 1.0
DELTAS = [0.05, 0.10]
MONTE_CARLO_RUNS = 50
ADVANCE_SECONDS = 300
STATE_CLIP_MIN = 0.0
ACTION_CLIP_MIN = 0.0
ACTION_CLIP_MAX = 1.0
BASE_SEED = 20260714

# Quick-test switch:
# - False: run all configured rainfall events and all repeats.
# - True: run only selected events and selected repeat indices.
QUICK_TEST_MODE = False
QUICK_TEST_EVENT_LABELS = None
QUICK_TEST_EVENT_INDICES = None
QUICK_TEST_REPEAT_INDICES = None

# Easy-to-edit event list:
# source = "design" reads TRAINING_RAIN_PATH using rain_id
# source = "real" reads REAL_RAIN_PATH using rain_id and rescales intensity by REAL_RAIN_SCALE
RAIN_EVENT_SPECS = [
    # These four design storms are the same events plotted in Drawing_control_performance_all_DRL.ipynb.
    # Their actual baseline/result keys remain rainfall70, rainfall64, rainfall67, rainfall51,
    # but the process-plot notebook renumbers rainfall50~79 to Design Rainfall 1~30 by subtracting 49.
    {"source": "design", "rain_id": 70, "label": "design_21", "title": "Design Rainfall 21"},
    {"source": "design", "rain_id": 64, "label": "design_15", "title": "Design Rainfall 15"},
    {"source": "design", "rain_id": 67, "label": "design_18", "title": "Design Rainfall 18"},
    {"source": "design", "rain_id": 51, "label": "design_2", "title": "Design Rainfall 2"},
    {"source": "real", "rain_id": 0, "label": "real_01", "title": "Real Rainfall 1"},
    {"source": "real", "rain_id": 1, "label": "real_02", "title": "Real Rainfall 2"},
    {"source": "real", "rain_id": 2, "label": "real_03", "title": "Real Rainfall 3"},
    {"source": "real", "rain_id": 3, "label": "real_04", "title": "Real Rainfall 4"},
]


def discover_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "states_yaml").exists() and (candidate / "SWMM_GR").exists():
            return candidate
    raise FileNotFoundError("Project root not found from script location.")


PROJECT_ROOT = discover_project_root()
RESULTS_ROOT = PROJECT_ROOT / "uncertainty_analysis_results_three_dqn"
RAW_DIR = None
TEMP_DIR = None


def configure_model(model_spec: dict) -> None:
    global MODEL_DIR, ACTIVE_MODEL_KEY, ACTIVE_MODEL_LABEL, RAW_DIR, TEMP_DIR

    MODEL_DIR = model_spec["model_dir"]
    ACTIVE_MODEL_KEY = model_spec["key"]
    ACTIVE_MODEL_LABEL = model_spec["label"]
    RAW_DIR = RESULTS_ROOT / ACTIVE_MODEL_KEY / SCRIPT_TYPE / "raw_data"
    TEMP_DIR = RESULTS_ROOT / ACTIVE_MODEL_KEY / SCRIPT_TYPE / "temp_swmm"


def ensure_directories() -> None:
    for path in [RAW_DIR, TEMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


class DQNAgent:
    def __init__(self, params: dict):
        self.params = params
        self.action_table = params["action_table"]

        observation_input = keras.Input(
            shape=(self.params["state_dim"],), dtype=tf.float32, name="state_input"
        )
        q_values = mlp(
            observation_input,
            self.params["encoding_layer"]
            + self.params["value_layer"]
            + [self.params["action_dim"]],
            tf.tanh,
            None,
        )
        self.model = keras.Model(inputs=observation_input, outputs=q_values)
        self.target_model = keras.Model(inputs=observation_input, outputs=q_values)

    def load_model(self, model_dir: Path) -> None:
        self.model.load_weights(str(model_dir / "dqn.h5"))
        self.target_model.load_weights(str(model_dir / "target_dqn.h5"))

    def select_action(self, observation: np.ndarray) -> tuple[int, np.ndarray]:
        q_values = self.model(observation, training=False)
        action_index = int(tf.argmax(q_values, axis=1)[0].numpy())
        action_vector = self.action_table[action_index].astype(np.float64).copy()
        return action_index, action_vector


def build_agent(action_table: np.ndarray, state_dim: int, action_dim: int) -> DQNAgent:
    agent_params = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "encoding_layer": [50, 50, 50],
        "value_layer": [50, 50, 50],
        "advantage_layer": [50, 50, 50],
        "num_rain": 50,
        "train_iterations": 20,
        "training_step": 800,
        "gamma": 0.01,
        "epsilon": 0.1,
        "ep_min": 1e-50,
        "ep_decay": 0.9,
        "learning_rate": 0.0001,
        "action_table": action_table,
    }
    agent = DQNAgent(agent_params)
    agent.load_model(PROJECT_ROOT / MODEL_DIR)
    return agent


def load_action_table() -> np.ndarray:
    data = np.loadtxt(PROJECT_ROOT / ACTION_TABLE, delimiter=",", skiprows=1)
    return data[:, 1:]


def load_yaml_config() -> dict:
    with open(PROJECT_ROOT / f"{STATE_CONFIG}.yaml", "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def load_selected_rain_events() -> list[dict]:
    design_raindata = np.load(PROJECT_ROOT / TRAINING_RAIN_PATH, allow_pickle=True)
    real_raindata = np.load(PROJECT_ROOT / REAL_RAIN_PATH, allow_pickle=True)

    events = []
    for spec in RAIN_EVENT_SPECS:
        if spec["source"] == "design":
            rain_series = design_raindata[spec["rain_id"]].tolist()
        elif spec["source"] == "real":
            rain_series = scale_real_rainfall(real_raindata[spec["rain_id"]].tolist(), REAL_RAIN_SCALE)
        else:
            raise ValueError(f"Unsupported rainfall source: {spec['source']}")

        events.append(
            {
                "source": spec["source"],
                "rain_id": spec["rain_id"],
                "label": spec["label"],
                "title": spec["title"],
                "series": rain_series,
            }
        )
    return events


def resolve_event_selection(events: list[dict]) -> list[tuple[int, dict]]:
    indexed_events = list(enumerate(events))
    if not QUICK_TEST_MODE:
        return indexed_events

    if QUICK_TEST_EVENT_LABELS:
        requested = set(QUICK_TEST_EVENT_LABELS)
        selected = [(idx, event) for idx, event in indexed_events if event["label"] in requested]
    elif QUICK_TEST_EVENT_INDICES:
        requested = set(QUICK_TEST_EVENT_INDICES)
        selected = [(idx, event) for idx, event in indexed_events if idx in requested]
    else:
        selected = indexed_events[:2]

    if not selected:
        raise ValueError("Quick test mode selected no rainfall events. Check QUICK_TEST_EVENT_LABELS or QUICK_TEST_EVENT_INDICES.")

    return selected


def resolve_repeat_indices() -> list[int]:
    if not QUICK_TEST_MODE:
        return list(range(MONTE_CARLO_RUNS))

    if QUICK_TEST_REPEAT_INDICES is None:
        return [0, 1, 2]

    repeat_indices = sorted(set(QUICK_TEST_REPEAT_INDICES))
    for repeat_index in repeat_indices:
        if repeat_index < 0 or repeat_index >= MONTE_CARLO_RUNS:
            raise ValueError(f"Repeat index {repeat_index} is outside [0, {MONTE_CARLO_RUNS - 1}].")
    return repeat_indices


def scale_real_rainfall(rain_series: list[list[str]], factor: float) -> list[list[str]]:
    scaled = []
    for timestamp, value in rain_series:
        scaled.append([timestamp, str(float(value) * factor)])
    return scaled


def initialize_results_container() -> dict:
    return {
        "CSO": [0.0],
        "flooding": [0.0],
        "inflow": [0.0],
        "total_flooding_time": [0.0],
        "total_CSO_time": [0.0],
        "res": [0.0],
        "state": [],
        "action": [],
        "rewards": [],
    }


def get_step_results(results, nodes, links, rgs, system_stats, config, params):
    delta_flooding = 0.0
    delta_cso = 0.0
    cso_total = 0.0
    for target in config["reward_targets"]:
        if target[1] == "flooding":
            if target[0] == "system":
                delta_flooding += system_stats.routing_stats[target[1]] - results["flooding"][-1]
            else:
                delta_flooding += nodes[target[0]].statistics["flooding_volume"]
            results["flooding"].append(system_stats.routing_stats[target[1]])
        else:
            cso_total += nodes[target[0]].cumulative_inflow

    delta_cso = cso_total - results["CSO"][-1]
    results["CSO"].append(cso_total)

    wet_weather_inflow = system_stats.routing_stats["wet_weather_inflow"]
    delta_inflow = wet_weather_inflow - results["inflow"][-1]
    results["inflow"].append(wet_weather_inflow)

    flooding_duration = 0.0
    for node in nodes:
        flooding_duration = max(flooding_duration, node.statistics["flooding_duration"])
    results["total_flooding_time"].append(flooding_duration)

    delta_cso_time = params["advance_seconds"] / 3600 if delta_cso > 0 else 0.0
    results["total_CSO_time"].append(results["total_CSO_time"][-1] + delta_cso_time)

    if delta_inflow == 0:
        reward = 0.0
    else:
        reward = 1.0 / (1.0 + delta_flooding / delta_inflow + delta_cso / delta_inflow) - 1.0

    return results, reward


class StandaloneSWMMEnv:
    def __init__(self, params: dict):
        self.params = params
        self.config = load_yaml_config()
        self.sim = None
        self.results = None
        self.current_inp_path = None

    def reset(self, rain_series: list[list[str]], rain_label: str) -> list[float]:
        self.close()

        run_dir = TEMP_DIR / rain_label
        run_dir.mkdir(parents=True, exist_ok=True)
        self.current_inp_path = run_dir / f"{self.params['orf_save']}_{rain_label}.inp"

        inp = read_inp_file(str(PROJECT_ROOT / f"{self.params['orf']}.inp"))
        inp[TIMESERIES]["rainfall"] = TimeseriesData("rainfall", rain_series)
        inp.write_file(str(self.current_inp_path))

        self.sim = Simulation(str(self.current_inp_path))
        self.sim.start()

        if self.params["advance_seconds"] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params["advance_seconds"])

        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rain_gages = RainGages(self.sim)
        states = self._extract_states(nodes, links, rain_gages)
        self.results = initialize_results_container()
        return states

    def step(self, action: np.ndarray):
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rain_gages = RainGages(self.sim)
        system_stats = SystemStats(self.sim)

        states = self._extract_states(nodes, links, rain_gages)

        for asset, action_value in zip(self.config["action_assets"], action):
            links[asset].target_setting = float(action_value)

        try:
            if self.params["advance_seconds"] is None:
                current_time = self.sim._model.swmm_step()
            else:
                current_time = self.sim._model.swmm_stride(self.params["advance_seconds"])
            done = current_time <= 0
        except Exception:
            done = True

        self.results, reward = get_step_results(
            self.results, nodes, links, rain_gages, system_stats, self.config, self.params
        )
        self.results["state"].append(states)
        self.results["action"].append(np.asarray(action, dtype=np.float64).tolist())
        self.results["rewards"].append(reward)

        if done:
            self.close()

        return states, reward, self.results, done

    def close(self) -> None:
        if self.sim is not None:
            try:
                self.sim._model.swmm_end()
            except Exception:
                pass
            try:
                self.sim._model.swmm_close()
            except Exception:
                pass
        self.sim = None

    def _extract_states(self, nodes, links, rain_gages) -> list[float]:
        states = []
        for state_name, state_type in self.config["states"]:
            if state_type == "depthN":
                states.append(nodes[state_name].depth)
            elif state_type == "flow":
                states.append(links[state_name].flow)
            elif state_type == "inflow":
                states.append(nodes[state_name].total_inflow)
            else:
                states.append(rain_gages[state_name].rainfall)
        return states


def build_environment() -> StandaloneSWMMEnv:
    env_params = {
        "orf": SWMM_INP,
        "orf_save": "chaohu_uncertainty_input",
        "parm": STATE_CONFIG,
        "advance_seconds": ADVANCE_SECONDS,
        "kf": 1,
        "kc": 1,
        "reward_type": "3",
    }
    return StandaloneSWMMEnv(env_params)


def make_repeat_seed(event_index: int, delta: float, repeat_index: int) -> int:
    return BASE_SEED + event_index * 100000 + int(round(delta * 1000)) * 100 + repeat_index


def sample_multiplicative_noise(rng: np.random.Generator, vector_size: int, delta: float) -> np.ndarray:
    # Independent sampling for every dimension at every timestep.
    return rng.uniform(NOISE_MEAN - delta, NOISE_MEAN + delta, size=vector_size)


def sanitize_state(state: np.ndarray) -> np.ndarray:
    return np.clip(state, STATE_CLIP_MIN, None)


def clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(action, ACTION_CLIP_MIN, ACTION_CLIP_MAX)


def run_single_simulation(
    event: dict,
    event_index: int,
    delta: float,
    repeat_index: int,
    agent: DQNAgent,
) -> dict:
    seed = make_repeat_seed(event_index, delta, repeat_index)
    rng = np.random.default_rng(seed)

    # A fresh environment object is created for every Monte Carlo repeat to
    # guarantee full re-initialization and avoid any state carry-over.
    env = build_environment()
    run_label = f"{event['label']}_delta{int(delta * 1000):03d}_run{repeat_index:03d}"
    observation = env.reset(event["series"], run_label)

    noise_history = []
    clean_state_history = []
    noisy_state_history = []
    action_index_history = []
    executed_action_history = []

    done = False
    while not done:
        clean_state = np.asarray(observation, dtype=np.float64)
        noise_vector = sample_multiplicative_noise(rng, clean_state.size, delta)

        # Noise is re-sampled independently for each state dimension at each timestep.
        noisy_state = sanitize_state(clean_state * noise_vector)
        action_index, action_vector = agent.select_action(noisy_state.reshape(1, -1))
        executed_action = clip_action(action_vector)

        observation, reward, results, done = env.step(executed_action)

        clean_state_history.append(clean_state)
        noise_history.append(noise_vector)
        noisy_state_history.append(noisy_state)
        action_index_history.append(action_index)
        executed_action_history.append(executed_action.copy())

    final_flooding = float(results["flooding"][-1])
    final_cso = float(results["CSO"][-1])
    total_overflow = final_flooding + final_cso

    return {
        "seed": seed,
        "event": event,
        "delta": delta,
        "repeat_index": repeat_index,
        "noise_history": np.asarray(noise_history, dtype=np.float64),
        "clean_state_history": np.asarray(clean_state_history, dtype=np.float64),
        "noisy_state_history": np.asarray(noisy_state_history, dtype=np.float64),
        "action_index_history": np.asarray(action_index_history, dtype=np.int32),
        "executed_action_history": np.asarray(executed_action_history, dtype=np.float64),
        "num_steps": len(noise_history),
        "final_flooding": final_flooding,
        "final_cso": final_cso,
        "total_overflow": total_overflow,
    }


def save_run_artifacts(run_result: dict) -> None:
    event = run_result["event"]
    delta_tag = f"delta{int(run_result['delta'] * 1000):03d}"
    repeat_tag = f"run{run_result['repeat_index']:03d}"
    stem = f"{SCRIPT_TYPE}_{event['label']}_{delta_tag}_{repeat_tag}"

    np.savez_compressed(
        RAW_DIR / f"{stem}.npz",
        script_type=SCRIPT_TYPE,
        model_key=ACTIVE_MODEL_KEY,
        model_label=ACTIVE_MODEL_LABEL,
        model_dir=MODEL_DIR,
        source=event["source"],
        rain_id=event["rain_id"],
        event_label=event["label"],
        event_title=event["title"],
        delta=run_result["delta"],
        repeat_index=run_result["repeat_index"],
        seed=run_result["seed"],
        noise_factors=run_result["noise_history"],
        clean_states=run_result["clean_state_history"],
        noisy_states=run_result["noisy_state_history"],
        action_indices=run_result["action_index_history"],
        executed_actions=run_result["executed_action_history"],
        num_steps=run_result["num_steps"],
        final_flooding=run_result["final_flooding"],
        final_cso=run_result["final_cso"],
        total_overflow=run_result["total_overflow"],
    )


def write_summary_csv(summary_rows: list[dict]) -> None:
    csv_path = RAW_DIR / f"{SCRIPT_TYPE}_summary.csv"
    fieldnames = [
        "script_type",
        "model_key",
        "model_label",
        "model_dir",
        "source",
        "rain_id",
        "event_label",
        "event_title",
        "delta",
        "repeat_index",
        "seed",
        "num_steps",
        "final_flooding",
        "final_cso",
        "total_overflow",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def clear_temp_dir() -> None:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    events = load_selected_rain_events()
    selected_events = resolve_event_selection(events)
    repeat_indices = resolve_repeat_indices()

    print(f"Quick test mode: {QUICK_TEST_MODE}")
    print(f"Selected event labels: {[event['label'] for _, event in selected_events]}")
    print(f"Selected repeat indices: {repeat_indices}")

    for model_spec in MODEL_SPECS:
        configure_model(model_spec)
        ensure_directories()
        clear_temp_dir()
        tf.keras.backend.clear_session()

        action_table = load_action_table()
        config = load_yaml_config()
        agent = build_agent(
            action_table=action_table,
            state_dim=len(config["states"]),
            action_dim=2 ** len(config["action_assets"]),
        )
        summary_rows = []

        print(f"\n{'=' * 80}")
        print(f"Model: {ACTIVE_MODEL_LABEL} ({ACTIVE_MODEL_KEY})")
        print(f"Model dir: {MODEL_DIR}")

        for event_index, event in selected_events:
            print(f"\nProcessing event: {event['title']} ({event['label']})")
            for delta in DELTAS:
                print(f"  Delta = {delta:.2f}")
                for run_counter, repeat_index in enumerate(repeat_indices, start=1):
                    print(f"    Run {run_counter}/{len(repeat_indices)} (repeat_index={repeat_index})")
                    run_result = run_single_simulation(event, event_index, delta, repeat_index, agent)
                    save_run_artifacts(run_result)
                    summary_rows.append(
                        {
                            "script_type": SCRIPT_TYPE,
                            "model_key": ACTIVE_MODEL_KEY,
                            "model_label": ACTIVE_MODEL_LABEL,
                            "model_dir": MODEL_DIR,
                            "source": event["source"],
                            "rain_id": event["rain_id"],
                            "event_label": event["label"],
                            "event_title": event["title"],
                            "delta": delta,
                            "repeat_index": repeat_index,
                            "seed": run_result["seed"],
                            "num_steps": run_result["num_steps"],
                            "final_flooding": run_result["final_flooding"],
                            "final_cso": run_result["final_cso"],
                            "total_overflow": run_result["total_overflow"],
                        }
                    )

        write_summary_csv(summary_rows)
        clear_temp_dir()

    print("\nInput noise analysis for three DQN models finished. Plotting is handled separately in the control notebook.")


if __name__ == "__main__":
    main()
