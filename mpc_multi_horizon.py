import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pyswmm import Links, Nodes, RainGages, Simulation, SystemStats
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import CONTROLS, TIMESERIES
from swmm_api.input_file.sections import Control
from swmm_api.input_file.sections.others import TimeseriesData

from SWMM import SWMM_ENV


ROOT = Path(__file__).resolve().parent
START_TIME = datetime.datetime(2015, 8, 28, 8, 0, 0)
RULES = """
RULE R0
IF SIMULATION TIME > 0
THEN PUMP CC-R1 SETTING = TIMESERIES pump0

RULE R1
IF SIMULATION TIME > 0
THEN PUMP CC-R2 SETTING = TIMESERIES pump1

RULE R2
IF SIMULATION TIME > 0
THEN PUMP CC-S1 SETTING = TIMESERIES pump2

RULE R3
IF SIMULATION TIME > 0
THEN PUMP CC-S2 SETTING = TIMESERIES pump3

RULE R4
IF SIMULATION TIME > 0
THEN PUMP JK-R1 SETTING = TIMESERIES pump4

RULE R5
IF SIMULATION TIME > 0
THEN PUMP JK-R2 SETTING = TIMESERIES pump5

RULE R6
IF SIMULATION TIME > 0
THEN PUMP JK-S SETTING = TIMESERIES pump6
"""


def default_env_params():
    return {
        "orf": "SWMM_GR\\chaohu_noHC",
        "orf_save": "chaohu_RTC",
        "parm": "states_yaml\\chaohu",
        "advance_seconds": 300,
        "kf": 1,
        "kc": 1,
        "reward_type": "3",
    }


def default_pop_params():
    return {
        "pop_size": 10,
        "max_value": 127,
        "optstep": 150,
        "simulation_steps": 95,
        "worker_count": 10,
    }


def load_context(env_params=None):
    env_params = default_env_params() if env_params is None else dict(env_params)
    config_path = ROOT / f"{env_params['parm']}.yaml"
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.load(fh, yaml.FullLoader)
    action_table = pd.read_csv(ROOT / "SWMM_GR" / "action_table.csv").values[:, 1:].tolist()
    raindata = np.load(ROOT / "rainfall" / "training_raindata.npy", allow_pickle=True).tolist()
    return env_params, config, action_table, raindata


def make_timestamp(step_index, advance_seconds):
    return (START_TIME + datetime.timedelta(seconds=step_index * advance_seconds)).strftime("%m/%d/%Y %H:%M:%S")


def trans_action(action_values, advance_seconds):
    return [[make_timestamp(step_index, advance_seconds), action_values[step_index]] for step_index in range(len(action_values))]


def ensure_temp_dirs(rain_ids):
    temp_root = ROOT / "SWMM_GR" / "_temopt"
    temp_root.mkdir(parents=True, exist_ok=True)
    for rain_id in rain_ids:
        (temp_root / f"tem_rain{rain_id}").mkdir(parents=True, exist_ok=True)


def build_action_history(action_count):
    return {f"pump{index}": [0] for index in range(action_count)}


def build_action_timeseries(action_history, advance_seconds):
    return {
        pump_name: trans_action(values, advance_seconds)
        for pump_name, values in action_history.items()
    }


def append_action(action_history, action_values):
    for action_index, action_value in enumerate(action_values):
        action_history[f"pump{action_index}"].append(float(action_value))


def action_history_to_matrix(action_history, simulation_steps):
    action_matrix = []
    for step in range(simulation_steps):
        action_matrix.append([float(action_history[f"pump{index}"][step]) for index in range(len(action_history))])
    return action_matrix


def write_controlled_inp(
    rain_id,
    rainfall_timeseries,
    action_history,
    advance_seconds,
    copy_count,
    base_inp_name="chaohu_noHC.inp",
):
    inp = read_inp_file(ROOT / "SWMM_GR" / base_inp_name)
    inp[TIMESERIES]["rainfall"] = TimeseriesData("rainfall", rainfall_timeseries)
    inp[CONTROLS] = Control.create_section(RULES)
    action_timeseries = build_action_timeseries(action_history, advance_seconds)
    for pump_name, series in action_timeseries.items():
        inp[TIMESERIES][pump_name] = TimeseriesData(pump_name, series)

    main_inp_path = ROOT / "SWMM_GR" / "_temopt" / f"chaohu_rain{rain_id}.inp"
    inp.write_file(main_inp_path)
    for worker_index in range(copy_count):
        worker_path = ROOT / "SWMM_GR" / "_temopt" / f"tem_rain{rain_id}" / f"tem{worker_index}_chaohu_rain{rain_id}.inp"
        inp.write_file(worker_path)
    return main_inp_path


def get_states(nodes, links, rain_gages, config):
    states = []
    for asset_name, state_type in config["states"]:
        if state_type == "depthN":
            states.append(nodes[asset_name].depth)
        elif state_type == "flow":
            states.append(links[asset_name].flow)
        elif state_type == "inflow":
            states.append(nodes[asset_name].total_inflow)
        else:
            states.append(rain_gages[asset_name].rainfall)
    return states


def init_result_payload(config):
    return {
        "CSO": [0],
        "flooding": [0],
        "inflow": [0],
        "total_flooding_time": [0],
        "total_CSO_time": [0],
        "res": [0],
        "state": [],
        "action": [],
        "action_index": [],
        "action_assets": config["action_assets"],
        "rewards": [],
        "combined_metric": [],
    }


def replay_results(main_inp_path, action_history_matrix, simulation_steps, env_params, config, action_table):
    results = init_result_payload(config)
    action_lookup = {tuple(map(float, action)): idx for idx, action in enumerate(action_table)}

    sim = Simulation(str(main_inp_path))
    sim.start()
    nodes = Nodes(sim)
    links = Links(sim)
    rain_gages = RainGages(sim)
    system_stats = SystemStats(sim)

    for step in range(simulation_steps):
        sim._model.swmm_stride(env_params["advance_seconds"])
        states = get_states(nodes, links, rain_gages, config)
        results, reward = SWMM_ENV.get_step_results(results, nodes, links, rain_gages, system_stats, config, env_params)

        current_action = action_history_matrix[step]
        results["state"].append(states)
        results["action"].append(current_action)
        results["action_index"].append(action_lookup.get(tuple(map(float, current_action))))
        results["rewards"].append(reward)
        results["combined_metric"].append(float(results["flooding"][-1] + results["CSO"][-1]))

    sim._model.swmm_end()
    sim._model.swmm_close()
    return results


class PSO:
    def __init__(
        self,
        population_size,
        max_steps,
        evaluate_sequences,
        control_dim,
        x_bound,
        warm_start_sequence=None,
        inertia=0.6,
        acceleration=0.5,
    ):
        self.population_size = population_size
        self.max_steps = max_steps
        self.evaluate_sequences = evaluate_sequences
        self.control_dim = control_dim
        self.x_bound = x_bound
        self.inertia = inertia
        self.acceleration = acceleration
        self.positions = np.random.rand(self.population_size, self.control_dim)
        if warm_start_sequence:
            warm = np.array(warm_start_sequence[: self.control_dim], dtype=float)
            if len(warm) < self.control_dim:
                warm = np.pad(warm, (0, self.control_dim - len(warm)), mode="edge")
            self.positions[0, :] = np.clip(warm, 0, 1)
        self.velocities = np.random.rand(self.population_size, self.control_dim)

        fitness = np.array(self.evaluate_sequences(self.decode_particles(self.positions)))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = fitness.copy()
        self.global_best_position = self.positions[np.argmax(fitness)].copy()
        self.global_best_fitness = float(np.max(fitness))

    def decode_particles(self, particles):
        scaled = np.clip(np.rint(particles * self.x_bound), 0, self.x_bound).astype(int)
        return [row.tolist() for row in scaled]

    def evolve(self):
        best_curve = []
        for step_index in range(self.max_steps):
            rand1 = np.random.rand(self.population_size, self.control_dim)
            rand2 = np.random.rand(self.population_size, self.control_dim)
            self.velocities = (
                self.inertia * self.velocities
                + self.acceleration * rand1 * (self.personal_best_positions - self.positions)
                + self.acceleration * rand2 * (self.global_best_position - self.positions)
            )
            self.positions = np.clip(self.positions + 0.01 * self.velocities, 0, 1)

            fitness = np.array(self.evaluate_sequences(self.decode_particles(self.positions)))
            update_mask = fitness > self.personal_best_fitness
            self.personal_best_positions[update_mask] = self.positions[update_mask]
            self.personal_best_fitness[update_mask] = fitness[update_mask]

            current_best_index = int(np.argmax(fitness))
            current_best_value = float(np.max(fitness))
            if current_best_value > self.global_best_fitness:
                self.global_best_position = self.positions[current_best_index].copy()
                self.global_best_fitness = current_best_value

            best_curve.append(self.global_best_fitness)
            if step_index > 21 and abs(np.sum(best_curve[-10:]) - np.sum(best_curve[-20:-10])) < 0.01:
                break


class MPCExperiment:
    def __init__(
        self,
        rain_id,
        rainfall_timeseries,
        prediction_horizon,
        env_params,
        pop_params,
        config,
        action_table,
        warm_start_reference=None,
        warm_start_path=None,
    ):
        self.rain_id = rain_id
        self.rainfall_timeseries = rainfall_timeseries
        self.prediction_horizon = prediction_horizon
        self.env_params = env_params
        self.pop_params = pop_params
        self.config = config
        self.action_table = action_table
        self.action_count = len(action_table[0])
        self.worker_count = min(pop_params["worker_count"], pop_params["pop_size"])
        self.copy_count = pop_params["pop_size"]
        self.warm_start_reference = warm_start_reference
        self.warm_start_path = warm_start_path
        self._warm_start_logged = False
        self._warm_start_warned = False
        self.action_history = build_action_history(self.action_count)
        self.main_inp_path = write_controlled_inp(
            rain_id=self.rain_id,
            rainfall_timeseries=self.rainfall_timeseries,
            action_history=self.action_history,
            advance_seconds=self.env_params["advance_seconds"],
            copy_count=self.copy_count,
        )
        self.hotstart_path = ROOT / "SWMM_GR" / "_temopt" / f"chaohu_rain{self.rain_id}_hotstart.hsf"

    def get_warm_start_sequence(self, step):
        if not self.warm_start_reference:
            if not self._warm_start_warned:
                print(
                    f"[WARNING] Warm start unavailable for rainfall {self.rain_id}. "
                    "Optimization is using random PSO initialization."
                )
                self._warm_start_warned = True
            return None
        rainfall_key = f"rainfall{self.rain_id}"
        rainfall_data = self.warm_start_reference.get(rainfall_key)
        if rainfall_data is None:
            if not self._warm_start_warned:
                print(
                    f"[WARNING] Warm start rainfall key missing: {rainfall_key} "
                    f"in {self.warm_start_path}. Optimization is using random initialization."
                )
                self._warm_start_warned = True
            return None

        action_history, source_key = extract_warm_start_action_history(rainfall_data)
        if not action_history:
            if not self._warm_start_warned:
                available_keys = list(rainfall_data.keys()) if isinstance(rainfall_data, dict) else []
                print(
                    f"[WARNING] Warm start action history not found for {rainfall_key} in {self.warm_start_path}. "
                    f"Available keys: {available_keys}. Expected rainfall->env3->action or rainfall->action. "
                    "Optimization is using random initialization."
                )
                self._warm_start_warned = True
            return None

        if not self._warm_start_logged:
            print(
                f"[INFO] Warm start is active for rainfall {self.rain_id} "
                f"from {self.warm_start_path} using source '{source_key}'."
            )
            self._warm_start_logged = True

        sequence = []
        for offset in range(self.prediction_horizon):
            action_step = min(step + offset, len(action_history) - 1)
            action_vector = action_history[action_step]
            action_index = 0
            for candidate_index, candidate_action in enumerate(self.action_table):
                if list(map(float, candidate_action)) == list(map(float, action_vector)):
                    action_index = candidate_index
                    break
            sequence.append(action_index / self.pop_params["max_value"])
        return sequence

    def evaluate_sequences(self, action_sequences, step):
        def evaluate_single(sequence, worker_index):
            trial_history = {key: values.copy() for key, values in self.action_history.items()}
            for action_index in sequence:
                append_action(trial_history, self.action_table[int(action_index)])

            worker_inp_path = ROOT / "SWMM_GR" / "_temopt" / f"tem_rain{self.rain_id}" / f"tem{worker_index}_chaohu_rain{self.rain_id}.inp"
            worker_inp = read_inp_file(worker_inp_path)
            action_timeseries = build_action_timeseries(trial_history, self.env_params["advance_seconds"])
            for pump_name, series in action_timeseries.items():
                worker_inp[TIMESERIES][pump_name] = TimeseriesData(pump_name, series)
            worker_inp.write_file(worker_inp_path)

            sim = Simulation(str(worker_inp_path))
            sim.use_hotstart(str(self.hotstart_path))
            sim.start_time = START_TIME + datetime.timedelta(seconds=step * self.env_params["advance_seconds"])
            sim.start()

            cumulative_reward = 0.0
            for _ in range(len(sequence)):
                sim._model.swmm_stride(self.env_params["advance_seconds"])
                _, reward, _ = self.step_results(sim)
                cumulative_reward += reward + 1

            sim._model.swmm_end()
            sim._model.swmm_close()
            return cumulative_reward

        return Parallel(n_jobs=self.worker_count)(
            delayed(evaluate_single)(action_sequences[index], index)
            for index in range(len(action_sequences))
        )

    def step_results(self, sim):
        nodes = Nodes(sim)
        links = Links(sim)
        rain_gages = RainGages(sim)
        system_stats = SystemStats(sim)
        states = get_states(nodes, links, rain_gages, self.config)
        results, reward = SWMM_ENV.get_step_results(
            init_result_payload(self.config),
            nodes,
            links,
            rain_gages,
            system_stats,
            self.config,
            self.env_params,
        )
        results["state"].append(states)
        results["rewards"].append(reward)
        return states, reward, results

    def initialize_hotstart(self):
        sim = Simulation(str(self.main_inp_path))
        sim.start()
        sim._model.swmm_stride(self.env_params["advance_seconds"])
        sim.save_hotstart(str(self.hotstart_path))
        sim._model.swmm_end()
        sim._model.swmm_close()

    def optimize_step(self, step):
        remaining_steps = self.pop_params["simulation_steps"] - step
        horizon = min(self.prediction_horizon, remaining_steps)
        warm_start_sequence = self.get_warm_start_sequence(step)
        pso = PSO(
            population_size=self.pop_params["pop_size"],
            max_steps=self.pop_params["optstep"],
            evaluate_sequences=lambda sequences: self.evaluate_sequences(sequences, step),
            control_dim=horizon,
            x_bound=self.pop_params["max_value"],
            warm_start_sequence=warm_start_sequence,
        )
        pso.evolve()
        best_sequence = np.clip(np.rint(pso.global_best_position * self.pop_params["max_value"]), 0, self.pop_params["max_value"]).astype(int).tolist()
        return best_sequence

    def update_control_file(self):
        write_controlled_inp(
            rain_id=self.rain_id,
            rainfall_timeseries=self.rainfall_timeseries,
            action_history=self.action_history,
            advance_seconds=self.env_params["advance_seconds"],
            copy_count=self.copy_count,
        )

    def advance_one_step(self, step):
        sim = Simulation(str(self.main_inp_path))
        sim.use_hotstart(str(self.hotstart_path))
        sim.start_time = START_TIME + datetime.timedelta(seconds=step * self.env_params["advance_seconds"])
        sim.start()
        sim._model.swmm_stride(self.env_params["advance_seconds"])
        sim.save_hotstart(str(self.hotstart_path))
        sim._model.swmm_end()
        sim._model.swmm_close()

    def run(self):
        self.initialize_hotstart()
        for step in range(1, self.pop_params["simulation_steps"]):
            best_sequence = self.optimize_step(step)
            best_action = self.action_table[int(best_sequence[0])]
            append_action(self.action_history, best_action)
            self.update_control_file()
            self.advance_one_step(step)

        action_history_matrix = action_history_to_matrix(self.action_history, self.pop_params["simulation_steps"])
        return replay_results(
            main_inp_path=self.main_inp_path,
            action_history_matrix=action_history_matrix,
            simulation_steps=self.pop_params["simulation_steps"],
            env_params=self.env_params,
            config=self.config,
            action_table=self.action_table,
        )


def load_warm_start_reference(warm_start_path):
    if not warm_start_path:
        print("[WARNING] warm_start_path is empty. PSO will run without warm start.")
        return None
    warm_path = ROOT / warm_start_path
    if not warm_path.exists():
        print(f"[WARNING] Warm start file not found: {warm_path}. PSO will run without warm start.")
        return None
    print(f"[INFO] Warm start file loaded: {warm_path}")
    return np.load(warm_path, allow_pickle=True).tolist()


def extract_warm_start_action_history(rainfall_data):
    if not isinstance(rainfall_data, dict):
        return None, None

    top_level_action = rainfall_data.get("action")
    if top_level_action:
        return top_level_action, "top_level"

    if "env3" in rainfall_data and isinstance(rainfall_data["env3"], dict):
        env3_action = rainfall_data["env3"].get("action")
        if env3_action:
            return env3_action, "env3"

    nested_candidates = []
    for key, value in rainfall_data.items():
        if isinstance(value, dict) and value.get("action"):
            nested_candidates.append((key, value["action"]))

    if len(nested_candidates) == 1:
        nested_key, nested_action = nested_candidates[0]
        return nested_action, nested_key

    return None, None


def run_single_horizon(
    prediction_horizon,
    rain_ids,
    result_path,
    env_params=None,
    pop_params=None,
    warm_start_path=None,
):
    env_params, config, action_table, raindata = load_context(env_params)
    pop_params = default_pop_params() if pop_params is None else dict(pop_params)
    warm_start_reference = load_warm_start_reference(warm_start_path)
    ensure_temp_dirs(rain_ids)

    batch_results = {}
    for rain_id in rain_ids:
        print(f"Running MPC for rainfall {rain_id} with horizon={prediction_horizon}")
        
        experiment = MPCExperiment(
            rain_id=rain_id,
            rainfall_timeseries=raindata[rain_id],
            prediction_horizon=prediction_horizon,
            env_params=env_params,
            pop_params=pop_params,
            config=config,
            action_table=action_table,
            warm_start_reference=warm_start_reference,
            warm_start_path=warm_start_path,
        )
        batch_results[f"rainfall{rain_id}"] = experiment.run()

    save_path = ROOT / result_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, batch_results)
    print(f"Saved horizon={prediction_horizon} results to: {save_path}")
    return batch_results


def run_horizon_batch_experiments(
    horizons,
    rain_start=50,
    num_rainfalls=50,
    result_dir="results_mpc_horizons",
    env_params=None,
    pop_params=None,
    warm_start_path=None,
):
    rain_ids = list(range(rain_start, rain_start + num_rainfalls))
    outputs = {}
    for horizon in horizons:
        result_path = Path(result_dir) / f"MPC_GI_h{horizon}_all_rainfalls.npy"
        outputs[horizon] = run_single_horizon(
            prediction_horizon=horizon,
            rain_ids=rain_ids,
            result_path=result_path,
            env_params=env_params,
            pop_params=pop_params,
            warm_start_path=warm_start_path,
        )
    return outputs


def load_result_file(result_path):
    return np.load(ROOT / result_path, allow_pickle=True).tolist()


def aggregate_metric_curve(result_dict, value_getter):
    rainfall_keys = sorted(result_dict.keys())
    stacked = np.array([value_getter(result_dict[rainfall_key]) for rainfall_key in rainfall_keys], dtype=float)
    return stacked.mean(axis=0)


def get_combined_metric_curve(rainfall_result):
    flooding = np.asarray(rainfall_result["flooding"][1:], dtype=float)
    cso = np.asarray(rainfall_result["CSO"][1:], dtype=float)
    return flooding + cso


def get_reward_curve(rainfall_result):
    return np.asarray(rainfall_result["rewards"], dtype=float)


def get_action_curve(rainfall_result):
    action_matrix = np.asarray(rainfall_result["action"], dtype=float)
    return action_matrix.sum(axis=1)


def plot_curve_group(curves_by_horizon, ylabel, title, output_path):
    plt.figure(figsize=(12, 6))
    for horizon, curve in curves_by_horizon.items():
        plt.plot(curve, label=f"h={horizon}", linewidth=1.8)
    plt.xlabel("Simulation step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_file = ROOT / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=160)
    plt.close()
    return output_file


def plot_horizon_comparisons(horizons, result_dir="results_mpc_horizons", plot_dir="results_mpc_horizons/plots"):
    loaded_results = {
        horizon: load_result_file(Path(result_dir) / f"MPC_GI_h{horizon}_all_rainfalls.npy")
        for horizon in horizons
    }

    combined_curves = {
        horizon: aggregate_metric_curve(result_dict, get_combined_metric_curve)
        for horizon, result_dict in loaded_results.items()
    }
    reward_curves = {
        horizon: aggregate_metric_curve(result_dict, get_reward_curve)
        for horizon, result_dict in loaded_results.items()
    }
    action_curves = {
        horizon: aggregate_metric_curve(result_dict, get_action_curve)
        for horizon, result_dict in loaded_results.items()
    }

    outputs = {
        "combined_metric": plot_curve_group(
            combined_curves,
            ylabel="Mean flooding + CSO",
            title="MPC comparison across all rainfalls: flooding + CSO",
            output_path=Path(plot_dir) / "mpc_horizon_compare_flooding_plus_cso.png",
        ),
        "reward": plot_curve_group(
            reward_curves,
            ylabel="Mean reward",
            title="MPC comparison across all rainfalls: reward",
            output_path=Path(plot_dir) / "mpc_horizon_compare_reward.png",
        ),
        "action": plot_curve_group(
            action_curves,
            ylabel="Mean action sum across 7 pumps",
            title="MPC comparison across all rainfalls: action curve",
            output_path=Path(plot_dir) / "mpc_horizon_compare_action.png",
        ),
    }
    return outputs
