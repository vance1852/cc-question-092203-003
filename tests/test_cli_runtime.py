"""运行入口与端到端回归测试。

覆盖：
- 非法输入在创建输出目录之前被拒绝（不留半成品目录）
- 命令行覆写的关键参数真实进入模型并记录到 results.json
- 台数扫描不破坏主流程对象，且可稳定重放
- 同一份配置（固定种子）重复运行结果一致（交互/批量可重放）
"""

import json
import sys

import numpy as np
import pytest

from wind_farm_opt.config import WindFarmConfig, create_sample_config
from wind_farm_opt.cli import WindFarmOptimizerCLI, main


def tiny_config(output_dir: str, **overrides) -> WindFarmConfig:
    config = create_sample_config()
    config.n_turbines = 3
    config.boundary_params = {
        "width": 1500, "height": 1500, "center_x": 0, "center_y": 0
    }
    config.optimization.population_size = 4
    config.optimization.max_iterations = 2
    config.visualization.save_plots = False
    config.visualization.plot_wake_heatmap = False
    config.visualization.save_dir = output_dir
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def run_cli(config: str) -> int:
    """通过真实命令行入口运行一份配置文件。"""
    argv_backup = sys.argv
    sys.argv = ["wind_farm_opt", "--config", config, "--no-plots"]
    try:
        return main()
    finally:
        sys.argv = argv_backup


# ----------------------------------------------------------------------
# 错误输入不得先创建半成品输出目录
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "cli_args",
    [
        ["--n-turbines", "1000"],            # 几何不可行
        ["--wake-decay", "0"],               # 衰减系数非法
        ["--wake-decay", "-0.2"],
        ["--discount-rate", "1.2"],          # 折现率非法
        ["--population", "1"],               # 算法规模非法
        ["--iterations", "0"],               # 迭代次数非法
        ["--min-spacing", "0"],              # 间距倍数非法
        ["--width", "10", "--height", "10"], # 边界过小放不下
    ],
)
def test_invalid_input_creates_no_output_dir(tmp_path, cli_args):
    out_dir = tmp_path / "should_not_exist"
    cfg_path = tmp_path / "cfg.json"
    tiny_config(str(out_dir)).to_json(str(cfg_path))

    argv_backup = sys.argv
    sys.argv = [
        "wind_farm_opt", "--config", str(cfg_path),
        "--no-plots", "--output-dir", str(out_dir),
        *cli_args,
    ]
    try:
        rc = main()
    finally:
        sys.argv = argv_backup

    assert rc != 0
    assert not out_dir.exists(), f"非法输入却创建了输出目录: {out_dir}"


def test_unknown_config_key_rejected_before_fs(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    tiny_config(str(tmp_path / "out")).to_json(str(cfg_path))
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw["wake_decay_typo"] = 0.5
    cfg_path.write_text(json.dumps(raw), encoding="utf-8")

    assert run_cli(str(cfg_path)) != 0
    assert not (tmp_path / "out").exists()


# ----------------------------------------------------------------------
# 覆写参数真实生效并写入结果文件
# ----------------------------------------------------------------------

def test_effective_parameters_recorded_in_results(tmp_path):
    out_dir = tmp_path / "run"
    cfg_path = tmp_path / "cfg.json"
    tiny_config(
        str(out_dir),
        wake_model="gaussian",
        wake_decay=0.035,
    ).to_json(str(cfg_path))

    argv_backup = sys.argv
    sys.argv = [
        "wind_farm_opt", "--config", str(cfg_path),
        "--no-plots",
        "--wake-decay", "0.091",
        "--discount-rate", "0.083",
        "--algorithm", "pso",
        "--population", "5",
        "--iterations", "2",
        "--min-spacing", "4.5",
    ]
    try:
        rc = main()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))

    effective = results["effective_config"]
    assert effective["wake_model"] == "gaussian"
    assert effective["wake_decay"] == pytest.approx(0.091)
    assert effective["economic"]["discount_rate"] == pytest.approx(0.083)
    assert effective["optimization"]["algorithm"] == "pso"
    assert effective["optimization"]["population_size"] == 5
    assert effective["optimization"]["max_iterations"] == 2
    assert effective["optimization"]["min_spacing_multiple"] == pytest.approx(4.5)

    # 扁平化快速查阅区同样记录最终值
    quick = results["config"]
    assert quick["wake_decay"] == pytest.approx(0.091)
    assert quick["discount_rate"] == pytest.approx(0.083)

    # 随结果保存的配置可再次加载且与生效配置等价
    saved_cfg = WindFarmConfig.from_json(str(out_dir / "config.json"))
    assert saved_cfg.to_dict() == effective


def test_runtime_wake_model_uses_effective_decay(tmp_path):
    config = tiny_config(str(tmp_path / "out"), wake_model="gaussian", wake_decay=0.077)
    cli = WindFarmOptimizerCLI(config)
    assert cli.wake_model.wake_decay == pytest.approx(0.077)


# ----------------------------------------------------------------------
# 台数扫描隔离性
# ----------------------------------------------------------------------

def test_sweep_does_not_mutate_main_run_objects(tmp_path):
    config = tiny_config(str(tmp_path / "out"))
    cli = WindFarmOptimizerCLI(config)
    cli.run_baseline()

    n_before = cli.config.n_turbines
    turbine_ids_before = [id(t) for t in cli.turbines]
    aep_n_before = len(cli.aep_calc.turbines)

    cli.run_turbine_sweep(min_turbines=2, max_turbines=4, step=2)

    # 主流程对象完全不受扫描影响
    assert cli.config.n_turbines == n_before
    assert [id(t) for t in cli.turbines] == turbine_ids_before
    assert len(cli.aep_calc.turbines) == aep_n_before
    assert cli.sweep_results["n_turbines"] == [2, 4]


def test_sweep_uses_configured_discount_rate(tmp_path):
    config = tiny_config(str(tmp_path / "out"))
    config.economic.discount_rate = 0.10
    cli = WindFarmOptimizerCLI(config)
    cli.run_turbine_sweep(min_turbines=3, max_turbines=3, step=1)
    assert cli.sweep_results["n_turbines"] == [3]
    assert len(cli.sweep_results["lcoe"]) == 1
    assert np.isfinite(cli.sweep_results["lcoe"][0])


def test_sweep_validates_arguments(tmp_path):
    config = tiny_config(str(tmp_path / "out"))
    cli = WindFarmOptimizerCLI(config)
    with pytest.raises(ValueError):
        cli.run_turbine_sweep(min_turbines=0, max_turbines=5)
    with pytest.raises(ValueError):
        cli.run_turbine_sweep(min_turbines=10, max_turbines=5)


# ----------------------------------------------------------------------
# 可重放性：固定种子下交互分析与批量运行结果一致
# ----------------------------------------------------------------------

def test_same_config_replays_identically(tmp_path):
    def baseline_once():
        config = tiny_config(str(tmp_path / "out"))
        cli = WindFarmOptimizerCLI(config)
        cli.run_baseline()
        return cli.baseline_positions.copy()

    first = baseline_once()
    second = baseline_once()
    np.testing.assert_array_equal(first, second)


def test_optimization_replays_identically(tmp_path):
    def optimize_once(algo):
        config = tiny_config(str(tmp_path / f"out_{algo}"))
        config.optimization.algorithm = algo
        cli = WindFarmOptimizerCLI(config)
        # 直接调用优化器，避免打印干扰断言
        if algo == "ga":
            from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
            opt = GeneticAlgorithm(
                n_turbines=config.n_turbines,
                rotor_diameters=cli.rotor_diameters,
                boundary=cli.boundary,
                fitness_fn=cli.aep_calc.evaluate_layout,
                config=GAConfig(
                    population_size=config.optimization.population_size,
                    max_generations=config.optimization.max_iterations,
                    min_spacing_multiple=config.optimization.min_spacing_multiple,
                    seed=config.optimization.seed,
                ),
            )
        else:
            from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
            opt = ParticleSwarmOptimizer(
                n_turbines=config.n_turbines,
                rotor_diameters=cli.rotor_diameters,
                boundary=cli.boundary,
                fitness_fn=cli.aep_calc.evaluate_layout,
                config=PSOConfig(
                    swarm_size=config.optimization.population_size,
                    max_iterations=config.optimization.max_iterations,
                    min_spacing_multiple=config.optimization.min_spacing_multiple,
                    seed=config.optimization.seed,
                ),
            )
        return opt.optimize(verbose=False).best_positions.copy()

    for algo in ("ga", "pso"):
        np.testing.assert_array_equal(optimize_once(algo), optimize_once(algo))


def test_turbine_position_isolation_during_use(tmp_path):
    """复现事故：给一台风机写位置不能影响其他机组。"""
    config = tiny_config(str(tmp_path / "out"))
    cli = WindFarmOptimizerCLI(config)
    cli.turbines[0].position = (10.0, 20.0)
    assert all(t.position is None for t in cli.turbines[1:])
