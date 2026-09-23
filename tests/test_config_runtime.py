"""回归测试：配置 → 运行对象的一对一语义、参数校验、结果可复现。"""

import json
import os

import numpy as np
import pytest

from wind_farm_opt.config import (
    ConfigError,
    OptimizationConfig,
    WindFarmConfig,
    create_sample_config,
)
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.optimization.baseline import generate_grid_layout


def make_small_config(**overrides) -> WindFarmConfig:
    """小规模配置，保证测试快速且确定。"""
    config = WindFarmConfig(
        n_turbines=6,
        wake_model="gaussian",
        wake_decay=0.035,
        boundary_type="rectangular",
        boundary_params={"width": 3000, "height": 3000, "center_x": 0, "center_y": 0},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=8,
            max_iterations=2,
            min_spacing_multiple=5.0,
            seed=42,
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# 尾流衰减系数必须真实进入模型
# ---------------------------------------------------------------------------

def test_gaussian_wake_decay_used():
    config = make_small_config(wake_model="gaussian", wake_decay=0.05)
    model = config.create_wake_model()
    assert type(model).__name__ == "GaussianWake"
    assert model.wake_decay == pytest.approx(0.05)


def test_gaussian_wake_decay_changes_deficit():
    d0, ct, dist = 126.0, 0.82, 5.0 * 126.0
    deficits = []
    for decay in (0.02, 0.05, 0.09):
        model = make_small_config(wake_decay=decay).create_wake_model()
        deficits.append(model.velocity_deficit(dist, d0, ct))
    # 系数不同，亏损必须不同，且更大的衰减系数对应更小的中心亏损。
    assert deficits[0] != deficits[1] != deficits[2]
    assert deficits[0] > deficits[1] > deficits[2]


def test_wake_decay_loaded_from_json(tmp_path):
    path = tmp_path / "cfg.json"
    data = {"wake_model": "gaussian", "wake_decay": 0.061}
    path.write_text(json.dumps(data), encoding="utf-8")
    config = WindFarmConfig.from_json(str(path))
    assert config.create_wake_model().wake_decay == pytest.approx(0.061)


# ---------------------------------------------------------------------------
# 每台风机必须是独立实例
# ---------------------------------------------------------------------------

def test_turbines_are_independent_instances():
    config = make_small_config()
    turbines = config.create_turbines()
    assert len(turbines) == config.n_turbines
    assert len(set(map(id, turbines))) == config.n_turbines

    turbines[0].position = (123.0, 456.0)
    for other in turbines[1:]:
        assert other.position is None

    turbines[1].thrust_coefficient  # 只读，确保数组彼此独立
    turbines[0].position = (1.0, 2.0)
    assert turbines[2].position is None


def test_repeated_create_turbines_returns_fresh_objects():
    config = make_small_config()
    first = config.create_turbines()
    second = config.create_turbines()
    assert not any(a is b for a, b in zip(first, second))


def test_power_curves_not_shared_between_turbines():
    config = make_small_config()
    turbines = config.create_turbines()
    # 底层 ndarray 也必须是独立副本，防止一处改写污染全部。
    assert not np.shares_memory(turbines[0].power_curve, turbines[1].power_curve)


# ---------------------------------------------------------------------------
# 非法参数必须尽早报明确错误
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c, "n_turbines", 0),
        lambda c: setattr(c, "n_turbines", -3),
        lambda c: setattr(c, "n_turbines", 2.5),
        lambda c: setattr(c, "wake_decay", 0.0),
        lambda c: setattr(c, "wake_decay", -0.01),
        lambda c: setattr(c, "wake_model", "eddy"),
        lambda c: setattr(c, "superposition_method", "quadratic"),
        lambda c: c.boundary_params.update(width=0.0),
        lambda c: c.boundary_params.update(height=-10.0),
        lambda c: setattr(c.optimization, "population_size", 1),
        lambda c: setattr(c.optimization, "max_iterations", 0),
        lambda c: setattr(c.optimization, "min_spacing_multiple", 0.0),
        lambda c: setattr(c.economic, "discount_rate", 1.0),
        lambda c: setattr(c.economic, "discount_rate", -0.05),
        lambda c: setattr(c.economic, "electricity_price", 0.0),
        lambda c: setattr(c.optimization, "algorithm", "annealing"),
        lambda c: setattr(c, "turbine_model", "V999-99MW"),
    ],
)
def test_invalid_configs_raise(mutate):
    config = make_small_config()
    mutate(config)
    with pytest.raises(ConfigError):
        config.validate()


def test_too_many_turbines_for_site_raises():
    config = make_small_config(
        n_turbines=200,
        boundary_params={"width": 1000, "height": 1000},
    )
    with pytest.raises(ConfigError, match="容纳"):
        config.validate()


def test_capacity_check_respects_override_n():
    config = make_small_config()  # 默认 6 台可行
    config.validate()
    with pytest.raises(ConfigError):
        config.validate(n_turbines=10_000)
    # 临时校验不得修改配置对象本身。
    assert config.n_turbines == 6


def test_invalid_json_file_raises_configerror(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ConfigError):
        WindFarmConfig.from_json(str(missing))

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        WindFarmConfig.from_json(str(bad))


def test_unknown_config_keys_rejected(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"n_turbines": 5, "unknown_field": 1}), encoding="utf-8")
    with pytest.raises(ConfigError, match="未知"):
        WindFarmConfig.from_json(str(path))


def test_sweep_bounds_validated():
    from wind_farm_opt.cli import WindFarmOptimizerCLI

    config = make_small_config()
    config.visualization.save_plots = False
    cli = WindFarmOptimizerCLI(config)
    with pytest.raises(ConfigError):
        cli.run_turbine_sweep(min_turbines=10, max_turbines=5)
    with pytest.raises(ConfigError):
        cli.run_turbine_sweep(min_turbines=1, max_turbines=10_000)


# ---------------------------------------------------------------------------
# 非法输入不得先创建输出目录
# ---------------------------------------------------------------------------

def test_invalid_config_does_not_create_output_dir(tmp_path):
    from wind_farm_opt.cli import WindFarmOptimizerCLI

    target = tmp_path / "should_not_exist"
    config = make_small_config(n_turbines=0)
    config.visualization.save_dir = str(target)
    with pytest.raises(ConfigError):
        WindFarmOptimizerCLI(config)
    assert not target.exists()
    assert not os.path.exists(str(target))


def test_main_returns_2_and_no_dir_for_bad_args(tmp_path, monkeypatch):
    from wind_farm_opt.cli import main

    target = tmp_path / "cli_bad_out"
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--n-turbines", "0", "--output-dir", str(target)],
    )
    rc = main()
    assert rc == 2
    assert not target.exists()


# ---------------------------------------------------------------------------
# 配置保存 / 加载等价
# ---------------------------------------------------------------------------

def test_json_roundtrip_equivalent(tmp_path):
    config = make_small_config()
    path = tmp_path / "cfg.json"
    config.to_json(str(path))

    loaded = WindFarmConfig.from_json(str(path))

    assert loaded.to_dict() == config.to_dict()
    assert loaded.effective_parameters() == config.effective_parameters()

    # 由两份配置构造的运行对象行为一致。
    positions = _fixed_positions(config)
    aep1 = _aep(config, positions)
    aep2 = _aep(loaded, positions)
    assert aep1.net_aep == pytest.approx(aep2.net_aep)


def test_partial_nested_params_keep_defaults(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"boundary_params": {"width": 2500}}),
        encoding="utf-8",
    )
    loaded = WindFarmConfig.from_json(str(path))
    # 未给出的 height/center 不得被清空。
    assert loaded.boundary_params["width"] == 2500
    assert loaded.boundary_params["height"] == 4000
    assert loaded.boundary_params["center_x"] == 0


def test_roundtrip_partial_params_stays_equivalent(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"wake_decay": 0.08}), encoding="utf-8")
    loaded_once = WindFarmConfig.from_json(str(path))
    path2 = tmp_path / "cfg2.json"
    loaded_once.to_json(str(path2))
    loaded_twice = WindFarmConfig.from_json(str(path2))
    assert loaded_twice.to_dict() == loaded_once.to_dict()


# ---------------------------------------------------------------------------
# 交互分析与批量运行之间的稳定复现
# ---------------------------------------------------------------------------

def _fixed_positions(config: WindFarmConfig) -> np.ndarray:
    turbines = config.create_turbines()
    diameters = np.array([t.rotor_diameter for t in turbines])
    boundary = config.create_boundary()
    rng = np.random.default_rng(123)
    return generate_grid_layout(
        boundary, config.n_turbines, diameters,
        min_multiple=config.optimization.min_spacing_multiple, rng=rng,
    )


def _aep(config: WindFarmConfig, positions: np.ndarray):
    calc = AEPCalculator(
        turbines=config.create_turbines(),
        wind_resource=config.create_wind_resource(),
        wake_model=config.create_wake_model(),
        wake_superposition=config.superposition_method,
    )
    return calc.compute_farm_aep(positions)


def test_results_reproducible_across_runs():
    config = make_small_config()
    positions = _fixed_positions(config)
    first = _aep(config, positions).net_aep
    second = _aep(config, positions).net_aep
    assert first == pytest.approx(second)


def test_sweep_does_not_mutate_main_state(tmp_path):
    from wind_farm_opt.cli import WindFarmOptimizerCLI

    config = make_small_config()
    config.visualization.save_dir = str(tmp_path / "sweep_out")
    config.visualization.save_plots = False
    cli = WindFarmOptimizerCLI(config)

    n_before = cli.config.n_turbines
    turbines_before = list(cli.turbines)
    arrays_before = (
        cli.rotor_diameters.copy(),
        cli.rated_powers.copy(),
        cli.thrust_coefficients.copy(),
    )
    aep_before = cli.aep_calc

    cli.run_turbine_sweep(min_turbines=4, max_turbines=8, step=2)

    assert cli.config.n_turbines == n_before
    assert cli.turbines == turbines_before
    assert cli.aep_calc is aep_before
    np.testing.assert_array_equal(cli.rotor_diameters, arrays_before[0])
    np.testing.assert_array_equal(cli.rated_powers, arrays_before[1])
    np.testing.assert_array_equal(cli.thrust_coefficients, arrays_before[2])

    # 扫描后主流程结果仍可稳定复现。
    positions = _fixed_positions(config)
    assert cli.aep_calc.compute_farm_aep(positions).net_aep == pytest.approx(
        _aep(config, positions).net_aep
    )


# ---------------------------------------------------------------------------
# 结果文件记录最终生效参数
# ---------------------------------------------------------------------------

def test_results_json_records_effective_parameters(tmp_path):
    from wind_farm_opt.cli import WindFarmOptimizerCLI

    out = tmp_path / "run_out"
    config = make_small_config(
        wake_model="gaussian",
        wake_decay=0.044,
    )
    config.visualization.save_dir = str(out)
    config.visualization.save_plots = False
    config.economic.discount_rate = 0.08

    cli = WindFarmOptimizerCLI(config)
    cli.run_full_analysis(
        run_baseline=True,
        run_opt=False,
        run_econ=True,
        run_sweep=False,
        run_viz=False,
        save=True,
    )

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    eff = results["effective_parameters"]

    assert eff["wake_model"] == "gaussian"
    assert eff["wake_decay"] == pytest.approx(0.044)
    assert eff["n_turbines"] == 6
    assert eff["optimization"]["max_iterations"] == 2
    assert eff["optimization"]["min_spacing_multiple"] == pytest.approx(5.0)
    assert eff["economic"]["discount_rate"] == pytest.approx(0.08)
    assert eff["site"]["width"] == 3000

    # 保存的配置与内存配置等价。
    saved_cfg = WindFarmConfig.from_json(str(out / "config.json"))
    assert saved_cfg.to_dict() == config.to_dict()


def test_cli_override_takes_effect_and_replays(tmp_path, monkeypatch):
    """命令行覆写尾流系数后，结果文件中的生效值与实际模型一致。"""
    from wind_farm_opt.cli import main

    out = tmp_path / "cli_out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--wake-model", "gaussian",
            "--wake-decay", "0.058",
            "--n-turbines", "6",
            "--population", "8",
            "--iterations", "2",
            "--no-plots",
            "--no-optimization",
            "--output-dir", str(out),
        ],
    )
    rc = main()
    assert rc == 0

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["effective_parameters"]["wake_decay"] == pytest.approx(0.058)
