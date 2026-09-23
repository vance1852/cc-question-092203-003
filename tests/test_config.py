"""配置校验与序列化的回归测试。

覆盖：
- JSON / dataclass 字段真正进入模型（重点：Gaussian 衰减系数）
- 非法值尽早抛出明确错误（台数、边界、算法规模、迭代、间距、衰减、折现率等）
- 未知配置字段被拒绝（避免“改了但静默不生效”）
- 保存 -> 加载保持等价
"""

import json

import numpy as np
import pytest

from wind_farm_opt.config import (
    WindFarmConfig,
    OptimizationConfig,
    EconomicConfig,
    create_sample_config,
)
from wind_farm_opt.core.wake import GaussianWake, JensenWake


def make_config(**overrides) -> WindFarmConfig:
    """构造一个小规模、几何可行的基础配置。"""
    config = create_sample_config()
    config.n_turbines = 3
    config.boundary_params = {"width": 1500, "height": 1500, "center_x": 0, "center_y": 0}
    config.optimization.population_size = 4
    config.optimization.max_iterations = 2
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ----------------------------------------------------------------------
# 一对一语义：配置值必须真实进入模型
# ----------------------------------------------------------------------

def test_gaussian_wake_decay_flows_from_config():
    """回归：JSON 中调整 gaussian 衰减系数后结果完全不变。"""
    config = make_config(wake_model="gaussian", wake_decay=0.05)
    model = config.create_wake_model()
    assert isinstance(model, GaussianWake)
    assert model.wake_decay == pytest.approx(0.05)


def test_gaussian_non_default_decay_changes_aep():
    """不同的 Gaussian 衰减系数必须产生不同的尾流结果（端到端）。"""
    from wind_farm_opt.farm.aep import AEPCalculator
    from wind_farm_opt.optimization.baseline import generate_grid_layout

    losses = {}
    for decay in (0.02, 0.08):
        config = make_config(wake_model="gaussian", wake_decay=decay)
        turbines = config.create_turbines()
        diameters = np.array([t.rotor_diameter for t in turbines])
        rng = np.random.default_rng(1)
        positions = generate_grid_layout(
            config.create_boundary(), 3, diameters, min_multiple=5.0, rng=rng
        )
        calc = AEPCalculator(
            turbines=turbines,
            wind_resource=config.create_wind_resource(),
            wake_model=config.create_wake_model(),
            wake_superposition=config.superposition_method,
            speed_step=1.0,
        )
        losses[decay] = calc.compute_farm_aep(positions).wake_loss_pct

    assert losses[0.02] != pytest.approx(losses[0.08])


def test_jensen_wake_decay_flows_from_config():
    config = make_config(wake_model="jensen", wake_decay=0.11)
    model = config.create_wake_model()
    assert isinstance(model, JensenWake)
    assert model.wake_decay == pytest.approx(0.11)


def test_gaussian_decay_loaded_from_json(tmp_path):
    """复现方案团队的操作路径：改 JSON -> 加载 -> 生效。"""
    path = tmp_path / "plan.json"
    config = make_config(wake_model="gaussian", wake_decay=0.042)
    config.to_json(str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["wake_decay"] = 0.061
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = WindFarmConfig.from_json(str(path))
    assert loaded.create_wake_model().wake_decay == pytest.approx(0.061)


# ----------------------------------------------------------------------
# 每台风机独立实例
# ----------------------------------------------------------------------

def test_each_turbine_is_independent_instance():
    """回归：给一台风机写位置不能改变其他机组。"""
    config = make_config()
    turbines = config.create_turbines()

    assert len(turbines) == 3
    assert len({id(t) for t in turbines}) == 3
    assert all(t.position is None for t in turbines)

    turbines[0].position = (123.0, 456.0)
    assert turbines[1].position is None
    assert turbines[2].position is None

    # 功率曲线数组也必须各自独立
    assert turbines[0].power_curve is not turbines[1].power_curve
    assert not np.shares_memory(turbines[0].power_curve, turbines[1].power_curve)


def test_create_turbines_with_explicit_count():
    config = make_config(n_turbines=3)
    assert len(config.create_turbines(7)) == 7
    # 不传参时仍使用配置台数
    assert len(config.create_turbines()) == 3


# ----------------------------------------------------------------------
# 非法值：尽早、明确地失败
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("n_turbines", 0),
        ("n_turbines", -3),
        ("n_turbines", 2.5),
        ("wake_decay", 0.0),
        ("wake_decay", -0.1),
        ("wake_decay", 1.5),
        ("wake_decay", float("nan")),
        ("wake_model", "fancy"),
        ("superposition_method", "quadratic"),
    ],
)
def test_invalid_top_level_fields(field_name, bad_value):
    config = make_config(**{field_name: bad_value})
    with pytest.raises(ValueError, match=field_name.split(".")[-1]):
        config.validate()


@pytest.mark.parametrize(
    "section,attr,bad_value",
    [
        ("optimization", "algorithm", "ant_colony"),
        ("optimization", "population_size", 1),
        ("optimization", "population_size", 0),
        ("optimization", "max_iterations", 0),
        ("optimization", "max_iterations", -5),
        ("optimization", "min_spacing_multiple", 0.0),
        ("optimization", "min_spacing_multiple", -2.0),
        ("optimization", "seed", 1.5),
        ("economic", "discount_rate", -0.01),
        ("economic", "discount_rate", 1.0),
        ("economic", "discount_rate", 2.0),
        ("economic", "electricity_price", 0.0),
        ("economic", "electricity_price", -0.5),
    ],
)
def test_invalid_nested_fields(section, attr, bad_value):
    config = make_config()
    setattr(getattr(config, section), attr, bad_value)
    with pytest.raises(ValueError, match=attr):
        config.validate()


@pytest.mark.parametrize(
    "params,match",
    [
        ({"width": 0, "height": 1000, "center_x": 0, "center_y": 0}, "width"),
        ({"width": 1000, "height": -5, "center_x": 0, "center_y": 0}, "height"),
        ({"width": float("inf"), "height": 1000, "center_x": 0, "center_y": 0}, "width"),
    ],
)
def test_invalid_rectangular_boundary(params, match):
    config = make_config(boundary_params=params)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_hexagonal_boundary_radius_must_be_positive():
    config = make_config(boundary_type="hexagonal", boundary_params={"radius": 0})
    with pytest.raises(ValueError, match="radius"):
        config.validate()


def test_infeasible_geometry_rejected():
    """台数相对于场地/间距明显放不下时，应在校验阶段直接报错。"""
    config = make_config(
        n_turbines=100,
        boundary_params={"width": 500, "height": 500, "center_x": 0, "center_y": 0},
    )
    with pytest.raises(ValueError, match="容纳"):
        config.validate()


def test_invalid_wind_resource():
    config = make_config()
    config.wind_resource_params = {"num_sectors": 0}
    with pytest.raises(ValueError, match="num_sectors"):
        config.validate()

    config = make_config()
    config.wind_resource_params = {"num_sectors": 12, "mean_speed": 0}
    with pytest.raises(ValueError, match="mean_speed"):
        config.validate()


def test_unknown_json_keys_rejected(tmp_path):
    """拼写错误的字段名必须报错，而不是被静默忽略。"""
    path = tmp_path / "bad.json"
    config = make_config()
    config.to_json(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["wake_decai"] = 0.99  # 拼写错误
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="wake_decai"):
        WindFarmConfig.from_json(str(path))


def test_unknown_nested_json_keys_rejected(tmp_path):
    path = tmp_path / "bad_nested.json"
    config = make_config()
    config.to_json(str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["optimization"]["pop_size"] = 99  # 正确名称是 population_size
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="pop_size"):
        WindFarmConfig.from_json(str(path))


# ----------------------------------------------------------------------
# 保存 -> 加载等价
# ----------------------------------------------------------------------

def test_save_load_roundtrip_equivalent(tmp_path):
    config = make_config(wake_model="gaussian", wake_decay=0.044)
    config.optimization.algorithm = "pso"
    config.economic.discount_rate = 0.08

    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = WindFarmConfig.from_json(str(path))

    assert loaded.to_dict() == config.to_dict()
    assert loaded.create_wake_model().wake_decay == pytest.approx(0.044)


def test_effective_config_is_json_serializable(tmp_path):
    config = make_config()
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    # to_dict 必须可以无损失地再次序列化
    json.dumps(config.to_dict(), ensure_ascii=False)
