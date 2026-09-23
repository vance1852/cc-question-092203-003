"""配置管理模块。

用于从JSON文件加载配置，或通过命令行参数构建配置。

配置对象与运行期对象（风机、尾流模型等）保持一对一语义：

- ``create_turbines()`` 每次都返回相互独立的风机实例；
- ``create_wake_model()`` 真实使用配置中的衰减系数；
- ``validate()`` 在任何输出目录被创建之前拦截会导致无效运行的参数。
"""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

import numpy as np

from .core.turbine import Turbine, create_default_turbine
from .core.wind_resource import WindResource, create_default_wind_resource
from .core.wake import JensenWake, GaussianWake, WakeModel
from .constraints.boundary import (
    SiteBoundary,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)


class ConfigError(ValueError):
    """配置无效时抛出，错误信息可直接展示给用户。"""


_VALID_WAKE_MODELS = ("jensen", "gaussian")
_VALID_SUPERPOSITION = ("sum_of_squares", "linear")
_VALID_BOUNDARY_TYPES = ("rectangular", "hexagonal", "irregular", "custom")
_VALID_WIND_RESOURCE_TYPES = ("default", "uniform")
_VALID_ALGORITHMS = ("ga", "pso")


def _is_real_number(value: Any) -> bool:
    """是否为有限实数（排除布尔值）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_positive_number(value: Any, name: str) -> None:
    if not _is_real_number(value) or float(value) <= 0.0:
        raise ConfigError(f"{name} 必须是大于0的有限数值，当前为 {value!r}")


@dataclass
class OptimizationConfig:
    """优化算法配置。"""
    algorithm: str = "ga"
    population_size: int = 40
    max_iterations: int = 80
    min_spacing_multiple: float = 5.0
    seed: Optional[int] = 42


@dataclass
class VisualizationConfig:
    """可视化配置。"""
    save_dir: str = "output"
    save_plots: bool = True
    show_plots: bool = False
    plot_wake_heatmap: bool = True


@dataclass
class EconomicConfig:
    """经济性分析配置。"""
    electricity_price: float = 0.45
    discount_rate: float = 0.06
    enable_analysis: bool = True


@dataclass
class WindFarmConfig:
    """完整的风电场分析配置。"""
    n_turbines: int = 15
    turbine_model: str = "V126-3.45MW"
    wake_model: str = "jensen"
    wake_decay: float = 0.07
    superposition_method: str = "sum_of_squares"

    boundary_type: str = "rectangular"
    boundary_params: dict = field(default_factory=lambda: {
        "width": 4000,
        "height": 4000,
        "center_x": 0,
        "center_y": 0,
    })

    wind_resource_type: str = "default"
    wind_resource_params: dict = field(default_factory=lambda: {
        "num_sectors": 12,
        "dominant_direction": 270.0,
        "mean_speed": 8.5,
    })

    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """返回可 JSON 序列化的完整配置字典（深拷贝）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "WindFarmConfig":
        """从字典构建配置，未知配置项或缺省嵌套参数都会尽早报错/补默认值。"""
        if not isinstance(data, dict):
            raise ConfigError(f"配置根节点必须是 JSON 对象，实际为 {type(data).__name__}")

        nested_types = {
            "optimization": OptimizationConfig,
            "visualization": VisualizationConfig,
            "economic": EconomicConfig,
        }

        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}

        for key, value in data.items():
            if key in nested_types:
                continue
            if key not in allowed:
                raise ConfigError(f"未知的配置项: {key!r}")
            kwargs[key] = value

        for key, sub_cls in nested_types.items():
            sub_data = data.get(key, {})
            if sub_data is None:
                sub_data = {}
            if not isinstance(sub_data, dict):
                raise ConfigError(f"配置项 {key!r} 必须是 JSON 对象")
            sub_allowed = {f.name for f in fields(sub_cls)}
            for sub_key in sub_data:
                if sub_key not in sub_allowed:
                    raise ConfigError(f"配置项 {key!r} 中存在未知字段: {sub_key!r}")
            kwargs[key] = sub_cls(**sub_data)

        config = cls(**kwargs)

        # 用户只给出部分边界/风资源参数时，与默认值合并而不是整体覆盖。
        defaults = cls()
        if isinstance(data.get("boundary_params", {}), dict):
            config.boundary_params = {
                **defaults.boundary_params,
                **data.get("boundary_params", {}),
            }
        if isinstance(data.get("wind_resource_params", {}), dict):
            config.wind_resource_params = {
                **defaults.wind_resource_params,
                **data.get("wind_resource_params", {}),
            }

        return config

    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置，并立即执行校验。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"配置文件不存在: {filepath}")
        except json.JSONDecodeError as exc:
            raise ConfigError(f"配置文件不是合法的JSON（{filepath}）: {exc}") from exc

        config = cls.from_dict(data)
        config.validate()
        return config

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def effective_parameters(self) -> dict:
        """最终生效的关键参数（含嵌套参数补全后的实际值），用于结果复核。"""
        bp = self.boundary_params
        wrp = self.wind_resource_params
        if self.boundary_type.lower() == "rectangular":
            site = {
                "boundary_type": "rectangular",
                "width": float(bp.get("width", 4000)),
                "height": float(bp.get("height", 4000)),
                "center_x": float(bp.get("center_x", 0)),
                "center_y": float(bp.get("center_y", 0)),
            }
        elif self.boundary_type.lower() == "hexagonal":
            site = {
                "boundary_type": "hexagonal",
                "radius": float(bp.get("radius", 2500)),
            }
        else:
            site = {"boundary_type": self.boundary_type}

        return {
            "n_turbines": int(self.n_turbines),
            "turbine_model": self.turbine_model,
            "wake_model": self.wake_model.lower(),
            "wake_decay": float(self.wake_decay),
            "superposition_method": self.superposition_method,
            "site": site,
            "wind_resource": {
                "type": self.wind_resource_type.lower(),
                "num_sectors": int(wrp.get("num_sectors", 12)),
                "dominant_direction": float(wrp.get("dominant_direction", 270.0)),
                "mean_speed": float(wrp.get("mean_speed", 8.5)),
            },
            "optimization": {
                "algorithm": self.optimization.algorithm.lower(),
                "population_size": int(self.optimization.population_size),
                "max_iterations": int(self.optimization.max_iterations),
                "min_spacing_multiple": float(self.optimization.min_spacing_multiple),
                "seed": self.optimization.seed,
            },
            "economic": {
                "electricity_price": float(self.economic.electricity_price),
                "discount_rate": float(self.economic.discount_rate),
                "enable_analysis": bool(self.economic.enable_analysis),
            },
        }

    # ------------------------------------------------------------------
    # 运行期对象构造（每次返回独立实例）
    # ------------------------------------------------------------------

    def create_turbines(self, n_turbines: Optional[int] = None) -> list[Turbine]:
        """根据配置创建风机列表，每台风机都是互不共享状态的独立实例。"""
        n = self.n_turbines if n_turbines is None else n_turbines
        return [create_default_turbine(self.turbine_model) for _ in range(n)]

    def create_wake_model(self) -> WakeModel:
        """根据配置创建尾流模型，衰减系数一律取本配置的 ``wake_decay``。"""
        model = self.wake_model.lower()
        if model == "jensen":
            return JensenWake(wake_decay=self.wake_decay)
        elif model == "gaussian":
            return GaussianWake(wake_decay=self.wake_decay)
        else:
            raise ConfigError(f"未知的尾流模型: {self.wake_model}")

    def create_boundary(self) -> SiteBoundary:
        """根据配置创建场地边界。"""
        bp = self.boundary_params
        btype = self.boundary_type.lower()
        if btype == "rectangular":
            return create_rectangular_boundary(
                width=bp.get("width", 4000),
                height=bp.get("height", 4000),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif btype == "hexagonal":
            return create_hexagonal_boundary(
                radius=bp.get("radius", 2500),
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif btype == "irregular":
            return create_irregular_boundary()
        elif btype == "custom":
            if "vertices" not in bp:
                raise ConfigError("custom 边界必须在 boundary_params 中提供 vertices")
            try:
                vertices = np.array(bp["vertices"], dtype=float)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"custom 边界顶点无法解析为数值数组: {exc}") from exc
            return SiteBoundary(vertices)
        else:
            raise ConfigError(f"未知的边界类型: {self.boundary_type}")

    def create_wind_resource(self) -> WindResource:
        """根据配置创建风资源。"""
        wrp = self.wind_resource_params
        rtype = self.wind_resource_type.lower()
        if rtype == "default":
            return create_default_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                dominant_direction=wrp.get("dominant_direction", 270.0),
                mean_speed=wrp.get("mean_speed", 8.5),
            )
        elif rtype == "uniform":
            from .core.wind_resource import create_simple_wind_resource
            return create_simple_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                uniform=True,
                mean_speed=wrp.get("mean_speed", 8.0),
            )
        else:
            raise ConfigError(f"未知的风资源类型: {self.wind_resource_type}")

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self, n_turbines: Optional[int] = None) -> None:
        """校验全部参数。任何不满足都会抛出 ``ConfigError``。

        Parameters
        ----------
        n_turbines : Optional[int]
            临时以指定台数做容量可行性校验（例如台数扫描的最大台数），
            不修改本配置对象。
        """
        n = self.n_turbines if n_turbines is None else n_turbines

        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ConfigError(f"风机台数必须是不小于1的整数，当前为 {n!r}")

        if not isinstance(self.turbine_model, str):
            raise ConfigError("风机型号必须是字符串")
        try:
            prototype = create_default_turbine(self.turbine_model)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

        if not isinstance(self.wake_model, str) or self.wake_model.lower() not in _VALID_WAKE_MODELS:
            raise ConfigError(
                f"未知的尾流模型: {self.wake_model!r}，可选值: {', '.join(_VALID_WAKE_MODELS)}"
            )
        if self.superposition_method not in _VALID_SUPERPOSITION:
            raise ConfigError(
                f"未知的尾流叠加方法: {self.superposition_method!r}，"
                f"可选值: {', '.join(_VALID_SUPERPOSITION)}"
            )
        _require_positive_number(self.wake_decay, "尾流衰减系数 wake_decay")

        if not isinstance(self.boundary_type, str) or self.boundary_type.lower() not in _VALID_BOUNDARY_TYPES:
            raise ConfigError(
                f"未知的边界类型: {self.boundary_type!r}，可选值: {', '.join(_VALID_BOUNDARY_TYPES)}"
            )
        if not isinstance(self.boundary_params, dict):
            raise ConfigError("boundary_params 必须是 JSON 对象")
        btype = self.boundary_type.lower()
        if btype == "rectangular":
            _require_positive_number(self.boundary_params.get("width"), "场地宽度 width")
            _require_positive_number(self.boundary_params.get("height"), "场地高度 height")
        elif btype == "hexagonal":
            _require_positive_number(self.boundary_params.get("radius"), "六边形外接圆半径 radius")

        if not isinstance(self.wind_resource_type, str) or self.wind_resource_type.lower() not in _VALID_WIND_RESOURCE_TYPES:
            raise ConfigError(
                f"未知的风资源类型: {self.wind_resource_type!r}，"
                f"可选值: {', '.join(_VALID_WIND_RESOURCE_TYPES)}"
            )
        wrp = self.wind_resource_params
        if not isinstance(wrp, dict):
            raise ConfigError("wind_resource_params 必须是 JSON 对象")
        num_sectors = wrp.get("num_sectors", 12)
        if not isinstance(num_sectors, int) or isinstance(num_sectors, bool) or not (1 <= num_sectors <= 360):
            raise ConfigError(f"风向扇区数必须是 1~360 的整数，当前为 {num_sectors!r}")
        _require_positive_number(wrp.get("mean_speed", 8.5), "平均风速 mean_speed")
        if "dominant_direction" in wrp and not _is_real_number(wrp["dominant_direction"]):
            raise ConfigError(f"主风向必须是数值，当前为 {wrp['dominant_direction']!r}")

        opt = self.optimization
        if not isinstance(opt.algorithm, str) or opt.algorithm.lower() not in _VALID_ALGORITHMS:
            raise ConfigError(
                f"未知的优化算法: {opt.algorithm!r}，可选值: {', '.join(_VALID_ALGORITHMS)}"
            )
        if not isinstance(opt.population_size, int) or isinstance(opt.population_size, bool) or opt.population_size < 2:
            raise ConfigError(f"种群/粒子群规模必须是不小于2的整数，当前为 {opt.population_size!r}")
        if not isinstance(opt.max_iterations, int) or isinstance(opt.max_iterations, bool) or opt.max_iterations < 1:
            raise ConfigError(f"最大迭代次数必须是不小于1的整数，当前为 {opt.max_iterations!r}")
        _require_positive_number(opt.min_spacing_multiple, "最小间距倍数 min_spacing_multiple")
        if opt.seed is not None and not isinstance(opt.seed, int):
            raise ConfigError(f"随机种子必须是整数或 null，当前为 {opt.seed!r}")

        econ = self.economic
        _require_positive_number(econ.electricity_price, "上网电价 electricity_price")
        if not _is_real_number(econ.discount_rate) or not (0.0 <= float(econ.discount_rate) < 1.0):
            raise ConfigError(
                f"折现率必须位于 [0, 1) 区间，当前为 {econ.discount_rate!r}"
            )

        if not isinstance(self.visualization.save_dir, str) or not self.visualization.save_dir.strip():
            raise ConfigError("输出目录 save_dir 必须是非空字符串")

        # 构造一次边界，以提前发现 custom 顶点非法等问题。
        boundary = self.create_boundary()

        # 容量可行性：按最小间距把外接矩形划分为边长 d/2 的网格，
        # 同一格内任意两点距离 < d，因此每格至多一台风机。
        min_spacing = float(opt.min_spacing_multiple) * float(prototype.rotor_diameter)
        x_span = boundary.x_max - boundary.x_min
        y_span = boundary.y_max - boundary.y_min
        if x_span <= 0.0 or y_span <= 0.0:
            raise ConfigError("场地边界的外接矩形宽高都必须为正")
        cols = max(1, math.ceil(x_span / (min_spacing / 2.0)))
        rows = max(1, math.ceil(y_span / (min_spacing / 2.0)))
        capacity = cols * rows
        if n > capacity:
            raise ConfigError(
                f"场地无法容纳 {n} 台风机：最小间距 {min_spacing:.0f} m "
                f"（{opt.min_spacing_multiple:g}×{prototype.rotor_diameter:g} m），"
                f"当前边界最多约可布置 {capacity} 台；请减小台数/间距倍数或扩大场地"
            )


def create_sample_config() -> WindFarmConfig:
    """创建示例配置。"""
    return WindFarmConfig(
        n_turbines=12,
        turbine_model="V126-3.45MW",
        wake_model="jensen",
        wake_decay=0.07,
        boundary_type="rectangular",
        boundary_params={"width": 3500, "height": 3500, "center_x": 0, "center_y": 0},
        wind_resource_type="default",
        wind_resource_params={"num_sectors": 12, "dominant_direction": 270.0, "mean_speed": 8.5},
        optimization=OptimizationConfig(
            algorithm="ga",
            population_size=30,
            max_iterations=50,
            min_spacing_multiple=5.0,
            seed=42,
        ),
    )
