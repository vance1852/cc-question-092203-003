"""配置管理模块。

用于从JSON文件加载配置，或通过命令行参数构建配置。

配置对象与运行对象保持一对一语义：每个配置字段都会真实传入对应的
模型/边界/风资源构造函数；``create_turbines`` 每次都创建相互独立的
风机实例。配置在构造运行对象之前通过 :meth:`WindFarmConfig.validate`
完成完整校验，非法值会尽早抛出 ``ValueError``。
"""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

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

KNOWN_TURBINE_MODELS = ("V126-3.45MW", "V164-9.5MW")
KNOWN_WAKE_MODELS = ("jensen", "gaussian")
KNOWN_ALGORITHMS = ("ga", "pso")
KNOWN_SUPERPOSITION_METHODS = ("sum_of_squares", "linear")
KNOWN_BOUNDARY_TYPES = ("rectangular", "hexagonal", "irregular", "custom")
KNOWN_WIND_RESOURCE_TYPES = ("default", "uniform")


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
    # 序列化
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, filepath: str) -> "WindFarmConfig":
        """从JSON文件加载配置。

        未知字段（含拼写错误的字段名）会直接抛出 ``ValueError``，
        避免“改了配置但计算结果不变”的静默失效。加载完成后立即执行
        :meth:`validate`。
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"配置文件 {filepath} 的顶层结构必须是 JSON 对象")

        nested = {"optimization", "visualization", "economic"}
        allowed_top = {f.name for f in fields(cls)}
        unknown = set(data) - allowed_top
        if unknown:
            raise ValueError(
                f"配置文件包含未知参数: {sorted(unknown)}；"
                f"可用参数: {sorted(allowed_top)}"
            )

        opt_config = _build_section(
            OptimizationConfig, data.get("optimization", {}), "optimization"
        )
        vis_config = _build_section(
            VisualizationConfig, data.get("visualization", {}), "visualization"
        )
        econ_config = _build_section(
            EconomicConfig, data.get("economic", {}), "economic"
        )

        kwargs = {k: v for k, v in data.items() if k not in nested}
        config = cls(
            optimization=opt_config,
            visualization=vis_config,
            economic=econ_config,
            **kwargs,
        )
        config.validate()
        return config

    def to_dict(self) -> dict:
        """返回可 JSON 序列化的完整配置字典（包含最终生效值）。"""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """保存配置到JSON文件。"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """校验全部配置字段。

        任何会导致无效运行的取值（台数、边界尺寸、算法规模、迭代次数、
        间距倍数、衰减系数、折现率等）都会在这里抛出带有明确说明的
        ``ValueError``，而不是拖到计算中途才以晦涩的方式失败。
        """
        # ---- 基本类型与取值范围 ----
        _require_int("n_turbines", self.n_turbines, min_value=1)

        if not isinstance(self.turbine_model, str):
            raise ValueError("turbine_model 必须是字符串")
        if self.turbine_model not in KNOWN_TURBINE_MODELS:
            raise ValueError(
                f"未知的风机型号: {self.turbine_model!r}，"
                f"可选: {list(KNOWN_TURBINE_MODELS)}"
            )

        if not isinstance(self.wake_model, str):
            raise ValueError("wake_model 必须是字符串")
        if self.wake_model.lower() not in KNOWN_WAKE_MODELS:
            raise ValueError(
                f"wake_model（尾流模型）取值非法: {self.wake_model!r}，"
                f"可选: {list(KNOWN_WAKE_MODELS)}"
            )

        _require_finite("wake_decay", self.wake_decay)
        if not 0.0 < float(self.wake_decay) <= 1.0:
            raise ValueError(
                f"wake_decay（尾流衰减系数）必须在 (0, 1] 范围内，"
                f"当前为 {self.wake_decay}"
            )

        if not isinstance(self.superposition_method, str):
            raise ValueError("superposition_method 必须是字符串")
        if self.superposition_method not in KNOWN_SUPERPOSITION_METHODS:
            raise ValueError(
                f"superposition_method（尾流叠加方法）取值非法: "
                f"{self.superposition_method!r}，可选: {list(KNOWN_SUPERPOSITION_METHODS)}"
            )

        # ---- 优化参数 ----
        opt = self.optimization
        if not isinstance(opt.algorithm, str) or opt.algorithm.lower() not in KNOWN_ALGORITHMS:
            raise ValueError(
                f"optimization.algorithm（优化算法）取值非法: {opt.algorithm!r}，"
                f"可选: {list(KNOWN_ALGORITHMS)}"
            )
        _require_int("optimization.population_size", opt.population_size, min_value=2)
        _require_int("optimization.max_iterations", opt.max_iterations, min_value=1)
        _require_finite("optimization.min_spacing_multiple", opt.min_spacing_multiple)
        if float(opt.min_spacing_multiple) <= 0.0:
            raise ValueError(
                "optimization.min_spacing_multiple（最小间距倍数）必须大于 0，"
                f"当前为 {opt.min_spacing_multiple}"
            )
        if opt.seed is not None:
            _require_int("optimization.seed", opt.seed)

        # ---- 经济性参数 ----
        econ = self.economic
        _require_finite("economic.electricity_price", econ.electricity_price)
        if float(econ.electricity_price) <= 0.0:
            raise ValueError(
                "economic.electricity_price（上网电价）必须大于 0，"
                f"当前为 {econ.electricity_price}"
            )
        _require_finite("economic.discount_rate", econ.discount_rate)
        if not 0.0 <= float(econ.discount_rate) < 1.0:
            raise ValueError(
                "economic.discount_rate（折现率）必须在 [0, 1) 范围内，"
                f"当前为 {econ.discount_rate}"
            )

        # ---- 可视化参数 ----
        if not isinstance(self.visualization.save_dir, str) or not self.visualization.save_dir.strip():
            raise ValueError("visualization.save_dir 必须是非空字符串")

        # ---- 风资源参数 ----
        self._validate_wind_resource()

        # ---- 边界参数（构造真实边界对象以暴露几何错误）----
        boundary = self.create_boundary()

        # ---- 几何可行性：边界必须放得下 n 台满足最小间距的风机 ----
        rotor_diameter = create_default_turbine(self.turbine_model).rotor_diameter
        min_spacing = float(opt.min_spacing_multiple) * rotor_diameter
        # 平面内等距点集的最密排布容量（正三角排布）上界，
        # 取 25% 边界余量以容纳有限多边形的边缘效应。
        packing_capacity = (
            2.0 / math.sqrt(3.0) * boundary.area / min_spacing ** 2
        ) * 1.25
        if self.n_turbines > packing_capacity:
            raise ValueError(
                f"场地面积 {boundary.area / 1e6:.3f} km² 无法以 "
                f"{opt.min_spacing_multiple:g} 倍转子直径 "
                f"(最小间距 {min_spacing:.0f} m) 容纳 {self.n_turbines} 台风机；"
                f"按最密排布估算最多约 {math.floor(packing_capacity)} 台，"
                f"请增大场地、减小台数或减小间距倍数"
            )

    def _validate_wind_resource(self) -> None:
        """校验风资源相关字段。"""
        if not isinstance(self.wind_resource_type, str):
            raise ValueError("wind_resource_type 必须是字符串")
        if self.wind_resource_type.lower() not in KNOWN_WIND_RESOURCE_TYPES:
            raise ValueError(
                f"未知的风资源类型: {self.wind_resource_type!r}，"
                f"可选: {list(KNOWN_WIND_RESOURCE_TYPES)}"
            )
        if not isinstance(self.wind_resource_params, dict):
            raise ValueError("wind_resource_params 必须是 JSON 对象")

        wrp = self.wind_resource_params
        num_sectors = wrp.get("num_sectors", 12)
        _require_int("wind_resource_params.num_sectors", num_sectors, min_value=1)
        if num_sectors > 360:
            raise ValueError(
                f"wind_resource_params.num_sectors 不能超过 360，当前为 {num_sectors}"
            )
        if "mean_speed" in wrp:
            _require_finite("wind_resource_params.mean_speed", wrp["mean_speed"])
            if float(wrp["mean_speed"]) <= 0.0:
                raise ValueError(
                    "wind_resource_params.mean_speed 必须大于 0，"
                    f"当前为 {wrp['mean_speed']}"
                )
        if "dominant_direction" in wrp:
            _require_finite(
                "wind_resource_params.dominant_direction", wrp["dominant_direction"]
            )

    # ------------------------------------------------------------------
    # 运行对象构造（一对一语义）
    # ------------------------------------------------------------------

    def create_turbines(self, n_turbines: Optional[int] = None) -> list[Turbine]:
        """根据配置创建风机列表。

        每台风机都是独立实例（含独立的功率曲线数组），修改其中一台的
        位置或参数不会影响其他机组。

        Parameters
        ----------
        n_turbines : Optional[int]
            可临时指定台数（如台数扫描），不传时使用配置中的台数。
        """
        n = self.n_turbines if n_turbines is None else n_turbines
        return [create_default_turbine(self.turbine_model) for _ in range(n)]

    def create_wake_model(self) -> WakeModel:
        """根据配置创建尾流模型。

        ``wake_decay`` 对 Jensen 和 Gaussian 模型都会真实生效。
        """
        if self.wake_model.lower() == "jensen":
            return JensenWake(wake_decay=float(self.wake_decay))
        elif self.wake_model.lower() == "gaussian":
            return GaussianWake(wake_decay=float(self.wake_decay))
        else:
            raise ValueError(f"未知的尾流模型: {self.wake_model}")

    def create_boundary(self) -> SiteBoundary:
        """根据配置创建场地边界。"""
        if not isinstance(self.boundary_params, dict):
            raise ValueError("boundary_params 必须是 JSON 对象")

        bp = self.boundary_params
        btype = self.boundary_type.lower() if isinstance(self.boundary_type, str) else None

        if btype == "rectangular":
            width = bp.get("width", 4000)
            height = bp.get("height", 4000)
            _require_finite("boundary_params.width", width)
            _require_finite("boundary_params.height", height)
            if float(width) <= 0.0 or float(height) <= 0.0:
                raise ValueError(
                    f"矩形场地的 width/height 必须大于 0，"
                    f"当前为 width={width}, height={height}"
                )
            return create_rectangular_boundary(
                width=width,
                height=height,
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif btype == "hexagonal":
            radius = bp.get("radius", 2500)
            _require_finite("boundary_params.radius", radius)
            if float(radius) <= 0.0:
                raise ValueError(
                    f"六边形场地的 radius 必须大于 0，当前为 {radius}"
                )
            return create_hexagonal_boundary(
                radius=radius,
                center_x=bp.get("center_x", 0),
                center_y=bp.get("center_y", 0),
            )
        elif btype == "irregular":
            return create_irregular_boundary()
        elif btype == "custom":
            if "vertices" not in bp:
                raise ValueError("custom 边界必须在 boundary_params.vertices 中提供顶点")
            try:
                vertices = np.array(bp["vertices"], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"custom 边界顶点无法解析为数值数组: {exc}") from exc
            if not np.all(np.isfinite(vertices)):
                raise ValueError("custom 边界顶点必须全部为有限数值")
            return SiteBoundary(vertices)
        else:
            raise ValueError(
                f"未知的边界类型: {self.boundary_type!r}，"
                f"可选: {list(KNOWN_BOUNDARY_TYPES)}"
            )

    def create_wind_resource(self) -> WindResource:
        """根据配置创建风资源。"""
        wrp = self.wind_resource_params
        if self.wind_resource_type.lower() == "default":
            return create_default_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                dominant_direction=wrp.get("dominant_direction", 270.0),
                mean_speed=wrp.get("mean_speed", 8.5),
            )
        elif self.wind_resource_type.lower() == "uniform":
            from .core.wind_resource import create_simple_wind_resource
            return create_simple_wind_resource(
                num_sectors=wrp.get("num_sectors", 12),
                uniform=True,
                mean_speed=wrp.get("mean_speed", 8.0),
            )
        else:
            raise ValueError(f"未知的风资源类型: {self.wind_resource_type}")


# ----------------------------------------------------------------------
# 校验辅助函数
# ----------------------------------------------------------------------

def _require_finite(name: str, value) -> None:
    """要求值为有限浮点数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值，当前为 {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} 必须是有限数值，当前为 {value}")


def _require_int(name: str, value, min_value: Optional[int] = None) -> None:
    """要求值为整数（拒绝布尔和浮点），可选下界。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是整数，当前为 {value!r}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} 必须 >= {min_value}，当前为 {value}")


def _build_section(section_cls, raw: dict, label: str):
    """从字典构造嵌套配置 dataclass，拒绝未知字段。"""
    if not isinstance(raw, dict):
        raise ValueError(f"配置段 {label!r} 必须是 JSON 对象，当前为 {type(raw).__name__}")
    allowed = {f.name for f in fields(section_cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"配置段 {label!r} 包含未知参数: {sorted(unknown)}；"
            f"可用参数: {sorted(allowed)}"
        )
    try:
        return section_cls(**raw)
    except TypeError as exc:
        raise ValueError(f"配置段 {label!r} 构造失败: {exc}") from exc


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
