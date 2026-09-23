# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 配置语义与校验

- 配置文件与命令行覆写合并为一份**最终生效配置**，其中的参数（含 `wake_decay`，对 Jensen 与 Gaussian 模型同样生效）会一对一传入对应模型；每台风机都是独立实例，修改单台机位不会影响其他机组。
- 风机台数、边界尺寸、种群/粒子规模、迭代次数、最小间距倍数、尾流衰减系数、折现率、扫描范围等非法取值会在**创建任何输出目录或运行对象之前**被拒绝，并给出明确的中文错误（退出码 2）。
- 配置文件中的未知字段（常见于字段名拼写错误）会被直接拒绝，避免“改了参数但结果不变”。
- 结果目录中的 `results.json` 会记录 `effective_config`（最终生效的完整配置）以便复核，同目录的 `config.json` 重新加载后与该配置等价；固定随机种子时同一份配置可稳定重放。

## 回归测试

```bash
python -m pytest tests/
```

测试覆盖配置校验、JSON 存取等价、参数真实进入尾流模型、风机实例隔离、非法输入不产生半成品输出目录、台数扫描隔离性以及固定种子下的可重放性。
