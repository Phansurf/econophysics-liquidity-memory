# 流动性记忆、随机价格稳定性与极端波动

Project 5：经济物理学开放性前沿研究。该仓库可离线生成全部合成实验、相图与压力测试结果；技术研究报告（LaTeX 源码与 PDF）位于同级目录 `../article/`。

基于 Wand, Wiedemann, Harren & Kamps, **Physical Review E 109, 024226 (2024)**，DOI [10.1103/PhysRevE.109.024226](https://doi.org/10.1103/PhysRevE.109.024226)，复现其多项式 Langevin 推断核心与合成参数回收。原文正式发表于 2024-02-28，符合报告日期 2026-11-14 向前近三年的要求。

**范围说明：**没有获取原论文 TAQ/CRSP 数据，也没有复现真实市场图 3–6；原图 1 未公开随机参数抽样范围，因此本项目改做明确代表系数的选阶实验，结果不冒充原图复现。全部 CSV 示例均为合成数据。指数 OU 随机波动率已有文献；本课题的扩展在于逆深度解释、与多项式恢复力结合，以及区分平均噪声预算与记忆效应的对照实验。

## 快速开始

建议 Python 3.12（支持 Python 3.10+）。在解压后的项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m frontier_econophysics reproduce
```

Windows 激活方式：`.venv\Scripts\activate`。精确测试环境见 `requirements.txt` 与 `results/environment.json`，依赖范围见 `pyproject.toml`。安装依赖后模拟不需要网络、账号或付费数据库。

也可在已有依赖环境中无需安装包运行（Linux/macOS）：

```bash
PYTHONPATH=src python -m frontier_econophysics demo --steps 1000 --paths 128
PYTHONPATH=src python -m unittest discover -s tests -v
```

完整基线约 5 秒、扩展约 43 秒；总用时依硬件而变。结果是固定种子的统计复现，不保证不同 BLAS/Python 平台位级相同。`reproduce` 会覆盖同名实验结果；`scripts/run_extension.py --quick` 是较小样本的调试模式，会覆盖正式结果，调试后请重新跑正式模式。

## 提交材料

| 内容 | 位置 |
|---|---|
| 中文 A4 技术研究报告（LaTeX 源 + PDF） | `../article/main.tex`，`../article/technical_report_zh.pdf` |
| 完整 Python 源码 | `src/frontier_econophysics/` |
| 依赖与环境 | `pyproject.toml`, `requirements.txt`, `results/environment.json` |
| 基线与扩展结果 | `results/*_summary.json`, `results/data/`, `results/figures/` |
| 多尺度记忆探索（原创） | `src/frontier_econophysics/multiscale.py`, `scripts/run_multiscale.py`, `results/multiscale_summary.json` |
| 半页诊断与详细推导 | 报告第 7 节，`../article/notes/theory_review.md` |
| 答辩调参与逐行代码解读（独立 note，非报告正文） | `../article/notes/defense.md` |
| 原文范围与补全设置 | `../article/notes/literature_scope.md` |
| 物理与集成验证 | `tests/`，`results/verification.txt` |

## 模型与创新问题

基线：`dP = (Σ α_i P^i) dt + σP dW`，`i=1..q`，无常数项。对 EM 一步高斯似然进行列缩放加权最小二乘，`AIC=-2 logL+2(q+1)`。四链 Gibbs MCMC 使用相对于 `dα dσ²` 的平坦先验，输出势函数集合、经典 split-Rhat 和自相关 ESS。

扩展（均为 Itô、无量纲、独立 W/B）：

```text
dP = (a P - b P²) dt + σ P exp(h/2) dW
dh = -(h/m) dt + sqrt(2 λ²/m) dB
h = -log(effective order-book depth)
```

`m>0` 是流动性记忆时间，`λ≥0` 是平稳对数逆深度的标准差。低价切向入侵指数为 `g=a-σ² exp(λ²/2)/2`，边界 `a_c=σ² exp(λ²/2)/2`。有限 `m` 不进入长期边界，却改变波动聚集与有限窗口首达风险。**完整稳定过程的长期 log 增长为零，不能用它测正入侵率**，所以边界实验用 `b=0` 切向过程。

破坏性对照 `--normalize-variance` 将方差改为 `σ² exp(h-λ²/2)`，保持平均方差不变。此时长期边界恢复为 `a_c=σ²/2`；这区分了 Jensen 方差抬升与深度记忆。

积分使用 log-price 正性分裂和精确 OU 端点，保留 Itô 修正，不裁剪价格或隐藏数值失败。时间步内 h 冻结，因此不宣称二阶随机收敛。共噪声收敛检验对 OU 创新作指数核加权，而不只复用随机种子。

**多尺度记忆扩展（原创探索）。** 单一 OU 只有一个松弛时间，波动率自相关是指数衰减；把 $h$ 替换为多个独立 OU 分量的叠加
$h=\sum_k h_k$（`src/frontier_econophysics/multiscale.py`），
使平稳方差仍为 $\lambda^2$、长期入侵指数 $g=a-\sigma^2/2\,e^{\lambda^2/2}$ 对记忆谱 $\{m_k\}$ **不变**（退火不变定理），
但有限时间自平均性、首达风险与 $|r|$ 自相关强烈依赖记忆谱——谱越宽，$|r|$ 自相关越接近幂律。实验结果见 `results/multiscale_summary.json` 与 `results/figures/multiscale_*.pdf`。

## 已实际获得的结果

| 指标 | 常噪声基线 | 流动性扩展 |
|---|---:|---:|
| 收益超额峰度 | 0.0394 | 5.1468 |
| 绝对收益 ACF(1) | 0.00057 | 0.30008 |
| 收益 ACF(1) | -0.00733 | -0.00450 |

基线固定参数实验 100 条全部完成，真值均落在跨轨迹均值 ± 标准差内；这不是置信区间覆盖率检验。代表系数下 q1/q2/q3/q4 的 AIC 准确率为 **32%/79%/44%/27%**，不能声称全面重现原文选阶表现。四链后验最大 Rhat≈1.000215，最低 ESS≈15169/16000。

扩展的切向增长率与理论最大差约 0.00606，最大逐点标准误约 0.02389。固定平均方差后 λ=1.8 的边界从约 1.2409 回到 0.2461（理论 0.245，σ=.7）。所有正式实验数值失败为零。详细设置、路径数与误差条以 JSON/CSV 为准。

## 参数压力测试（demo 命令）

```bash
# 初始状态；默认 a=.9, b=1, sigma=1, T=100
python -m frontier_econophysics demo --m 5 --lambda 0 --out results/live0

# 加强深度异质性；理论边界由 .5 变为 .82436
python -m frontier_econophysics demo --m 5 --lambda 1 --out results/live1

# 加长记忆；长期边界不变，风险与聚集持续时间改变
python -m frontier_econophysics demo --m 20 --lambda 1 --out results/live_memory

# 固定平均方差；理论边界回到 .5
python -m frontier_econophysics demo --m 20 --lambda 1 \
  --normalize-variance --out results/live_control

# JSON 配置，可用命令行覆盖；反向开关为 --no-normalize-variance
python -m frontier_econophysics demo --config configs/live_demo.json --m 10
```

每次生成 `summary.json`（理论边界、首达概率、Wilson95%、失败数、参数和种子）、`paths.npz` 与 `live_phase.png`。`--steps`, `--dt`, `--paths`, `--a`, `--b`, `--sigma`, `--seed`, `--threshold-ratio` 均可修改。有限网格首达会漏掉网格间穿越，风险图不是热力学相变的证据。`m` 对有限时间风险不必单调。

## 数据处理与 CSV 入口

```bash
python scripts/make_example_data.py
python -m frontier_econophysics fit-csv data/synthetic_prices.csv \
  --time-unit-seconds 3000 --out results/example_fit.json
```

`--time-unit-seconds` 定义 SDE 的一个时间单位，示例 300 秒采样映射为 dt=.1；更改时间单位会重标度漂移与噪声。输入 `timestamp,price`，时间严格使用显式 UTC ISO-8601，如 `2020-01-02T09:30:00Z`。命令只拟合**最长连续有效段**，不跨隔夜与长缺口拼接；默认将 UTC 日作为保守会话代理，因此不能直接拿这个默认入口分析日频数据。应先按交易所日历定义日频时间。

报价入口可直接从 Python 使用：

```python
from frontier_econophysics.data import prepare_quotes
prepared = prepare_quotes("data/synthetic_quotes.csv")
print(prepared.diagnostics)
print(prepared.frame)
```

报价输入为 `timestamp,bid,ask`，剔除非法时间、非有限/非正价格、交叉报价、价差≥5；同刻保留原输入最后有效报价，30 分钟取最后有效中价，不 forward-fill。午夜采样箱碰撞会全部剔除并审计。`observed_at` 是实际报价时刻，`timestamp` 是采样箱边界；报价的 `eligible_dt_seconds` 来自实际观测间隔。使用者必须尊重 `transition_eligible`，且应提供交易所日历与公司行动信息。模块不自动“去噪”跳跃，也不会凭空修正拆股或股息。

## 单独重跑与审查

```bash
python scripts/run_baseline.py
python scripts/run_extension.py
python scripts/run_multiscale.py
python scripts/make_architecture.py
```

单元测试检查物理极限，集成测试还检查 JSON 配置覆盖、CSV 分段与命令行真实运行。

`../article/notes/defense.md` 为实际函数逐行解释（答辩备忘，独立于报告正文）；修改代码后行号可能漂移，应以函数名称定位。重要抽查函数：`fit_polynomial`, `sample_posterior`, `step_log_state`, `step_multiscale`, `coupled_convergence`, `prepare_quotes`。

## 研究局限

这是机制探索，不是经过真实市场校准的预测模型。没有显式撮合/订单队列、内生撤单反馈、杠杆约束或经验杠杆效应；单一 OU 是单指数记忆，多尺度叠加把单指数延展为有限尺度的慢衰减，但仍是有限个指数的凸组合，不等于真正的幂律长期记忆。合成厚尾不意味着确认幂律或 SOC。大 λ 和 m≫T 下稀有事件与混合不足会让有限样本显著误导。

源代码为独立实现，使用 MIT 许可；论文与资料的版权归其原作者。完整参考文献见报告及 `../article/notes/literature_scope.md`。
