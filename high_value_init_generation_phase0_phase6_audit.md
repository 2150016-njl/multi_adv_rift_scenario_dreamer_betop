# 高价值初始状态生成 Phase 0-6 审核小结

本文档记录当前代码相对 `high_value_init_generation_experiment_status_next_steps.md`
的实现状态。文档内容是审核结论，不是运行时指令。

## 总体结论

当前 Phase 0-6 的代码改进整体符合原始技术路线：

- 不把 RIFT label 作为生成模型输入。
- 继续复用 Scenario Dreamer AutoEncoder / LDM，不重写核心扩散模型。
- online 侧新增 `sparse_inpainting`，避免低交通密度场景因 hidden slots 不足直接阻断生成。
- 生成后仍通过 RIFT 原始 interaction logic 做 audit。
- 只有通过 generated-only RIFT audit 的生成 actor 才进入 CBV 注册候选。
- 数据侧增加 sparse context 样本，让训练输入更接近 rolling online。
- 训练侧不仅报告状态误差，也可把 decoded state loss 和物理边界 penalty 纳入训练目标。
- Phase 6 已补齐离线、online smoke、多 route、summary 的标准命令生成脚本。

仍未完成的是实验结果本身：当前本机没有可用 Python/CARLA 环境，因此不能在本机确认
`py_compile`、离线训练、离线评估或 online RIFT/CARLA smoke 的真实结果。

## Phase 对照

### Phase 0/1: sparse online generation 入口

已完成：

- `RIFT/scripts/run.py` 暴露 `--hv_generation_mode sparse_inpainting`。
- `RIFT/rift/cbv/initialization/hv_online_generation.py` 在 sparse 模式下不要求 hidden slots。
- `scenario-dreamer/high_value_init/scene_generator.py` 将 online `sparse_inpainting`
  映射为模型侧 `inpainting` 记录。

审核结论：

- 符合“低密度场景不应因隐藏不足直接失败”的要求。
- sparse 是 online / 数据采样语义，不是独立模型训练 mode。

### Phase 2: failure dump 和诊断信息

已完成：

- online result 增加 `failure_stage`、`generation_attempts`、`spawn_failures`。
- 记录 `generator_called`、`skipped`、`hidden_context_debug`、observed actor 数量。
- summary 脚本统计 `failure_stage_counts`、`failure_stage_by_mode`、`skipped_by_stage`。

审核结论：

- 符合“区分没有调用模型、生成失败、spawn 失败、audit 失败”的要求。

### Phase 3: generated-only audit 和 CBV 注册

已完成：

- RIFT audit pass 改为看 `generated_rift_positive_actor_ids`，不再让原场景 positive actor 贡献成功。
- `carla_env.py` 新增 `_register_high_value_generated_cbvs()`。
- 注册结果回写 online dump，并由 summary 汇总 `cbv_registration_count` 等指标。

风险：

- CBV 注册依赖 `CBV_route_planner.inter_info`。如果 online smoke 出现
  `missing_interaction_info`，需要在 registration 前刷新或重建 generated actor 的 interaction info。

### Phase 4: sparse context 数据集

已完成：

- `build_high_value_scene_dataset.py` 新增 `--sparse_context_copies` 等参数。
- sparse 样本保留 `hv_init_inpainting_v1` schema，仅通过 `context_sampling.policy`
  标记为 `sparse_random_observed`。
- audit 脚本新增 low observed、zero observed、split context summary 等指标。

审核结论：

- 符合“训练覆盖低 observed_count / sparse context”的要求。
- 未引入新的训练 mode，兼容现有训练代码。

### Phase 5: 状态 loss 和物理约束

已完成：

- 训练脚本暴露 decoded position/heading/speed/size loss 权重。
- 新增 `physical_bounds_loss_weight`，默认关闭。
- physical bounds loss 已修正为先反归一化 speed/length/width 后再计算。
- eval 脚本新增 size error、physical invalid、FOV boundary saturation 指标。
- online 和 eval 均支持可选 `clamp_generated_states`。

审核结论：

- “只评估状态误差”不足以改进训练。当前代码已允许把状态误差和物理边界作为训练目标。
- 但最终是否提升 RIFT audit pass rate，仍必须靠重新训练和 online audit 验证。

### Phase 6: 实验闭环命令

已完成：

- 新增 `RIFT/scripts/print_high_value_generation_phase6_commands.py`。
- 扩展 `RIFT/scripts/audit_high_value_generation_pipeline.py`，覆盖 Phase 0-6 关键能力检查。
- `summarize_high_value_generation_online.py` 已汇总 generation/spawn/audit/CBV/rolling/RIFT route statistics。

仍需执行：

- dataset audit。
- Phase 5/6 retraining。
- offline s1 evaluation 与 clamp 对照。
- online smoke route。
- multi-route online evaluation。
- 汇总 route completion、driving score、collision ratio、generated RIFT-positive、CBV registration。

## 推荐执行入口

在服务器环境中先生成完整命令：

```bash
cd $ROOT/RIFT
$PY scripts/print_high_value_generation_phase6_commands.py \
  --root "$ROOT" \
  --py "$PY" \
  --carla_root "$CARLA_ROOT" \
  --hv_root "$HV_ROOT"
```

随后按打印结果依次执行：

1. Static sanity checks。
2. Sparse context dataset audit。
3. Phase 5/6 retraining。
4. Offline s1 evaluation。
5. Offline clamp comparison。
6. Online smoke route。
7. Multi-route online evaluation。
8. Online summary aggregation。

## 必须重点看的指标

- `generation_mode_counts`
- `generator_called_rate_by_mode`
- `generation_success_rate`
- `carla_spawn_success_rate`
- `generated_rift_audit_pass_rate`
- `generated_rift_positive_agent_count`
- `cbv_registration_count`
- `cbv_registration_failure_count`
- `rolling_trigger_count`
- `failure_stage_counts`
- `rift_statistics.route_completion`
- `rift_statistics.driving_score`
- `rift_statistics.collision_routes_ratio`

## 当前验证边界

本地环境中的 `python` 是 Windows Store stub，`py` 不存在，不能实际执行 Python
脚本或 CARLA online 实验。本轮审核因此只完成代码级静态检查、链路检查和命令产物补齐。
