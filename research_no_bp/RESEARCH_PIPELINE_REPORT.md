# Research Pipeline Report

日期：2026-06-03

## Direction

替代反向传播的类脑学习算法，调研 0 阶优化、BP 替代、STDP / Hebbian、神经科学和医学相关动机，并完成 idea、实验、review、改进、复跑闭环。

## Chosen Idea

**三因子 Direct Feedback Alignment**

形式：

```text
delta_w = pre_activity * post_eligibility * modulatory_feedback
```

它保留局部突触更新形式，同时使用固定随机反馈通道传递任务误差信号，避免标准 BP 的逐层精确反传和 weight transport。

## Pipeline

1. 文献调研
2. idea 生成和筛选
3. pilot 实验
4. full_v1 扩大实验
5. review 后增加 `output_only` 对照，得到 full_v2
6. review 后增加 `dfa_resampled` 消融，得到 full_v3
7. 生成 narrative report

## Outputs

| file | purpose |
|---|---|
| `idea-stage/LITERATURE_REVIEW.md` | 文献调研 |
| `idea-stage/IDEA_REPORT.md` | idea 排序和选择 |
| `no_bp_mnist_experiment.py` | 可复现实验脚本 |
| `results/pilot_v1/` | pilot 结果 |
| `results/full_v1/` | 扩大训练结果 |
| `results/full_v2/` | 增加 output-only 对照 |
| `results/full_v3/` | 增加 dfa_resampled 消融 |
| `review-stage/AUTO_REVIEW.md` | review 和改进记录 |
| `NARRATIVE_REPORT.md` | 写作交接报告 |

## Final Result

第一阶段静态分类结果来自 `results/full_v3`：

| method | mean test acc | std |
|---|---:|---:|
| BP | 0.9128 | 0.0072 |
| dfa_3factor | 0.9042 | 0.0099 |
| output_only | 0.8245 | 0.0150 |
| dfa_resampled | 0.7970 | 0.0215 |
| STDP | 0.2865 | 0.0349 |
| SPSA zero-order | 0.1688 | 0.0088 |

## Supported Claim

本轮支持一个有限 claim：

> 在小规模 MNIST 上，固定随机反馈调制的三因子局部学习规则能接近 BP，并明显强于固定随机特征读出、重采样反馈、minimal STDP 和朴素零阶 SPSA。

第二阶段时序 next-token 结果来自 `temporal/results/delayed_hard_v1` 和 `temporal/results/bptt_tuned_v2`：

| method | target acc |
|---|---:|
| bigram | 0.0000 |
| reservoir | 0.0000 |
| eprop_resampled | 0.0000 |
| eprop_3factor | 1.0000 |
| BPTT tuned | 1.0000 |

这支持一个更贴近最终目标的 claim：

> 在低维长延迟 next-token 任务上，三因子 eligibility trace 可以像调好参的 BPTT 一样训练隐藏状态保留 cue；固定 reservoir、bigram 和重采样反馈无法完成目标位置预测。

## Not Supported Yet

本轮还不支持：

- 该方法可替代大规模 BP
- 该方法解释人脑全部学习机制
- STDP 本身足以做监督分类
- 可直接迁移到 LLM
- 可解释复杂时序联想

## Verification

已执行：

```bash
python3 -m py_compile research_no_bp/no_bp_mnist_experiment.py
python3 research_no_bp/no_bp_mnist_experiment.py --out-dir research_no_bp/results/full_v3 --train-size 5000 --test-size 2000 --seeds 0,1,2 --epochs 12 --hidden-dim 128 --methods bp,dfa_3factor,dfa_resampled,output_only,zo_spsa,stdp --stdp-neurons 128 --stdp-steps 10 --zo-epochs 8
```

## Next Loop

已完成一个时序 next-token loop。接下来最应该继续做：

```text
delay sweep + compositional cue + frozen backbone adapter
```

目标：

- 验证是否只是固定 delay 记忆
- 验证是否能做组合关系而不是查表
- 验证是否能兼容已有模型，作为 no-BP recurrent memory adapter
