# 验证记录 / Validation

验证日期：2026-11-15。所有结果限于下述配置；不宣称完成全部公开模型的任务质量评估。

## 环境

- Python 3.13，PyTorch 2.10.0+cpu + 匹配的 torch_npu，Transformers 4.57.1。
- 健康 NPU 0：真实权重；健康 NPU 2：架构契约矩阵。
- 基础环境的 torchvision 二进制无法正确加载 `torchvision::nms`；在独立 venv 中安装
  匹配的 torchvision 0.25.0+cpu 后测试，没有替换机器原有包或驱动。
- Ruff 0.15.22；Chromium 159 离线 HTML 检查。

## 验证层次

| 层次 | 结果 | 实际验证内容 |
|---|---|---|
| CPU 完整套件 | 60 通过 | 原有回归、数学参考、输入边界、序列移位、CSV/JSON、比较和报告 |
| 24 类架构 CPU | 全部通过 | 使用真实 Transformers 模块、小型随机权重，HF 保存/加载/前向 |
| 24 类架构 NPU FP32 | 全部通过 | 原生 logits 与最后一层 readout 一致性、目标评分 |
| NPU BF16 完整套件 | 60 通过 | 其中 24 类架构在 NPU BF16 执行；其余数学/报告单测在 CPU |
| 真实预训练权重 | 通过 | SmolLM2-135M，CPU FP32 / NPU BF16，同输入的评分、层轨迹、CKA 和报告 |
| 离线浏览器 | 通过 | 可靠性 SVG、主题切换、样本筛选、390px 手机宽度、Bootstrap/CKA 面板、无 JS 错误 |

24 类：Llama、Qwen2、Qwen3、Mistral、Mixtral、Qwen2-MoE、Qwen3-MoE、Gemma、Gemma2、Phi、
Phi3、OLMo、OLMo2、StarCoder2、StableLM、Granite、Cohere、GPT-2、GPT-J、GPT-NeoX、OPT、BLOOM、Falcon、MPT。

这些随机小模型测试用例检验调用契约，**不代表对应公开训练权重的任务效果测试**。
其中 OPT 特意使用 `word_embed_proj_dim != hidden_size`，Gemma2 使用显式 softcap，Cohere 使用
非默认输出缩放；均以最终层重放对齐原生 logits 作为正确性门槛。

## 原生评分与统计参考

- 均匀分布：PPL=词表大小，Brier=1−1/V；显式检验并列排名约定。
- 非均匀分布：概率、NLL、排名、Brier 与独立 Python 标量参考一致（FP32 误差范围）。
- 带上下文变化的 next-character 模型验证 logits/labels 的一位移位；多种窗口/步长下每个目标
  只计算一次，且 prompt 不进入分母。
- fast tokenizer 的跨边界合并、空 prompt/BOS、空 target、UNK 提示、非有限数值处理。
- Bootstrap：同结果自比较区间为 [0,0]；固定种子复现、单样本不估计区间。
- CKA：正交旋转、各向同性缩放、平移不变；常量/少样本不估计；512 观测预算。
- 比较拒绝不匹配的 prompt、target、样本集合、重复 ID、层/位置覆盖和评分协议。

## 真实权重 Smoke Test

模型：[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)，
固定 Hub revision：`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`。
输入为仓内 `examples/research_smoke.jsonl`，4 个样本、11 个目标 token。

| 项目 | CPU FP32 | NPU BF16 |
|---|---:|---:|
| 目标 mean NLL | 1.2970793 | 1.2962460 |
| 条件 PPL | 3.6585954 | 3.6555478 |
| 下一 token accuracy | 0.6363636 | 0.6363636 |
| ECE，15 桶 | 0.2724470 | 0.2719800 |
| 最终层 logits 最大误差 | 3.2425e-5 | 0 |

同一 NPU 结果自比较：Top-1 一致率=1，Top-K 重合率=1，ΔNLL=0，95% 配对区间=[0,0]；
各层 CKA 在浮点误差范围内为 1。

CPU FP32 为 A、NPU BF16 为 B：ΔNLL=−0.00083334 nats/token，配对 95% 区间约
[−0.0071993, 0.0078233]。这是**同一权重的后端/数值精度对照**，不是训练改进。
4 个短样本只验证端到端流程，不能据此发表模型质量、校准能力或硬件精度等价结论。

复现（先安装与设备匹配的 torch_npu）：

```bash
llm-training-evaluator analyze \
  --model HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --device npu:0 --dtype bfloat16 --attn-implementation eager \
  --samples examples/research_smoke.jsonl --positions 0,-1 \
  --include-hidden-states --max-length 128 --scoring-stride 32 \
  --top-k 5 --similarity-top-k 3 --output outputs/smollm2-npu.json

# 同结果自比较应得到零差异；不需要再次加载模型
llm-training-evaluator compare-results \
  --analysis-a outputs/smollm2-npu.json --analysis-b outputs/smollm2-npu.json \
  --output outputs/smollm2-self.json

pytest -q
EVALUATOR_TEST_DEVICE=npu:2 EVALUATOR_TEST_DTYPE=bfloat16 pytest -q
ruff check src tests scripts
```

生成 JSON、中文离线 HTML、token CSV、layer CSV 和对比 delta CSV；权重、大型日志和生成图片不入库。

## 未覆盖项

- 本地完整 DeepSeek-V2-Lite（约 31 GB）在共享存储读取阶段加载进程终止，未取得完整运行结果；
  不将它列为本次真实权重验证通过项。
- 未对全部同族模型版本、Transformers 5、自定义远程代码、CUDA 量化、磁盘 offload、
  特殊 MoE dispatch 算法或多模态模型做本轮真实硬件验收。
- 没有进行大规模学术数据集基准、长文档质量评测或训练随机种子重复实验。
  方法可以用于此类研究，但需要调用方提供并冻结相应数据与实验设计。
