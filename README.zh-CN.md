# LLM Training Evaluator · 大模型训练与检查点分析

[English](README.md) | [方法、公式及论文依据](docs/METHODOLOGY.md)

面向 Hugging Face 格式 decoder-only 因果语言模型，分析目标续写的条件似然、逐层预测、
校准误差、Embedding 邻居、MoE 路由与检查点差异，生成**自包含中文 HTML、JSON 和整洁 CSV**。
支持 CPU、CUDA 与 Ascend NPU。

## 0.2 新增功能

- **有监督目标评分**：teacher-forced token 概率/排名、NLL、条件 PPL、Top-1/Top-5 accuracy、MRR。
- **不确定性诊断**：全词表熵、Brier、ECE；可靠性图、桶内样本数、目标 token 损失热图。
- **统计比较**：按完整样本配对 Bootstrap 的 ΔNLL 95% 区间，按标签切片及高损失样本列表。
- **表征比较**：可选 Linear CKA，比较训练前后同层表示，标注观测数及少样本偏差。
- **逐层诊断**：全词表相邻层 JS、预测稳定起始层，按 token 位置独立绘图。
- **兼容性**：扩展 MPT、OPT 输出投影、Gemma2 softcap、Cohere/Granite logits 缩放。
  24 类架构的离线小型模型测试覆盖保存、加载、前向和最终层一致性。
- **可复现导出**：实际 token 边界、上下文协议、样本与 tokenizer 哈希、库版本、CSV 表格。

## 支持范围

原生目标评分使用 `AutoModelForCausalLM` / `AutoTokenizer`，不要求模型具有某种内部模块命名。
**Hugging Face 权重格式本身不能保证模型可执行**：还需要当前 Transformers 支持相应架构、
完整 tokenizer/配置/自定义代码、正确依赖、可用设备算子及足够内存。

逐层探针覆盖 Llama、Qwen2/Qwen3、Mistral、Mixtral、Qwen2/3 MoE、Gemma/Gemma2、Phi/Phi3、
OLMo/OLMo2、StarCoder2、StableLM、Granite、Cohere、GPT-2/GPT-J/GPT-NeoX、OPT、BLOOM、Falcon、MPT。
这些是架构契约测试，不表示所有同名模型版本都已实测。未知布局可配置 `--layers-path` 和
`--norm-path`；不适用逐层探针时用 `--native-only` 进行目标评估。

本工具不声称支持 encoder-decoder、任意多模态输入、任意量化格式或设备不支持的自定义 kernel。
远程代码默认禁用；可信模型按需显式使用 `--trust-remote-code`。

## 安装与设备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

默认自动选择 NPU → CUDA → CPU。`torch_npu` 必须匹配 PyTorch、CANN 和驱动，项目不会自行升级。
如机器有异常卡，显式使用已检查可用的卡：`--device npu:0`。NPU 使用原生权重，不会启用
bitsandbytes；CUDA 小显存量化与 offload 用法见英文 README。当前依赖 Transformers 4.57.x。

## 输入

```json
{"id":"capital_001","prompt":"中国的首都是","target":"北京","tags":["knowledge","zh"]}
{"id":"math_001","prompt":"1 + 1 =","target":"2","tags":["math"]}
```

每行一个独立样本。`target` 可省略，但该样本将只做无监督诊断，不计入 NLL/PPL。
不自动使用聊天模板；需要聊天格式时应固定模型模板和特殊 token，并把格式记录在样本 metadata。

## 分析

```bash
llm-training-evaluator analyze \
  --model /path/to/hf_checkpoint --samples examples/samples.jsonl \
  --device npu:0 --dtype bfloat16 --attn-implementation eager \
  --max-length 2048 --scoring-stride 512 \
  --output outputs/model.json
```

生成 `model.html`、`model.json`、`model.tokens.csv`、`model.layers.csv`。
HTML 包含可靠性图、标签切片、逐 token 损失、逐层轨迹和原始元数据，可离线查看、切换明暗主题、
筛选样本并打印为 PDF。CSV 使用 UTF-8 BOM，便于 Excel 和论文作图工具读取。

```bash
# 不依赖内部层布局；只做原生目标评分，长 prompt 可按滑窗处理
llm-training-evaluator analyze --model /path/to/hf_checkpoint \
  --samples examples/samples.jsonl --native-only --output outputs/native.json

# 更多逐层位置；投影分块限制单次词表投影的内存
llm-training-evaluator analyze --model /path/to/hf_checkpoint \
  --samples examples/samples.jsonl --positions 0,-1 --projection-chunk-size 16 \
  --output outputs/positions.json
```

## 比较检查点

```bash
llm-training-evaluator compare --model-a /path/to/a --model-b /path/to/b \
  --samples examples/samples.jsonl --device npu:0 --output-dir outputs/comparison

llm-training-evaluator compare-results --analysis-a outputs/comparison/model_a.json \
  --analysis-b outputs/comparison/model_b.json --output outputs/recomparison.json
```

两个模型顺序加载。比较前校验 tokenizer、样本集合、target、层/位置和评分协议。
ΔNLL=B−A，负值表示 B 在这些目标上似然更高。95% 区间按样本配对重采样，
至少需要两个独立样本；单个样本不估计区间。多标签切片未做多重比较校正。

通过 `--include-hidden-states --position-mode all` 可启用比较报告的 Linear CKA；
默认关闭 hidden states 导出以节省内存和文件体积。每层至少四个配对观测，最多使用 512 个。

## 学术解释与验证

- 条件 PPL、下一 token accuracy 不等于自由生成任务正确率，且不同 tokenizer 的 PPL 不可直接比较。
- ECE 是有限样本、指定桶数下的下一 token 校准诊断，不能推断整个模型是否可靠。
- Logit Lens 是未训练探针；最终层校验通过不代表中间层预测已经校准，更不是安全提前退出依据。
- 最终层必须与原生 logits 一致，否则会报错；可使用 `--native-only` 保留原生评分能力。
- MoE gate 的 Top-K 不必然等同于实际 dispatch，特殊路由需要专用适配。

论文、公式、边界处理见 [METHODOLOGY](docs/METHODOLOGY.md)；
实测范围和复现步骤见 [VALIDATION](docs/VALIDATION.md)。

```bash
pytest -q
# 在健康 NPU 上运行相同的 24 类架构矩阵
EVALUATOR_TEST_DEVICE=npu:0 pytest -q tests/test_compatibility.py
ruff check src tests
```
