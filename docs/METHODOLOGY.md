# 评估方法、公式与解释边界 / Methodology

## 1. 原生条件续写评分

对每个 JSONL 样本，`prompt` 是条件，`target` 是需要评分的续写。采用 teacher forcing：

\[
\mathrm{NLL}= -\sum_{j\in\mathrm{target}}\log p_\theta(x_j\mid x_{<j}),\qquad
\mathrm{PPL}=\exp\left(\frac{\sum_s\mathrm{NLL}_s}{\sum_s n_s}\right).
\]

- 使用**原生模型 logits 的全词表 FP32 log-softmax**，与 Logit Lens 分开计算。
- 不把 prompt、padding、额外 EOS 计入分母。不会自动添加 EOS 或聊天模板。
- 无 target 的样本不参与有监督汇总；空 target 标为 `empty_target`。
- 总体 NLL 按 token 数加权，PPL 是总体平均 NLL 的指数，**不是每个样本 PPL 的平均值**。
- 汇总 MRR/Top-5 用 `1 + 严格大于目标 logit 的 token 数` 排名；并列采用乐观排名。
  Top-1 accuracy 则使用真实 argmax，故均匀分布下 MRR=1 并不意味着 accuracy=1。
- 极大 NLL 的 PPL 可能超出 JSON 有限浮点范围，此时存 `null`，仍保留 NLL。

### 分词边界

优先对 `prompt+target` 联合分词，用 fast tokenizer 的字符 offsets 确定目标边界。
若某个 token 跨越 prompt/target 边界，则改为分别分词后拼接，避免把 prompt 的一部分当作
目标计分。slow tokenizer 无 offsets 时也分别编码。结果保存 `boundary_policy`、实际
`input_ids` 和 `target_start`。这些策略可能产生不同的 token 序列，跨检查点比较会检查协议。
空 prompt 使用 BOS 作为上下文；缺少 BOS 时明确报错。没有隐式聊天包装。

### 有限上下文与滑窗

`--max-length` 包含条件 token 和目标 token，实际窗口不超过配置声明的上下文上限。
每组最多 `--scoring-stride` 个目标，窗口右对齐，向左保留尽可能多的上下文。
每个目标 token **恰好评分一次**，从前一个位置的 logits 取概率。要求 `1 <= stride < window`。
stride=1 提供窗口范围内的最大左上下文；较大 stride 更快，但不同目标获得的左上下文长度不同。
这不是有 KV cache 的测速，也不是无截断的长文档困惑度。比较实验必须固定窗口和步长。
参考：[Hugging Face, Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)。

## 2. 不确定性与校准

- **Entropy**：全词表预测熵，单位 nats；描述集中程度，不代表任务正确性。
- **Multiclass Brier**：`sum_v (p_v - 1[v=y])²`，实现为等价的稀疏目标公式，不分配 one-hot 矩阵。
  范围约 [0,2]；计算舍入导致的微小负数截为 0。
- **ECE**：把原生 Top-1 confidence 按 `[0,1]` 等宽分为 B 桶；
  `sum_b n_b/N * |accuracy_b - confidence_b|`。默认 B=15，最后一桶包括 1。
  空桶的均值记 null，并保留各桶样本数。HTML 给出可靠性图和完整分桶表。

Guo et al. 的 ECE 定义来自分类校准；本项目将它用于 **teacher-forced 下一 token 分类**，
不是对自由生成答案正确性的校准评估。ECE 受桶数、样本量、词频和数据分布影响；相关 token
不满足独立观测假设，不输出 token 级置信区间。

参考：[Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)。

## 3. 检查点比较与采样不确定性

只比较同 model type、深度、词表、tokenizer 配置及 scoring protocol 的结果。
样本 ID 集合必须一致，不允许重复；prompt、target、tags、实际 token IDs、层/位置集合和目标
边界必须对应。保留 model config hash、tokenizer backend hash、samples hash、库版本和解析出的
Hub revision。**本地目录的 config hash 不等于权重文件内容哈希**，严谨实验还应自行固定权重版本。

`ΔNLL = NLL_B/token - NLL_A/token`，负值倾向于 B。对样本索引进行配对、有放回重采样：
每次抽样保持 A/B 对应关系，保留一个样本内部的全部 token，再计算 token 加权 ΔNLL。
默认 2,000 次，seed=0，报告 2.5%/97.5% 的百分位区间。

- 至少两个独立样本才估计区间；一个样本只报告点估计。
- 假定 JSONL 行是独立抽样单位。同一文档切成多行时，应先按文档组织采样单位。
- 区间不覆盖训练随机性、模型选择偏差或分布偏移。
- 标签切片可重叠，未作多重比较校正；不能凭一次切片显著性宣称总体改进。
- 层级 Top-K 联集概率 L1 是截断描述量，**不是全词表 L1 或 JS**。

参考：[Koehn, Statistical Significance Tests for Machine Translation Evaluation, EMNLP 2004](https://aclanthology.org/W04-3250/)。
本项目借鉴配对重采样原则，将统计量改为条件续写 NLL 差异；不是复刻机器翻译任务指标。

## 4. 逐层诊断

读取 decoder block 输出，重放模型 final norm、可选输出投影、LM head，以及已知的输出
缩放/softcap。最后一层必须与原生 logits 在记录的 rtol/atol 下 allclose，否则终止逐层分析，
避免给出貌似合理但错误的轨迹。可以使用 `--native-only` 继续做架构无关的原生目标评分。

- 相邻层 JS：`0.5 KL(p||m)+0.5 KL(q||m)`，`m=(p+q)/2`，对**全词表**计算，范围 [0,ln2]。
- 稳定起始层：从该层开始，所选位置的 Top-1 一直等于最终层 Top-1，层编号从 0 开始。
- 每个 token 位置独立绘制层轨迹，不跨不同位置连接折线。
- 投影按 `projection_chunk_size` 分块，前一层只保留所选 hidden states；为了避免存储
  layers×positions×vocabulary 的概率张量，计算相邻 JS 时会额外重放一次前层 readout。
- 原生 forward 仍可能生成 sequence×vocabulary logits；分块投影不意味着整次运行常量内存。

本工具使用未训练的 Logit Lens。Belrose et al. 说明原始 Logit Lens 存在坐标系偏移等脆弱性，
提出训练 affine translator 的 Tuned Lens。本项目**未训练 Tuned Lens**；稳定深度、JS 和中间层
置信度只是描述性探针，不证明可提前退出，也不是模型质量分数。
参考：[Belrose et al., Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)。

## 5. MoE 与复现范围

常见 gate/router 可解码输出只提供路由诊断。若读取的是 gate logits，Top-K 不一定等同于经过
分组选择、容量裁剪、专家偏置修正后的真实 dispatch；特殊路由算法需要适配器。
报告中的警告应与数值一起保留。

小型随机权重矩阵测试验证架构调用与数学契约，**不验证预训练模型的任务质量**。
真实权重短样本 smoke 验证加载、NPU 运算和报告流程，**不等于学术基准结果**。
正式评估需要独立、具有代表性的未污染数据，固定预处理、模板、权重版本和运行环境。

## 6. 可选 Linear CKA 表征比较

`--include-hidden-states --position-mode all` 保存每个选定位置的原始 block hidden state。
比较时把同层、同样本、同位置的表示配对，逐特征中心化为 X/Y：

\[
\mathrm{CKA}(X,Y)=\frac{\langle XX^T,YY^T\rangle_F}{\|XX^T\|_F\|YY^T\|_F}.
\]

这是有偏的中心化线性 CKA，对特征的正交变换和各向同性缩放不变；与逐向量余弦互补。
每层最多保留 512 个均匀间隔的配对观测来构造 Gram 矩阵，输出总观测数、使用数和选择方法。
不足四个观测、或者常量表征，不输出数值。该截取是资源限制下的确定性描述，不是随机抽样。
高维、少样本下可能高估相似度；token 相关性、样本选择也影响解释。
CKA 高不代表任务效果更好，也不能证明计算路径具有相同因果功能。

参考：[Kornblith et al., Similarity of Neural Network Representations Revisited, ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)。
