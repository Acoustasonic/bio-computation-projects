# Project 1 · Protein Secondary Structure Prediction

## 当前状态

已于2026-11-24完成原始数据重建及三个模型各5个seed的实验，共15次运行。固定seed为42、43、44、45、46，其中42复用原结果，其余12次另行训练。重建矩阵与课程标准逐项一致；测试集1102条序列、244922个有效残基的预测通过 `FORMAT_VALID` 检查。测试标签未提供，因此不报告测试集准确率。

下表为五个seed的均值±样本标准差（`ddof=1`），各项均以百分数表示；标准差的数值单位为百分点。

| 方法 | validation Q3（%） | macro-F1（%） | H F1（%） | E F1（%） | C F1（%） |
|---|---:|---:|---:|---:|---:|
| 课程 CNN baseline | 67.44 ± 0.11 | 66.00 ± 0.21 | 71.53 ± 0.21 | 57.40 ± 0.74 | 69.07 ± 0.30 |
| 扩张残差 CNN | 69.05 ± 0.20 | 67.97 ± 0.34 | 73.45 ± 0.47 | 60.97 ± 0.89 | 69.49 ± 0.38 |
| BiLSTM（最终选择） | 69.94 ± 0.18 | 68.96 ± 0.22 | 75.10 ± 0.22 | 62.55 ± 0.66 | 69.23 ± 0.34 |

按五次验证Q3均值选择BiLSTM，较baseline平均提升2.50个百分点。最终权重使用预先固定的seed42，不从五次中挑最高分，也不使用ensemble。均值和SD只描述固定数据划分下训练随机性的变化。

实际环境为 Ubuntu 24.04、Python 3.11.18、PyTorch 2.5.1 / CUDA 12.4、单张 RTX 4090；使用GPU0和8个计算线程。seed42训练提交为 `c3538979a95d348d54256d8a51c28c45c3ec5585`，补跑提交为 `2c65968461749996ef39fb7ca7b343e70527a9bb`；训练核心、模型和原配置未改变。

任务和数据规则见课程单独提供的 `project1_assignment.pdf`、`DATA_DICTIONARY.md`；方法来源及简化见 `REFERENCES.md`。Git 仓库仅同步代码和文档，课程数据另行提供；提交 ZIP 另含报告、运行证据、已选模型和预测。

## 实现内容

| 部分 | 入口或模块 | 作用 |
|---|---|---|
| 原始数据处理 | `prepare_data.py`、`src/project1/preprocessing.py` | 严格解析、标签映射、传递关系排除、矩阵重建与标准值比较 |
| 模型 | `src/project1/models.py`、`factories.py` | 课程 CNN、扩张残差 CNN、双向 LSTM |
| 训练 | `train.py`、`pipeline.py` | 统一损失、早停、最佳 checkpoint、逐轮曲线与运行记录 |
| 五种子实验 | `run_multiseed.py`、`analyze_multiseed.py` | 固定42–46五个seed，复用seed42，汇总均值、样本标准差和配对差值 |
| 验证集比较 | `analyze_results.py`、`analysis.py` | 统一重算指标，比较边界/内部与长度分组，在两候选中选择最终方法 |
| 最终预测 | `predict.py` | 从已选 checkpoint 重载模型，保持测试 ID 顺序和 -100 padding |
| 格式检查 | `evaluate.py` | validation 指标或 test 格式检查 |
| 提交 | `package_submission.py` | 检查必需工件并按课程白名单打包，不包含课程数据或环境目录 |

### 三个模型

- **课程 baseline (`convolutional`)**：原来的两层 kernel=7 CNN，名义感受野13。保留原模型结构和初始配置。
- **候选一 (`dilated_cnn`)**：kernel=5，扩张率1、2、4、8的四个残差块，名义感受野61；每层掩蔽 PAD 隐状态，避免 padding 区域的偏置向有效位点传播。
- **候选二 (`bilstm`)**：两层双向 LSTM，每方向 hidden=64；按真实长度打包序列，恢复原宽度后逐残基分类。

三个模型只使用同一套氨基酸编码，不使用 profile、模板或预训练模型。输出均为 `[batch, 698, 3]`；PAD=0，UNK=21，H/E/C=0/1/2，损失忽略 -100。两候选采用相同的训练预算与早停规则，并启用梯度裁剪1.0；这些设置均写入各自配置和运行记录。

## 复现运行顺序

先按 `ENVIRONMENT.md` 准备 Python 3.11 和依赖，在干净的代码目录运行命令。提交包已有运行证据，重新训练时应将代码和配置复制到新的工作目录，另行提供课程数据；不要直接覆盖归档的 `outputs/`。以下给出CPU兼容命令；本次云端实跑将训练及预测命令中的 `--device cpu` 改为 `--device cuda`，并设置 `CUDA_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=8`、`OPENBLAS_NUM_THREADS=8`、`MKL_NUM_THREADS=8`。

### 1. 重建数据并验证

```text
python prepare_data.py
```

该命令从 `source_data/` 决定保留记录，之后才与 `standard_data/` 对照，输出 `processed_data/` 和 `outputs/data_audit/`。处理规则为：

- sequence 只接受单元素 bytes-list，移除 ASCII 空白并转大写；ss3 单独解析。
- 源标签 C/H/E=0/1/2 映射为课程 H/E/C=0/1/2。
- identity 分母包括所有对齐列；coverage 使用完整短序列长度。
- 按题面固定阈值建无向图，取传递连通分量；跨 split 只保留 `test > validation > train` 最高优先级成员。
- 在源顺序上编码并右填充到698；不读取 test 标签。

本次保留数量为8084/1057/1102，与课程标准一致；分别排除719/45/0条记录。审计包含原始/保留/排除计数、记录映射、接受边、连通组及 `verification.json`。比较失败时不发布处理结果。输出已存在时默认拒绝覆盖；`--overwrite` 只用于替换本程序先前生成的目录。

### 2. 分别运行 baseline 和两候选

```text
python train.py --config configs/baseline.json --data-dir processed_data --output-dir outputs/runs/baseline --device cpu
python train.py --config configs/dilated_cnn.json --device cpu
python train.py --config configs/bilstm.json --device cpu
```

同一实验的配置、数据划分、随机种子及选择指标应固定，不能根据 test 修改参数。三个默认配置均为 seed=42、batch=64、AdamW、learning rate=0.001、最多8轮、patience=2，按验证集 Q3 选择 checkpoint。

每次运行生成 `model.pt`、`config.json`、`metrics.json`、`history.json/csv`、`validation_predictions.npz`、`run_manifest.json`。后者只有在运行完成后才写入 `status=complete`，并记录环境和实际命令。已有运行工件默认不覆盖。

课程原入口 `starter/run_baseline.py` 仍保留；其默认输出下也保留 `baseline_validation.npz`、`baseline_model.pt` 等旧文件名。上面的三方法比较使用统一的新工件名称。

### 3. 补充五个 seed 并汇总比较

上一步生成三个seed42运行，之后执行：

```text
python run_multiseed.py --data processed_data --runs-root outputs --device cpu
python analyze_multiseed.py --data processed_data --runs-root outputs --output-dir outputs/multiseed_summary
```

固定seed为42、43、44、45、46，三个方法使用相同seed集合。入口核对已完成seed42，只补跑其余12次；模型结构、优化器、最多8轮、早停和数据划分均保持原设置。先写 `outputs/multiseed/run_plan.json`，逐次保存配置、权重、逐轮结果、验证预测及完成记录。遇到不完整或不一致的已有工件时会停止，供人工检查。

汇总输出 `per_seed.csv/json`（15次运行）、`summary.csv/json`（五项指标的均值与样本标准差）、`paired_differences.csv`（同seed减去baseline）及均值±SD图。标准差使用 `ddof=1`，描述随机初始化和训练顺序带来的波动，不是置信区间，也不反映更换数据划分的变化。

只在两候选方法中按五次验证Q3均值选择，平分时比较macro-F1均值，再按模型名称排序。最终checkpoint固定为该方法seed42，不挑最高分seed，不使用ensemble；主选择文件为 `outputs/multiseed_summary/selection.json`。

另外保留seed42的细节分析：

```text
python analyze_results.py --data processed_data --baseline outputs/runs/baseline --candidate outputs/runs/dilated_cnn --candidate outputs/runs/bilstm --output-dir outputs/comparison
```

三者使用同一 validation ID 顺序、真实残基及 H/E/C 指标。输出比较表、混淆矩阵、逐蛋白指标、长度分组、训练/验证曲线和边界分析。

边界定义为真实标签发生相邻转移的两个残基，各向外扩展2个位置并截断到序列内；其余为内部。长度分组为≤100、101–200、201–400、>400。空子组的指标为 `null`，不作为0分。

此处 `outputs/comparison/` 的表和图只描述预先指定的seed42；其中单seed选择记录不覆盖五seed最终选择。即使候选低于baseline，也应如实报告。

### 4. 用已选模型生成测试预测

```text
python predict.py --selection outputs/multiseed_summary/selection.json --data processed_data --split test --output predictions.npz --device cpu
python evaluate.py --data processed_data --split test --pred predictions.npz
```

预测文件仅含 Unicode `ids` 与整数 `q3` 两数组；测试顺序为课程发布的1102条，形状为1102×698，有效位置0/1/2，padding为-100。另写 `predictions.metadata.json` 记录 checkpoint 哈希和预测命令。test 检查只验证格式，不产生 test 分数。

可用 `python evaluate.py --data processed_data --split validation --pred outputs/runs/bilstm/validation_predictions.npz` 单独检查验证集结果。

### 5. 完成真实报告后打包

先根据运行结果撰写 `report.pdf`，再执行：

```text
python package_submission.py --data processed_data --selection outputs/multiseed_summary/selection.json --pred predictions.npz --report report.pdf --student-id 2023011618 --name 杜一郎 --output-dir submissions
```

五seed打包要求数据审计通过、15次运行完成且同方法配置仅seed/output路径不同，验证指标可由预测重算、汇总与固定选择协议一致，以及测试预测metadata与选定模型哈希一致。归档保留15次运行的配置、指标、曲线数据和验证预测，只包含最终所选seed42模型权重；排除课程数据、虚拟环境和下载文件。缺少真实工件时会报错。

归档后的模型位于 `model/model.pt`，可用 `python predict.py --selection outputs/multiseed_summary/selection.json --data <课程数据路径> --split test --output predictions_reloaded.npz --device cpu` 重载。重做原始处理时需另行提供课程的 `source_data/` 和 `standard_data/`。重新训练其余14个模型时应使用干净目录；提交包保留其指标和预测，不包含这些非最终模型的权重。

## 已完成的核验

`outputs/data_audit/verification.json` 记录矩阵逐项一致性；15次训练的 `run_manifest.json` 记录已完成状态、源码版本、实际环境及模型哈希；`outputs/comparison/` 保存统一重算的验证指标和图。`predictions.metadata.json`（提交包内位于 `outputs/`）记录最终模型与测试预测哈希。打包程序对每个归档文件回读校验，并生成 `submission_manifest.json` 与 ZIP 的 `.sha256` 文件。
