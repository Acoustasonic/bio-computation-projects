# Project 1 · Protein Secondary Structure Prediction

## 当前状态

本轮仅完成代码实现、配置整理和静态检查。**没有执行数据重建、训练、预测、评估或测试，也没有安装依赖。** 模型效果、运行耗时及标准矩阵的一致性尚待实际运行确认。课程给出的 baseline 参考值约为 Q3=0.676、macro-F1=0.662，不是本轮的实测结果。

任务和数据规则见 `project1_assignment.pdf`、`DATA_DICTIONARY.md`；方法来源及简化见 `REFERENCES.md`。以下命令用于后续运行，本轮没有执行。

## 实现内容

| 部分 | 入口或模块 | 作用 |
|---|---|---|
| 原始数据处理 | `prepare_data.py`、`src/project1/preprocessing.py` | 严格解析、标签映射、传递关系排除、矩阵重建与标准值比较 |
| 模型 | `src/project1/models.py`、`factories.py` | 课程 CNN、扩张残差 CNN、双向 LSTM |
| 训练 | `train.py`、`pipeline.py` | 统一损失、早停、最佳 checkpoint、逐轮曲线与运行记录 |
| 验证集比较 | `analyze_results.py`、`analysis.py` | 统一重算指标，比较边界/内部与长度分组，在两候选中选择最终方法 |
| 最终预测 | `predict.py` | 从已选 checkpoint 重载模型，保持测试 ID 顺序和 -100 padding |
| 格式检查 | `evaluate.py` | validation 指标或 test 格式检查 |
| 提交 | `package_submission.py` | 检查必需工件并按课程白名单打包，不包含课程数据或环境目录 |

### 三个模型

- **课程 baseline (`convolutional`)**：原来的两层 kernel=7 CNN，名义感受野13。保留原模型结构和初始配置。
- **候选一 (`dilated_cnn`)**：kernel=5，扩张率1、2、4、8的四个残差块，名义感受野61；每层掩蔽 PAD 隐状态，避免 padding 区域的偏置向有效位点传播。
- **候选二 (`bilstm`)**：两层双向 LSTM，每方向 hidden=64；按真实长度打包序列，恢复原宽度后逐残基分类。

三个模型只使用同一套氨基酸编码，不使用 profile、模板或预训练模型。输出均为 `[batch, 698, 3]`；PAD=0，UNK=21，H/E/C=0/1/2，损失忽略 -100。两候选采用相同的训练预算与早停规则，并启用梯度裁剪1.0；这些设置均写入各自配置和运行记录。

## 后续运行顺序

先按 `ENVIRONMENT.md` 准备 Python 3.11 和依赖，在本目录运行命令。示例使用 CPU；后续决定用 GPU 时，可将训练及预测命令中的 `--device cpu` 改为 `--device cuda`。

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

保留数量应与课程标准的8084/1057/1102一致，但该结果尚未在本轮运行确认。审计包含原始/保留/排除计数、记录映射、接受边、连通组及 `verification.json`。比较失败时不发布处理结果。输出已存在时默认拒绝覆盖；`--overwrite` 只用于替换本程序先前生成的目录。

### 2. 分别运行 baseline 和两候选

```text
python train.py --config configs/baseline.json --data-dir processed_data --output-dir outputs/runs/baseline --device cpu
python train.py --config configs/dilated_cnn.json --device cpu
python train.py --config configs/bilstm.json --device cpu
```

同一实验的配置、数据划分、随机种子及选择指标应固定，不能根据 test 修改参数。三个默认配置均为 seed=42、batch=64、AdamW、learning rate=0.001、最多8轮、patience=2，按验证集 Q3 选择 checkpoint。

每次运行生成 `model.pt`、`config.json`、`metrics.json`、`history.json/csv`、`validation_predictions.npz`、`run_manifest.json`。后者只有在运行完成后才写入 `status=complete`，并记录环境和实际命令。已有运行工件默认不覆盖。

课程原入口 `starter/run_baseline.py` 仍保留；其默认输出下也保留 `baseline_validation.npz`、`baseline_model.pt` 等旧文件名。上面的三方法比较使用统一的新工件名称。

### 3. 比较并选择候选

```text
python analyze_results.py --data processed_data --baseline outputs/runs/baseline --candidate outputs/runs/dilated_cnn --candidate outputs/runs/bilstm --output-dir outputs/comparison
```

三者使用同一 validation ID 顺序、真实残基及 H/E/C 指标。输出比较表、混淆矩阵、逐蛋白指标、长度分组、训练/验证曲线和边界分析。

边界定义为真实标签发生相邻转移的两个残基，各向外扩展2个位置并截断到序列内；其余为内部。长度分组为≤100、101–200、201–400、>400。空子组的指标为 `null`，不作为0分。

最终只在两候选中按 Q3 选择，平分时依次比较 macro-F1 和模型名称；`comparison.csv` 标记选中的方法，`selection.json` 保存选择。即使候选低于 baseline，也应如实报告。

### 4. 用已选模型生成测试预测

```text
python predict.py --selection outputs/comparison/selection.json --data processed_data --split test --output predictions.npz --device cpu
python evaluate.py --data processed_data --split test --pred predictions.npz
```

预测文件仅含 Unicode `ids` 与整数 `q3` 两数组；测试顺序为课程发布的1102条，形状为1102×698，有效位置0/1/2，padding为-100。另写 `predictions.metadata.json` 记录 checkpoint 哈希和预测命令。test 检查只验证格式，不产生 test 分数。

可用 `python evaluate.py --data processed_data --split validation --pred outputs/runs/bilstm/validation_predictions.npz` 单独检查验证集结果。

### 5. 完成真实报告后打包

先根据运行结果撰写 `report.pdf`，再执行：

```text
python package_submission.py --data processed_data --selection outputs/comparison/selection.json --pred predictions.npz --report report.pdf --student-id 2023011618 --name 杜一郎 --output-dir submissions
```

打包要求数据审计通过、三次运行完成、验证集比较和候选选择存在、测试预测格式正确。只保留所选模型，排除课程原始数据、标准矩阵、处理矩阵、虚拟环境和下载文件；缺少真实工件时会报错。

归档后的模型位于 `model/model.pt`，可用 `predict.py --checkpoint model/model.pt --data <课程数据路径> --split test --output predictions_reloaded.npz --device cpu` 重载。重做原始处理时需另行提供课程的 `source_data/` 和 `standard_data/`。

## 本轮尚待执行的检查

原始数据重建能否逐项匹配标准矩阵、模型 forward/反向传播、padding 行为、实际训练效果、设备兼容性、预测格式与提交 ZIP 的运行核验，均留待后续运行阶段确认。当前不提供虚构的成绩、checkpoint、测试预测或实验报告。
