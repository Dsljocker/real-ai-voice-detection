# 语音鉴真 —— 基于 LSTM 的 AI 语音真伪识别

一个可以直接跑通的「真实人声 vs AI 合成语音」二分类项目：把音频切成固定长度，提取过零率（ZCR）、均方根能量（RMS）和 13 维 MFCC，共 15 维声学特征，再送入两层 LSTM 判断这段语音是 **real（真人）** 还是 **fake（AI 伪造）**。

项目已经包含：

- 800 条训练音频（real / fake 各 400 条）
- 200 条测试音频（real / fake 各 100 条）
- 已训练好的模型 `modules/best_lstm.h5`
- 训练脚本、批量测试脚本、CSV 报告和训练曲线

也就是说，**即使不训练，也可以直接运行测试脚本看到结果**。

---

## 一、项目效果

使用仓库中当前模型（重新训练后的 `modules/best_lstm.h5`）对 `test_data` 做测试，结果保存在 `output/all_result.csv`：

| 测试子集 | 样本数 | 判为 real | 判为 fake | 判定正确 |
| :--- | ---: | ---: | ---: | ---: |
| `fake` | 100 | 7 | 93 | 93 |
| `real` | 100 | 94 | 6 | 94 |
| **合计** | **200** | **101** | **99** | **187（93.5%）** |

> 说明：以上是当前数据、当前模型在测试集上的结果，只能说明项目流程可用，不代表在任意真实场景下都有同样准确率。

---

## 二、功能特点

- **端到端流程**：音频解码 → 去静音 → 定长补齐/截断 → 特征提取 → LSTM 训练 → 批量测试 → CSV 报告。
- **特征缓存**：首次训练会提取特征并保存为 `cached_features.npz`，之后训练可直接加载，省去重复计算。
- **训练保护**：使用 `EarlyStopping` + `ModelCheckpoint`，自动保留验证集上表现最好的模型。
- **结果可追溯**：测试会为每个子文件夹生成明细 CSV，并额外生成一份汇总 CSV。
- **开箱即用**：数据、模型、依赖文件齐全，适合课程设计、入门学习和二次开发。

---

## 三、项目结构

```text
Speak/
├─ config.py                 # 路径与超参数配置
├─ features.py               # 音频加载、去静音、补齐/截断、特征提取
├─ dataset.py                # 数据集封装（训练时一次性提取特征）
├─ model.py                  # 两层 LSTM 模型定义
├─ train.py                  # 训练入口
├─ infer.py                  # 单个文件夹的批量预测
├─ main.py                   # 测试入口（评估 test_data 并生成汇总）
├─ requirements.txt          # Python 依赖
├─ cached_features.npz       # 已提取好的训练特征缓存（800 条）
├─ modules/
│  └─ best_lstm.h5           # 已训练好的 LSTM 模型
├─ output/
│  ├─ all_result.csv         # 测试汇总
│  ├─ fake_result.csv        # fake 文件夹逐条预测结果
│  ├─ real_result.csv        # real 文件夹逐条预测结果
│  └─ loss_curves.png        # 训练准确率 / 损失曲线
├─ train_data/
│  ├─ fake/                  # 400 条伪造语音
│  ├─ real/                  # 400 条真实语音
│  └─ labels.csv             # 数据来源记录（代码不读取）
└─ test_data/
   ├─ fake/                  # 100 条伪造语音
   └─ real/                  # 100 条真实语音
```

---

## 四、环境要求

### 4.1 基础环境

| 项目 | 要求 |
| :--- | :--- |
| 操作系统 | Windows 10/11、Linux、macOS 均可（本项目在 Windows 上验证） |
| Python | **3.9**（验证版本 3.9.13；不建议用 3.11+，TensorFlow 2.10 兼容性较差） |
| 内存 | 建议 8 GB 以上，最低 4 GB |
| 硬盘 | 预留约 2 GB（TensorFlow、PyTorch 等依赖较大） |
| GPU | **不强制**，CPU 即可完成训练和测试 |
| ffmpeg | 建议安装（读取 mp3/flac 时需要；本项目 wav 数据可先不装） |

### 4.2 Python 依赖

完整版本见 [`requirements.txt`](requirements.txt)：

| 包 | 版本 | 用途 |
| :--- | :--- | :--- |
| tensorflow | 2.10.0 | 构建、训练、加载 LSTM 模型 |
| keras | 2.10.0 | 模型层与训练接口（与 TF 2.10 配套） |
| torch | 2.5.1 | `dataset.py` 继承 `torch.utils.data.Dataset` |
| librosa | 0.10.2.post1 | ZCR、RMS、MFCC 特征提取 |
| pydub | 0.25.1 | 读取 wav/mp3/flac 音频 |
| soundfile | 0.13.1 | 音频读写后端 |
| numpy | 1.26.4 | 数组与数值计算 |
| scipy | 1.10.1 | 科学计算（librosa 依赖） |
| scikit-learn | 1.3.2 | 划分训练集 / 验证集 |
| numba | 0.59.1 | librosa 的 JIT 加速后端 |
| llvmlite | 0.42.0 | numba 的 LLVM 后端 |
| matplotlib | 3.8.2 | 绘制训练曲线 |
| tqdm | 4.70.1 | 测试进度条 |

---

## 五、新手操作步骤（从零开始）

> 以下命令需要在**项目根目录**（能看到 `train.py`、`main.py` 的那一层）执行。
> 示例路径为 `F:\shenduxuexi\Speak`，请换成你自己的实际路径。

### 第 1 步：确认已经安装 Python 3.9

打开 PowerShell（或 Anaconda Prompt），输入：

```powershell
python --version
```

看到 `Python 3.9.x` 即可。如果没有安装，推荐安装 Anaconda 或 Miniconda，再创建 Python 3.9 环境。

### 第 2 步：进入项目目录

```powershell
cd F:\shenduxuexi\Speak
```

确认当前目录下有 `train.py`：

```powershell
Get-ChildItem
```

### 第 3 步：创建并激活虚拟环境

推荐使用 conda（二选一）：

```powershell
conda create -n speak python=3.9 -y
conda activate speak
```

如果使用 Python 自带的 venv：

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> 如果 PowerShell 提示“禁止运行脚本”，先执行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> 然后重新激活环境。

激活成功后，命令行前面通常会出现 `(speak)` 或 `(.venv)`。

### 第 4 步：升级 pip 并安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

这一步会下载 TensorFlow、PyTorch 等较大的包，第一次安装可能需要几分钟到十几分钟，请耐心等待。

安装完成后验证：

```powershell
python -c "import tensorflow as tf, keras, librosa, torch; print(tf.__version__, keras.__version__, librosa.__version__, torch.__version__)"
```

正常应输出类似：

```text
2.10.0 2.10.0 0.10.2.post1 2.5.1+cpu
```

（安装的是 CPU 版 PyTorch 时，版本号会带 `+cpu`；只要能看到 `2.5.1` 即可。）

### 第 5 步：安装 ffmpeg（建议）

如果只用项目自带的 `.wav` 文件，可以先跳过；如果之后要测试 `.mp3`、`.flac`，建议安装。

Windows 推荐用 winget：

```powershell
winget install Gyan.FFmpeg
```

或者用 conda：

```powershell
conda install -c conda-forge ffmpeg -y
```

安装后**重新打开终端**，检查：

```powershell
ffmpeg -version
```

能输出版本信息即可。

### 第 6 步：检查数据是否完整

```powershell
Get-ChildItem train_data\real, train_data\fake, test_data\real, test_data\fake -File | Group-Object DirectoryName | Select-Object Name, Count
```

应能看到四个文件夹，每个文件夹的音频数分别为 400、400、100、100。

音频格式要求：

- 支持 `.wav`、`.mp3`、`.flac`
- 训练和测试音频建议为单声道、16 kHz 左右
- 当前数据为 16 kHz 单声道 WAV，时长约 5～8 秒

### 第 7 步：先直接测试（推荐第一次这样体验）

仓库里已经有训练好的 `modules/best_lstm.h5`，因此**不需要先训练**，直接运行：

```powershell
python main.py
```

脚本会：

1. 加载 `modules/best_lstm.h5`
2. 读取 `test_data/fake` 和 `test_data/real`
3. 对每条音频预测并保存明细 CSV
4. 在 `output/` 下生成 `all_result.csv`

运行结束后会打印类似：

```text
所有测试完毕，汇总结果已保存至: ...\output\all_result.csv
  真实文件夹: fake | 总数: 100 | 预测 real:   7 | 预测 fake:  93
  真实文件夹: real | 总数: 100 | 预测 real:  94 | 预测 fake:   6
```

### 第 8 步：重新训练模型（可选）

如果你想体验完整训练流程，执行：

```powershell
python train.py
```

训练流程会：

1. 优先读取 `cached_features.npz`（已存在，所以启动很快）
2. 按 8:2 划分训练集 / 验证集
3. 构建两层 LSTM 并训练最多 30 轮
4. 通过 EarlyStopping 提前停止，并保存验证集最优模型
5. 绘制准确率 / 损失曲线

产物：

| 文件 | 说明 |
| :--- | :--- |
| `modules/best_lstm.h5` | 最优模型（会覆盖旧模型） |
| `output/loss_curves.png` | 训练过程曲线 |

> 注意：`train.py` 会覆盖现有的 `best_lstm.h5`。如果想保留仓库自带的模型，请先备份：
> `Copy-Item modules\best_lstm.h5 modules\best_lstm.backup.h5`

如果删除了 `cached_features.npz`，第一次训练会重新提取 800 条音频的特征，耗时几分钟；之后再训练就会直接读缓存。

### 第 9 步：查看结果

用 Excel 或记事本打开：

- `output/all_result.csv`：汇总（每个文件夹总数、预测为 real/fake 的数量）
- `output/fake_result.csv`：`fake` 文件夹逐条结果
- `output/real_result.csv`：`real` 文件夹逐条结果
- `output/loss_curves.png`：训练曲线

逐条 CSV 的列含义：

| 列名 | 含义 |
| :--- | :--- |
| `id` | 音频文件名 |
| `label` | 模型判定结果：`real` 或 `fake` |
| `real_prob` | 模型认为它是真人语音的概率，范围 0～1，≥ 0.5 判为 real |

### 第 10 步：预测单条音频（可选）

如果想拿自己的音频试一下，可以在项目根目录新建一个临时脚本（例如 `try_one.py`），写入：

```python
import numpy as np
from keras.models import load_model

from config import MODEL_PATH
from features import load_and_pad_audio, extract_features

model = load_model(MODEL_PATH)

audio = load_and_pad_audio(r"test_data\real\test_real_0002.wav")
features = np.expand_dims(extract_features(audio), axis=0)
real_prob = float(model.predict(features, verbose=0)[0][1])

print("real" if real_prob >= 0.5 else "fake", f"{real_prob:.4f}")
```

把路径换成你的音频路径，然后运行：

```powershell
python try_one.py
```

> 第一次预测前会加载模型，可能等待几秒。`real_prob` 越接近 1，模型越倾向认为是真人；越接近 0，越倾向认为是伪造。

---

## 六、数据准备说明

### 6.1 目录约定

代码**只根据文件夹名判断标签**：

```text
train_data/
├─ fake/     # 标签 0
└─ real/     # 标签 1

test_data/
├─ fake/
└─ real/
```

- `fake` 对应标签 `0`
- `real` 对应标签 `1`
- 文件名不影响标签，只要扩展名是 `.wav` / `.mp3` / `.flac` 即可
- `train_data/labels.csv` 只是数据来源记录，训练代码不会读取它

### 6.2 使用自己的数据

1. 把真实语音放进 `train_data/real/`，伪造语音放进 `train_data/fake/`
2. 测试数据同理放入 `test_data/`
3. **删除旧的 `cached_features.npz`**（否则会继续使用旧特征）
4. 重新运行 `python train.py`
5. 再运行 `python main.py` 测试

建议训练集保持 real / fake 数量大致平衡，否则模型容易偏向数量多的一类。

---

## 七、配置说明（`config.py`）

| 参数 | 当前值 | 作用 | 备注 |
| :--- | :--- | :--- | :--- |
| `TRAIN_DATA_DIR` | `train_data` | 训练数据目录 | 实际使用 |
| `TEST_DATA_DIR` | `test_data` | 测试数据目录 | 实际使用；不要改成不存在的名字 |
| `MODEL_PATH` | `modules/best_lstm.h5` | 模型保存/加载路径 | 实际使用 |
| `OUTPUT_DIR` | `output` | 结果输出目录 | 实际使用 |
| `LOSS_CURVE_PATH` | `output/loss_curves.png` | 训练曲线路径 | 实际使用 |
| `BATCH_SIZE` | 32 | 训练批大小 | 实际使用；内存不足时可调小 |
| `EPOCHS` | 30 | 最大训练轮数 | 实际使用；可能被 EarlyStopping 提前终止 |
| `LEARNING_RATE` | 0.001 | 学习率 | **当前未接入训练代码**，模型使用 RMSProp 默认学习率 |
| `SAMPLE_RATE` | 44100 | 采样率记录 | **当前未被特征代码读取** |
| `DURATION` | 2 | 时长记录 | **当前未使用** |
| `AUDIO_LENGTH` | 88200 | 音频长度记录 | **当前未使用** |

特征提取的实际参数写在 `features.py` 中：

- 固定长度 `target_length = 50000` 个采样点
- 帧长 `2048`，帧移 `512`
- 13 维 MFCC，`sr=44100`

因此在当前数据上，每条音频最终得到 **98 帧 × 15 维** 的特征矩阵（1 维 ZCR + 1 维 RMS + 13 维 MFCC）。

---

## 八、工作原理

```text
音频文件（wav / mp3 / flac）
        │  pydub 解码
        ▼
去除前后静音（librosa.effects.trim, top_db=25）
        │
        ▼
补齐或截断到 50,000 个采样点
        │
        ▼
提取特征：ZCR(1) + RMS(1) + MFCC(13)
        │
        ▼
特征序列：98 帧 × 15 维
        │
        ▼
LSTM(64, return_sequences=True) → Dropout(0.3)
        │
        ▼
LSTM(64) → Dropout(0.3)
        │
        ▼
Dense(2, softmax)
        │
        ▼
fake(0) / real(1)
```

模型结构（共 53,634 个参数）：

| 层 | 输出形状 | 参数量 |
| :--- | :--- | ---: |
| LSTM(64, return_sequences=True) | (None, 98, 64) | 20,480 |
| Dropout(0.3) | (None, 98, 64) | 0 |
| LSTM(64) | (None, 64) | 33,024 |
| Dropout(0.3) | (None, 64) | 0 |
| Dense(2, softmax) | (None, 2) | 130 |

训练配置：

- 损失函数：`categorical_crossentropy`
- 优化器：`RMSProp`
- 评估指标：`accuracy`
- 训练集 / 验证集：8:2
- 早停：验证损失连续 10 轮不下降则停止，并恢复最优权重

---

## 九、输出文件说明

### `output/all_result.csv`

```csv
folder,total,pred_real,pred_fake
fake,100,7,93
real,100,94,6
```

### `output/fake_result.csv` / `output/real_result.csv`

```csv
id,label,real_prob
test_fake_0001.wav,fake,0.1594
test_fake_0005.wav,fake,0.0064
```

- `label` 是模型最终判定
- `real_prob` 是判为 real 的概率
- 模型阈值固定为 0.5

---

## 十、常见问题（FAQ）

### 1. 报错 `ModuleNotFoundError: No module named 'tensorflow'`

通常是虚拟环境没有激活，或用了另一个 Python。检查：

```powershell
where python
python -c "import sys; print(sys.executable)"
```

确认输出的是当前虚拟环境中的 Python，然后重新执行：

```powershell
conda activate speak
python -m pip install -r requirements.txt
```

### 2. 报错 `Couldn't find ffmpeg or avconv`

说明 `pydub` 找不到 ffmpeg。安装 ffmpeg 后**重新打开终端**，并确认 `ffmpeg -version` 能正常输出。只用 wav 时可以先忽略。

### 3. 报错与 Keras 3 有关，或提示 `keras` 版本不兼容

本项目要求 Keras 2.x（与 TensorFlow 2.10 配套）。检查：

```powershell
python -m pip show keras
```

如果版本是 3.x，建议删除环境重新按 `requirements.txt` 安装，不要单独安装最新版 Keras。

### 4. 提示 `错误: 模型文件不存在`

说明 `modules/best_lstm.h5` 不存在。先运行 `python train.py` 训练模型，或从备份中恢复该文件。

### 5. 提示 `错误：未提取到任何特征数据`

检查：

- `train_data/real` 和 `train_data/fake` 是否存在
- 音频扩展名是否为 `.wav` / `.mp3` / `.flac`
- 音频是否能正常播放、是否损坏

### 6. 换了数据，但训练结果没变化

删除特征缓存后重试：

```powershell
Remove-Item cached_features.npz
python train.py
```

### 7. 运行时出现 `Could not load dynamic library 'cudart64_110.dll'`

这是 TensorFlow 在找 CUDA 库。没有 NVIDIA GPU 时属于正常提示，**可以忽略**，CPU 仍能正常训练和预测。

### 8. 内存不足 / 训练很慢

- 把 `config.py` 中的 `BATCH_SIZE` 从 32 调小（如 16 或 8）
- 关闭其他占用内存的程序
- 减少训练数据量
- 没有 GPU 时训练较慢是正常现象

### 9. 准确率不理想

- 训练集和测试集最好来自同一批数据、同样的采样率和格式
- 保持 real / fake 数量平衡
- 数据量越大，模型通常越稳定
- 当前项目是入门级方案，复杂场景建议改用更大的声学模型或预训练语音模型

### 10. 报错 `SystemError: initialization of _internal failed without raising an exception`

这是 **numba 与 numpy 版本不匹配** 导致的，通常会出现在调用 `librosa.effects.trim` 时。旧环境中的 numba 0.55.x 与 numpy 1.26.x 不兼容。

解决方式（推荐按顺序尝试）：

1. 按本 README 新建一个干净环境，重新执行 `pip install -r requirements.txt`（已锁定 `numba==0.59.1`、`llvmlite==0.42.0`）。
2. 若在旧环境中直接升级 numba，pip 可能报 `Cannot uninstall llvmlite`，此时建议新建环境，或使用 conda 更新：

   ```powershell
   conda install -c conda-forge "numba>=0.59" "llvmlite>=0.42" -y
   ```

3. 确认版本：

   ```powershell
   python -c "import numba, llvmlite, numpy; print(numba.__version__, llvmlite.__version__, numpy.__version__)"
   ```

   应看到 `0.59.1 0.42.0 1.26.4`（或更高但互相兼容的版本）。

---

## 十一、已知限制

- 当前测试集与训练集同源，93.5% 只能说明项目流程有效，**不代表跨数据集、跨说话人、跨语言的泛化能力**。
- 特征提取参数（`target_length=50000`、MFCC 的 `sr=44100`）写死在 `features.py` 中；当前数据实际为 16 kHz，如果更换数据集后效果异常，建议优先统一采样率与特征参数。
- `config.py` 中的 `LEARNING_RATE`、`SAMPLE_RATE`、`DURATION`、`AUDIO_LENGTH` 目前没有真正接入训练/特征流程，修改它们不会生效。
- 模型输入被训练为固定的 98 帧，因此所有音频必须先经过 `load_and_pad_audio` 处理。
- 本项目仅用于学习、课程设计和研究演示，**不适合直接用于生产环境的鉴伪系统**。

---

## 十二、快速命令备忘

```powershell
# 1. 进入项目
cd F:\shenduxuexi\Speak

# 2. 创建并激活环境（以 conda 为例）
conda create -n speak python=3.9 -y
conda activate speak

# 3. 安装依赖
python -m pip install -r requirements.txt

# 4. 直接测试（使用自带模型）
python main.py

# 5. 重新训练（可选，会覆盖现有模型）
python train.py

# 6. 查看输出
Get-ChildItem output
```
