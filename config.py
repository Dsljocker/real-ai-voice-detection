# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "train_data")
# 测试数据目录；如果改了文件夹名，这里要同步改
TEST_DATA_DIR  = os.path.join(BASE_DIR, "test_data")

# 模型与输出路径
MODULE_DIR = os.path.join(BASE_DIR, "modules")
MODEL_PATH = os.path.join(MODULE_DIR, "best_lstm.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, "loss_curves.png")

# 训练超参数
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001  # 当前训练代码没有显式传入，RMSProp 用的是默认学习率

# 下面几个音频参数目前只作记录，实际取值在 features.py 里
SAMPLE_RATE = 44100
DURATION = 2  # 秒
AUDIO_LENGTH = 88200  # 实际截断长度是 features.py 里的 50000