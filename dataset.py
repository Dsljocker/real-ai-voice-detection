# dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
from tensorflow.keras.utils import to_categorical
from features import load_and_pad_audio, extract_features


class AudioDataset(Dataset):
    """训练/测试数据封装：训练时一次性把特征读进内存，测试时按需读取。"""
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.file_paths = []
        self.labels = []
        self._load_paths()

        # 训练模式下先提取好所有特征，后面训练直接按索引取
        self.X = []
        self.y = []
        if self.is_train:
            self._extract_all()

    def _load_paths(self):
        # 按文件夹名收集路径和标签：fake 是 0，real 是 1
        for label_name in ['fake', 'real']:
            folder = os.path.join(self.root_dir, label_name)
            if not os.path.exists(folder):
                print(f"警告: 文件夹不存在 {folder}")
                continue
            label = 0 if label_name == 'fake' else 1
            for file in os.listdir(folder):
                if file.endswith(('.wav', '.mp3', '.flac')):
                    self.file_paths.append(os.path.join(folder, file))
                    self.labels.append(label)
        print(f"[{self.root_dir}] 共找到 {len(self.file_paths)} 个音频文件。")

    def _extract_all(self):
        # 训练集数量不大，直接全部读进内存，训练时不用反复读音频
        for path, label in zip(self.file_paths, self.labels):
            audio = load_and_pad_audio(path)
            if audio is not None:
                feat = extract_features(audio)
                self.X.append(feat)
                self.y.append(label)
        self.X = np.array(self.X)
        # 标签转 one-hot，和模型输出、categorical_crossentropy 对应
        self.y = to_categorical(np.array(self.y), num_classes=2)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.is_train:
            return self.X[idx], self.y[idx]
        else:
            # 测试集按需加载，省内存；文件名也一起返回，后面写 CSV 用
            path = self.file_paths[idx]
            audio = load_and_pad_audio(path)
            feat = extract_features(audio)
            return feat, self.labels[idx], os.path.basename(path)