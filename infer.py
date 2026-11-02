# infer.py
import os
import csv
import numpy as np
from keras.models import load_model
from tqdm import tqdm

from config import TEST_DATA_DIR, MODEL_PATH, OUTPUT_DIR
from features import load_and_pad_audio, extract_features


def predict_folder(folder_path, model, true_label, csv_path):
    """预测一个文件夹里的所有音频，返回 real/fake 数量和总数，同时把明细写进 CSV。"""
    files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac'))]
    if not files:
        print(f"[{folder_path}] 未找到音频文件。")
        return 0, 0, 0

    n_real, n_fake = 0, 0
    rows = []

    with tqdm(total=len(files), desc=f"Testing {os.path.basename(folder_path)}") as pbar:
        # 一条一条预测，进度条方便看还剩多少
        for file in files:
            path = os.path.join(folder_path, file)
            audio = load_and_pad_audio(path)
            if audio is None:
                pbar.update(1)
                continue

            feat = extract_features(audio)
            # LSTM 要按 batch 输入，这里补一个维度：(98, 15) -> (1, 98, 15)
            feat = np.expand_dims(feat, axis=0)

            pred = model.predict(feat, verbose=0)
            # 输出两个概率：第 0 位是 fake，第 1 位是 real
            score = pred[0][1]  # real 的概率 (0是fake, 1是real)
            # 0.5 是分界线；阈值固定，不参与训练
            label = 'real' if score >= 0.5 else 'fake'

            if label == 'real':
                n_real += 1
            else:
                n_fake += 1

            rows.append([file, label, f"{score:.4f}"])
            pbar.update(1)

    # 保存明细 CSV：文件名、预测标签、real 概率
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label', 'real_prob'])
        writer.writerows(rows)

    total = n_real + n_fake
    print(f"  -> {folder_path} 测试完毕, 共 {total} 条, 判定 real={n_real}, fake={n_fake}")
    return n_real, n_fake, total