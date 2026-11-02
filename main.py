# main.py
import os
import csv
from keras.models import load_model

from config import TEST_DATA_DIR, MODEL_PATH, OUTPUT_DIR
from infer import predict_folder


def main():
    print("=" * 60)
    print("语音鉴真 —— LSTM 模型测试")
    print(f"测试数据路径: {TEST_DATA_DIR}")
    print("=" * 60)

    if not os.path.exists(TEST_DATA_DIR):
        print(f"错误: 测试数据目录不存在 {TEST_DATA_DIR}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}，请先运行 train.py")
        return

    # 1. 加载模型，预测前先确认模型文件存在
    print(f"正在加载模型: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 结果 CSV 都放在 output 下
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = []
    # 2. 测试集的真实标签由文件夹名决定：fake / real
    for folder_name in ['fake', 'real']:
        folder_path = os.path.join(TEST_DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"跳过：未找到 {folder_path}")
            continue

        csv_path = os.path.join(OUTPUT_DIR, f"{folder_name}_result.csv")
        # 每个文件夹单独预测，写一个明细 CSV
        n_real, n_fake, total = predict_folder(folder_path, model, folder_name, csv_path)
        summary.append([folder_name, total, n_real, n_fake])

    # 3. 再写一个汇总文件，方便直接看两个文件夹的结果
    all_csv = os.path.join(OUTPUT_DIR, "all_result.csv")
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'total', 'pred_real', 'pred_fake'])
        writer.writerows(summary)

    print("\n" + "=" * 60)
    print(f"所有测试完毕，汇总结果已保存至: {all_csv}")
    for row in summary:
        print(f"  真实文件夹: {row[0]:4s} | 总数: {row[1]:3d} | 预测 real: {row[2]:3d} | 预测 fake: {row[3]:3d}")


if __name__ == '__main__':
    main()