# train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from config import TRAIN_DATA_DIR, MODEL_PATH, MODULE_DIR, OUTPUT_DIR, LOSS_CURVE_PATH, EPOCHS, BATCH_SIZE
from dataset import AudioDataset
from model import build_lstm_model


def main():
    print("=" * 60)
    print("语音鉴真 —— LSTM 模型训练")
    print("=" * 60)

    os.makedirs(MODULE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 特征缓存文件：第一次训练后把特征存下来，后面就不用重复读音频
    FEATURE_CACHE = os.path.join(os.path.dirname(__file__), "cached_features.npz")

    # 1. 加载数据（带缓存）
    if os.path.exists(FEATURE_CACHE):
        print(f"检测到特征缓存，直接加载: {FEATURE_CACHE}")
        data = np.load(FEATURE_CACHE)
        X, y = data['X'], data['y']
    else:
        print("首次运行，正在提取特征（较慢，只等这一次）...")
        dataset = AudioDataset(TRAIN_DATA_DIR, is_train=True)
        X = dataset.X
        y = dataset.y
        if len(X) > 0:
            # 提完特征存一份，下次训练直接读
            np.savez_compressed(FEATURE_CACHE, X=X, y=y)
            print(f"特征已保存缓存至: {FEATURE_CACHE}")

    if len(X) == 0:
        print("错误：未提取到任何特征数据，请检查 train_data 目录！")
        return


    # 2. 划分训练集和验证集，测试集不参与这一步
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集: {X_train.shape[0]} 条, 验证集: {X_val.shape[0]} 条")
    input_shape = (X.shape[1], X.shape[2])  # (时间步, 特征维度)

    # 3. 构建模型
    model = build_lstm_model(input_shape)
    model.summary()

    # 4. 设置回调函数
    # 验证损失连续 10 轮没下降就停，并恢复到最好的那一轮
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # 每个 epoch 结束后检查一次，只保存验证损失最低的模型
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

    # 5. 开始训练
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint]
    )

    print(f"训练完成，最优模型已保存至: {MODEL_PATH}")

    # 6. 绘制曲线：左边准确率，右边损失
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH)
    print(f"训练曲线已保存至: {LOSS_CURVE_PATH}")


if __name__ == '__main__':
    main()