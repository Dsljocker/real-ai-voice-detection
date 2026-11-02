# model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape):
    """两层 LSTM 加一个分类头，输入形状是 (时间步, 特征数)。"""
    model = Sequential()
    # 第一层把每个时间步都输出给第二层
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    # 第二层只看最后时刻的状态，相当于把整段序列压成一个向量
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    # 两个节点对应 fake / real，softmax 输出两类概率
    model.add(Dense(2, activation='softmax'))

    # 多分类交叉熵要配合 one-hot 标签；RMSProp 是项目里用的优化器
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSProp',
                  metrics=['accuracy'])
    return model