# features.py
import numpy as np
import librosa
from pydub import AudioSegment


def load_and_pad_audio(audio_path, target_length=50000):
    """加载音频，去除静音，并统一填充/截断到固定长度"""
    try:
        # pydub 负责解码，wav / mp3 / flac 都能直接读
        raw_audio = AudioSegment.from_file(audio_path)
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')

        # 剪掉前后太安静的部分，减少空白段对特征的干扰
        trimmed, _ = librosa.effects.trim(samples, top_db=25)

        # 统一成固定长度：短了补零，长了截断
        padding = max(0, target_length - len(trimmed))
        padded = np.pad(trimmed, (0, padding), 'constant')
        if padded.shape[0] > target_length:
            padded = padded[:target_length]
        return padded
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def extract_features(audio):
    """提取 ZCR, RMS, MFCC 三种特征"""
    # 分帧参数：每帧 2048 个点，帧与帧之间移动 512 个点
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    # 13 维 MFCC；这里的 sr=44100 是写死的，当前数据实际是 16k，换数据时要留意
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)

    # librosa 默认输出 (特征数, 时间步)，这里转成 (时间步, 特征数)
    zcr = np.swapaxes(zcr, 0, 1)  # shape: (T, 1)
    rms = np.swapaxes(rms, 0, 1)  # shape: (T, 1)
    mfccs = np.swapaxes(mfccs, 0, 1)  # shape: (T, 13)

    # 1 维 ZCR + 1 维 RMS + 13 维 MFCC = 15 维
    features = np.concatenate([zcr, rms, mfccs], axis=1)
    return features