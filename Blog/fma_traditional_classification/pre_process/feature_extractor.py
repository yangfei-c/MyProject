from __future__ import annotations

import librosa
import numpy as np

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20


def add_mean_std(output,name,values) -> None:
    """
    例如：
        MFCC: [20, 时间帧]
        Chroma: [12, 时间帧]
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[np.newaxis, :]

    means = np.nan_to_num(np.mean(values, axis=1),nan=0.0,posinf=0.0,neginf=0.0,)
    stds = np.nan_to_num(np.std(values, axis=1),nan=0.0,posinf=0.0,neginf=0.0,)

    for index, value in enumerate(means, start=1):
        output[f"{name}_{index:02d}_mean"] = float(value)

    for index, value in enumerate(stds, start=1):
        output[f"{name}_{index:02d}_std"] = float(value)


def extract_feature(y,sr,):
    """
    提取一首音乐的87维传统特征。
    特征：
        MFCC、Chroma、Spectral Contrast、
        Spectral Centroid、Spectral Rolloff、
        RMS、ZCR、Tempo
    """
    if y.ndim != 1:
        y = librosa.to_mono(y)

    if len(y) < N_FFT:
        raise ValueError(
            f"音频过短：{len(y)}个采样点，"
            f"至少需要{N_FFT}个采样点。"
        )

    features: dict[str, float] = {}

    stft_complex = librosa.stft(y=y,n_fft=N_FFT,hop_length=HOP_LENGTH,window="hann",)

    magnitude = np.abs(stft_complex)
    power = magnitude ** 2

    # 1. MFCC：20维
    mel = librosa.feature.melspectrogram(S=power,sr=sr,n_mels=N_MELS,)
    log_mel = librosa.power_to_db(mel,ref=np.max,)
    mfcc = librosa.feature.mfcc(S=log_mel,n_mfcc=N_MFCC,)
    add_mean_std(features,"mfcc",mfcc,)

    # 2. Chroma：12维
    chroma = librosa.feature.chroma_stft(S=power,sr=sr,n_chroma=12,)
    add_mean_std(features,"chroma",chroma,)

    # 3. Spectral Contrast：7维
    contrast = librosa.feature.spectral_contrast(S=magnitude,sr=sr, n_bands=6,)
    add_mean_std(features,"contrast",contrast,)

    # 4. Spectral Centroid：1维
    centroid = librosa.feature.spectral_centroid(S=magnitude,sr=sr,)
    add_mean_std(features,"centroid",centroid,)

    # 5. Spectral Rolloff：1维
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr,roll_percent=0.85,)
    add_mean_std(features,"rolloff",rolloff,)

    # 6. RMS：1维
    rms = librosa.feature.rms(S=magnitude,frame_length=N_FFT,hop_length=HOP_LENGTH,)
    add_mean_std(features,"rms",rms,)

    # 7. Zero Crossing Rate：1维
    zcr = librosa.feature.zero_crossing_rate(y=y,frame_length=N_FFT,hop_length=HOP_LENGTH,)
    add_mean_std(features,"zcr",zcr,)

    # 8. Tempo：1维
    onset_env = librosa.onset.onset_strength(y=y,sr=sr,hop_length=HOP_LENGTH,)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env,sr=sr,hop_length=HOP_LENGTH,)
    tempo_array = np.asarray(tempo).reshape(-1)
    features["tempo"] = (float(tempo_array[0])if tempo_array.size > 0 else 0.0)

    # 检查维度
    if len(features) != 87:
        raise RuntimeError(
            f"特征维度错误：预期87维，"
            f"实际得到{len(features)}维。"
        )
    return features

if __name__ == "__main__":
    test_path = (
        r"D:\MyProject\MyProject\Blog"
        r"\data\fma_small\000\000002.mp3"
    )

    y, sr = librosa.load(test_path,sr=22050,mono=True,duration=30.0,)

    feature = extract_feature(y, sr)
    print(f"采样率：{sr}")
    print(f"特征维度：{len(feature)}")
    for name, value in list(feature.items()):
        print(f"{name}: {value:.6f}")