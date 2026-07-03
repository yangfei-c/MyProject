import librosa
import numpy as np


def extract_feature(slice, sr):
    config_dict = {
        'n_fft': 2048,
        'hop_length': 512,
        'n_mels': 128,
        'n_mfcc': 20,
        'roll_percent': 0.85,
        'tempogram_win_length': 192
    }

    features_dict = dict.fromkeys([
        'waveform', 'log_mel', 'mfcc', 'chroma','centroid', 'bandwidth',
        'contrast','flatness', 'rms', 'zcr','onset_env', 'tempo',
    ])

    print(f"[Feature Extraction]: 准备提取特征{list(features_dict.keys())}")

    n_fft = config_dict['n_fft']
    hop_length = config_dict['hop_length']

    # Log-Mel
    mel = librosa.feature.melspectrogram(y=slice, sr=sr, n_fft=n_fft,hop_length=hop_length,n_mels=config_dict['n_mels'])
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # 节奏特征
    onset_env = librosa.onset.onset_strength(y=slice,sr=sr,hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track( onset_envelope=onset_env,sr=sr,hop_length=hop_length)
    tempo = (float(np.asarray(tempo).reshape(-1)[0])if np.asarray(tempo).size else 0.0)

    features_dict.update({
        'waveform': slice,
        'log_mel': log_mel,
        'mfcc': librosa.feature.mfcc(S=log_mel,n_mfcc=config_dict['n_mfcc']),
        'chroma': librosa.feature.chroma_stft(y=slice,sr=sr,n_fft=n_fft,hop_length=hop_length),
        'centroid': librosa.feature.spectral_centroid( y=slice,sr=sr,n_fft=n_fft,hop_length=hop_length),
        'bandwidth': librosa.feature.spectral_bandwidth(y=slice,sr=sr,n_fft=n_fft,hop_length=hop_length),
        'contrast': librosa.feature.spectral_contrast(y=slice, sr=sr,n_fft=n_fft,hop_length=hop_length),
        'flatness': librosa.feature.spectral_flatness( y=slice,n_fft=n_fft,hop_length=hop_length),
        'rms': librosa.feature.rms(y=slice,frame_length=n_fft,hop_length=hop_length),
        'zcr': librosa.feature.zero_crossing_rate(y=slice,frame_length=n_fft,hop_length=hop_length),
        'onset_env': onset_env,
        'tempo': tempo,
    })
    print( f"[Feature Extraction]: 特征提取完成，共提取{len(features_dict)}项特征")

    return features_dict