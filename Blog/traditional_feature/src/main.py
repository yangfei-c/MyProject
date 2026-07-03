import numpy as np

from audio_process import load_audio,split_audio
from extract_feature import extract_feature
from visualize_feature import plot_feature
from pathlib import Path

# 获取项目根目录（Blog文件夹）
PROJECT_ROOT = Path(__file__).parent.parent.parent
audio_path = PROJECT_ROOT / "data" / "000002.mp3"



if __name__ == '__main__':
    y, sr = load_audio(str(audio_path))
    slice,slice_sr=split_audio(y,sr)
    feature_dict=extract_feature(slice[0],slice_sr)
    plot_feature(feature_dict,slice_sr)
    np.savez_compressed(
        audio_path ,
        sr=slice_sr,
        n_fft=2048,
        hop_length=512,
        slice_duration=len(slice[0]) / slice_sr,
        **feature_dict
    )
