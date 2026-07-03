import librosa


def load_audio(path,sr=22050,duration=30):
    print(f"[Audio Processing]: 正在加载音频：{path}")
    y, sr = librosa.load(path, sr=sr, duration=duration)
    length = duration * sr
    if len(y) < length:
        y = librosa.util.fix_length(data=y, size=length)
    else:
        y = y[:length]
    return y, sr


def split_audio(y,sr=22050,slice_duration=5):
    slice_samples=slice_duration*sr
    total_samples=len(y)
    num_slices=total_samples//slice_samples

    slices=[]
    for i in range(num_slices):
        start=i*slice_samples
        end=start+slice_samples
        slices.append(y[start:end])
    print(f"[Split Audio]:切分完成，一共切分音频为{len(slices)}片段")
    return slices,sr
