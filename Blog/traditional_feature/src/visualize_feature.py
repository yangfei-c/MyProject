import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_feature(features_dict,sr):
    print(f"[Visual Feature]:正在可视化特征")
    waveform = features_dict['waveform']
    log_mel = features_dict['log_mel']
    mfcc = features_dict['mfcc']
    chroma = features_dict['chroma']
    centroid = features_dict['centroid'][0]
    bandwidth = features_dict['bandwidth'][0]
    contrast = features_dict['contrast']
    flatness = features_dict['flatness'][0]
    rms = features_dict['rms'][0]
    zcr = features_dict['zcr'][0]
    onset_env = features_dict['onset_env']
    tempo = features_dict['tempo']

    hop_length=512
    times = librosa.times_like(centroid, sr=sr, hop_length=hop_length)
    onset_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    #4行3列 面板
    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(20, 22))
    fig.suptitle('Audio Traditional Features Visualization (Extended Version)', fontsize=18, fontweight='bold')

    #Waveform
    librosa.display.waveshow(waveform, sr=sr, ax=ax[0, 0])
    ax[0, 0].set_title('1. Waveform (Amplitude / Time)')
    ax[0, 0].set_xlabel('Time (s)')
    #Log-Mel Spectrogram
    img2 = librosa.display.specshow(log_mel,sr=sr,hop_length=hop_length,y_axis='mel',x_axis='time',ax=ax[0, 1])
    ax[0, 1].set_title('2. Log-Mel Spectrogram (Time-Frequency Energy)')
    fig.colorbar(img2, ax=ax[0, 1], format='%+2.0f dB')
    #MFCC
    img3 = librosa.display.specshow(mfcc,sr=sr,hop_length=hop_length,x_axis='time',ax=ax[0, 2])
    ax[0, 2].set_title('3. MFCC (Timbre / Texture)')
    fig.colorbar(img3, ax=ax[0, 2], format='%+2.0f')
    #Chroma
    img4 = librosa.display.specshow(chroma,sr=sr,hop_length=hop_length,y_axis='chroma',x_axis='time',ax=ax[1, 0])
    ax[1, 0].set_title('4. Chroma (Harmony / Pitch Class)')
    fig.colorbar(img4, ax=ax[1, 0])
    #Spectral Centroid
    ax[1, 1].plot(times, centroid, color='b')
    ax[1, 1].set_title('5. Spectral Centroid (Brightness)')
    ax[1, 1].set_ylabel('Hz')
    ax[1, 1].margins(x=0)

    #Spectral Bandwidth
    ax[1, 2].plot(times, bandwidth, color='orange')
    ax[1, 2].set_title('6. Spectral Bandwidth (Frequency Spread)')
    ax[1, 2].set_ylabel('Hz')
    ax[1, 2].margins(x=0)
    #Spectral Contrast
    img8 = librosa.display.specshow(contrast, sr=sr, hop_length=hop_length, x_axis='time', ax=ax[2, 0])
    ax[2, 0].set_title('8. Spectral Contrast (Peak-Valley Difference)')
    ax[2, 0].set_ylabel('Frequency Bands')
    fig.colorbar(img8, ax=ax[2, 0])
    #Spectral Flatness
    ax[2, 1].plot(times, flatness, color='brown')
    ax[2, 1].set_title('9. Spectral Flatness (Noise-like / Tone-like)')
    ax[2, 1].set_ylabel('Flatness')
    ax[2, 1].set_ylim(0, 1)
    ax[2, 1].margins(x=0)
    #RMS Energy
    ax[2, 2].plot(times, rms, color='r')
    ax[2, 2].set_title('10. RMS Energy (Loudness)')
    ax[2, 2].set_xlabel('Time (s)')
    ax[2, 2].margins(x=0)
    #Zero-Crossing Rate
    ax[3, 0].plot(times, zcr, color='purple')
    ax[3, 0].set_title('11. Zero-Crossing Rate (Signal Roughness)')
    ax[3, 0].set_xlabel('Time (s)')
    ax[3, 0].margins(x=0)
    #Onset Strength Envelope
    ax[3, 1].plot(onset_times, onset_env)
    ax[3, 1].set_title( '11. Onset Strength (Sound Attack / Change)')
    ax[3, 1].set_xlabel('Time (s)')
    ax[3, 1].set_ylabel('Strength')
    ax[3, 1].margins(x=0)
    #Tempo
    ax[3, 2].bar(['Tempo'],[tempo])
    ax[3, 2].set_title(f'12. Global Tempo ({tempo:.2f} BPM)' )
    ax[3, 2].set_ylabel('BPM')
    ax[3, 2].set_ylim(0,max(200, tempo + 30))
    ax[3, 2].text(0,tempo,f'{tempo:.2f} BPM',ha='center',va='bottom',fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
