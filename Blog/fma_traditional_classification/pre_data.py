import json
import os
import pandas as pd

def csv2json(csv_path,audio_dir,jsonl_path):
    #根据数据特点，把第0列作为索引即track_id,第0行到第1行拼为多级索引
    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    #csv中元数据是fma_small，fma_medium，fma_large三个规模的
    #具体看csv前三好难过数据就可以看出来
    small_tracks = tracks[tracks[('set', 'subset')] == 'small']
    # labels = small_tracks[('track', 'genre_top')]
    # splits = small_tracks[('set', 'split')]
    export_df = small_tracks[[('set', 'split'), ('track', 'genre_top')]].copy()
    export_df.columns = ['split', 'genre']  # 现在索引仍是 track_id

    cache_data = {}
    missing_count = 0
    success_count = 0

    print("正在检查音频文件并建立映射...")

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for track_id, row in export_df.iterrows():
            tid_str = f'{track_id:06d}'
            relative_path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

            if os.path.exists(relative_path):
                record = {
                    'track_id': track_id,
                    'path': relative_path,
                    'genre': row['genre'],
                    'split': row['split']
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                success_count += 1
            else:
                missing_count += 1

    print("\n--- 处理完成 ---")
    print(f"成功找到并缓存的音频数量: {success_count}")
    print(f"丢失或未下载的音频数量: {missing_count}")
    print(f"缓存文件已保存至: {jsonl_path}")
if __name__=="__main__":
    csv_path = r"D:\MyProject\MyProject\Blog\data\fma_metadata\tracks.csv"
    audio_dir = r"D:\MyProject\MyProject\Blog\data\fma_small"
    jsonl_path = r"D:\MyProject\MyProject\Blog\data\fma_metadata\tracks.jsonl"
    csv2json(csv_path,audio_dir,jsonl_path)