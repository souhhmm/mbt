import os
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def analyze_dataset(data_path, split_path):
    train_file = os.path.join(split_path, "ucfTrainTestlist", "trainlist01.txt")
    test_file = os.path.join(split_path, "ucfTrainTestlist", "testlist01.txt")
    
    with open(train_file, "r") as f:
        train_videos = [line.strip().split(" ")[0] for line in f.readlines()]
    
    with open(test_file, "r") as f:
        test_videos = [line.strip().split(" ")[0] for line in f.readlines()]
    
    data_path = Path(data_path)
    
    def check_audio(video_list):
        has_audio = 0
        no_audio = 0
        
        for video in tqdm(video_list):
            video_path = data_path / "UCF-101" / video
            try:
                audio_array, _ = torchaudio.load(str(video_path))
                if audio_array.shape[1] > 0:  # check if audio has content
                    has_audio += 1
                else:
                    no_audio += 1
            except:
                no_audio += 1
        
        return has_audio, no_audio
    
    print("analyzing training split...")
    train_has_audio, train_no_audio = check_audio(train_videos)
    
    print("analyzing test split...")
    test_has_audio, test_no_audio = check_audio(test_videos)
    
    # create visualization
    labels = ['Training Set', 'Test Set']
    has_audio = [train_has_audio, test_has_audio]
    no_audio = [train_no_audio, test_no_audio]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], has_audio, width, label='Has Audio')
    ax.bar([i + width/2 for i in x], no_audio, width, label='No Audio')
    
    ax.set_ylabel('Number of Videos')
    ax.set_title('Audio Availability in UCF101 Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i in x:
        ax.text(i - width/2, has_audio[i], str(has_audio[i]), ha='center', va='bottom')
        ax.text(i + width/2, no_audio[i], str(no_audio[i]), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('audio_availability.png')
    plt.close()
    
    print("\nSummary:")
    print(f"Training Set - Total: {len(train_videos)}")
    print(f"  - With Audio: {train_has_audio} ({train_has_audio/len(train_videos)*100:.1f}%)")
    print(f"  - Without Audio: {train_no_audio} ({train_no_audio/len(train_videos)*100:.1f}%)")
    
    print(f"\nTest Set - Total: {len(test_videos)}")
    print(f"  - With Audio: {test_has_audio} ({test_has_audio/len(test_videos)*100:.1f}%)")
    print(f"  - Without Audio: {test_no_audio} ({test_no_audio/len(test_videos)*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset("data/", "data/")
