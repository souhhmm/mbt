import os
import ssl
import wget
import urllib.request
import rarfile
import zipfile


context = ssl._create_unverified_context()
def download_url(url, path):
    print(f"downloading {url}...")
    with urllib.request.urlopen(url, context=context) as response:
        with open(path, 'wb') as f:
            f.write(response.read())

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    annotation_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    download_url(video_url, os.path.join(data_dir, "UCF101.rar"))
    download_url(annotation_url, os.path.join(data_dir, "UCF101_annotations.zip"))
    
    print("extracting video data...")
    rf = rarfile.RarFile(os.path.join(data_dir, "UCF101.rar"))
    rf.extractall(data_dir)
    
    print("extracting annotations...")
    with zipfile.ZipFile(os.path.join(data_dir, "UCF101_annotations.zip")) as zf:
        zf.extractall(data_dir)
    
    os.remove(os.path.join(data_dir, "UCF101.rar"))
    os.remove(os.path.join(data_dir, "UCF101_annotations.zip"))
    
    print("downloading ast...")
    os.makedirs('pretrained_weights', exist_ok=True)
    wget.download('https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1', os.path.join('pretrained_weights', 'audioset_16_16_0.4422.pth'))
    print("done")
    
if __name__ == "__main__":
    main()
