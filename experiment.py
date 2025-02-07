import torch
from model import Model
from dataloader import UCF101Dataset


def load_model(checkpoint_path, device="cuda"):
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_class_mapping():
    # create a dataset instance just to get the class mapping
    dataset = UCF101Dataset(
        data_path="data/", split_path="data/ucfTrainTestlist", split="test"
    )
    # invert the class_to_idx dictionary
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return idx_to_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    checkpoint_path = "checkpoints/wabt_20250206_182221/best.pt"
    model = load_model(checkpoint_path, device)

    # get class mapping
    idx_to_class = get_class_mapping()

    # create dataset and get a sample
    dataset = UCF101Dataset(
        data_path="data/",
        split_path="data/ucfTrainTestlist",
        split="test",
    )

    random_idxs = torch.randperm(len(dataset))[:10]

    for i in random_idxs:
        video, audio, true_label = dataset[i]

        # add batch dimension
        video = video.unsqueeze(0).to(device)
        audio = audio.unsqueeze(0).to(device)

        # get prediction
        with torch.no_grad():
            outputs = model(video, audio)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # get top 3 predictions
            top_prob, top_idx = torch.topk(probabilities, 3)

        print(f"\nsample {i + 1}")
        print(f"true class: {idx_to_class[true_label]}")
        print("top 3 predictions:")
        for prob, idx in zip(top_prob[0], top_idx[0]):
            print(f"{idx_to_class[idx.item()]}: {prob.item()*100:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    main()
