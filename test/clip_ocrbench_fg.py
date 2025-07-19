import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
from PIL import Image
from torchvision import transforms
import open_clip
from scipy.spatial.distance import cosine
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Load tokenizer + model architecture
model_name = "ViT-B-32"  # or "ViT-B-16" 
pretrained = "laion400m_e32"  # or "laion400m_e32" 
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# 2. Load fine-tuned weights
ckpt_path = "/workspace/code/neg_clip/src/logs/all_hn_far_5epoch/checkpoints/epoch_2.pt"
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

state_dict = checkpoint["state_dict"]

# Remove 'module.' prefix (from DataParallel)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# 3. Inference example
def get_embeddings(image_path, text_candidates):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer(text_candidates).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    return image_features, text_features

def evaluate_model(file_path):
    """Evaluates the model using cosine similarity and computes Top-1 Accuracy."""
    correct_predictions = 0
    total_samples = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue

        image_path = parts[0]
        text_candidates = parts[1].split(", ")
        ground_truth_index = int(parts[2])

        # Get embeddings
        image_embedding, text_embedding = get_embeddings(image_path, text_candidates)

        # Compute cosine similarities
        similarities = [
            # 1 - cosine(image_embedding.numpy().flatten(), t.numpy().flatten())
            1 - cosine(image_embedding.cpu().numpy().flatten(), t.cpu().numpy().flatten())

            for t in text_embedding
        ]

        # Normalize similarities to sum to 100% (min-max scaling)
        sim_array = np.array(similarities)
        sim_min = sim_array.min()
        sim_max = sim_array.max()

        if sim_max - sim_min > 1e-6:
            norm_sims = (sim_array - sim_min) / (sim_max - sim_min)
        else:
            norm_sims = np.ones_like(sim_array) / len(sim_array)

        percent_sims = (norm_sims / norm_sims.sum()) * 100

        predicted_index = int(np.argmax(similarities))
        if predicted_index == ground_truth_index:
            correct_predictions += 1

        total_samples += 1

        # === Softmax-like Output for First 10 Samples ===
        if i < 10:
            print(f"\n=== Sample {i + 1} ===")
            print(f"Image: {image_path}")
            for j, (cand, sim_percent) in enumerate(zip(text_candidates, percent_sims)):
                flag = ""
                if j == ground_truth_index:
                    flag += "[GT] "
                if j == predicted_index:
                    flag += "[PRED] "
                print(f"{flag}{cand:15s} → Sim: {sim_percent:.2f}%")
            print("✅ Correct" if predicted_index == ground_truth_index else "❌ Incorrect")

    accuracy = correct_predictions / total_samples
    print(f"\n==== Final Top-1 Accuracy on {total_samples} samples: {accuracy:.4f} ====")


# Example usage
evaluate_model("/fgclip/OCRbench-FG/ocrbench_fg.txt")