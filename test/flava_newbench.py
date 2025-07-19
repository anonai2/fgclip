import torch
from transformers import FlavaModel, FlavaProcessor
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
import os
from tqdm import tqdm
import torch.nn.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    
def get_embeddings(image_path, text_candidates):
    image = Image.open(image_path).convert("RGB")

    # 👉 이미지 인코딩 + projection
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_outputs = model.image_model(
            **image_inputs,
            return_dict=True,
            output_hidden_states=True
        )
        cls_image = image_outputs.last_hidden_state[:, 0, :]  # CLS token
        image_emb = model.image_projection(cls_image)
        image_emb = F.normalize(image_emb, dim=-1)

    # 👉 텍스트 인코딩 + projection
    text_inputs = processor(text=text_candidates, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(
            **text_inputs,
            return_dict=True,
            output_hidden_states=True
        )
        cls_texts = text_outputs.last_hidden_state[:, 0, :]
        text_embs = model.text_projection(cls_texts)
        text_embs = F.normalize(text_embs, dim=-1)

    return image_emb.squeeze(0).cpu(), text_embs.cpu()


def evaluate_flava_tsv(tsv_path, verbose=True):
    correct = 0
    total = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # tqdm으로 전체 라인 진행률 표시
    for i, line in enumerate(tqdm(lines, desc="Evaluating FLAVA")):
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue

        image_path = parts[0]
        text_candidates = [t.strip() for t in parts[1].split(",")]
        gt_index = int(parts[2])

        try:
            image_emb, text_embs = get_embeddings(image_path, text_candidates)
        except Exception as e:
            print(f"[ERROR] Skipping {image_path}: {e}")
            continue

        similarities = torch.matmul(text_embs, image_emb)
        pred_index = int(torch.argmax(similarities).item())

        sim_array = similarities.numpy()
        sim_min = sim_array.min()
        sim_max = sim_array.max()

        if sim_max - sim_min > 1e-6:
            norm_sims = (sim_array - sim_min) / (sim_max - sim_min)
        else:
            norm_sims = np.ones_like(sim_array) / len(sim_array)

        percent_sims = (norm_sims / norm_sims.sum()) * 100

        if verbose and i < 10:
            print(f"\n=== Sample {i+1} ===")
            print(f"Image: {image_path}")
            for j, (cand, sim_percent) in enumerate(zip(text_candidates, percent_sims)):
                flag = ""
                if j == gt_index:
                    flag += "[GT] "
                if j == pred_index:
                    flag += "[PRED] "
                print(f"{flag}{cand:15s} → Sim: {sim_percent:.2f}%")
            print("✅ Correct" if pred_index == gt_index else "❌ Incorrect")

        if pred_index == gt_index:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"corrcet: {correct}, total: {total}")
    print(f"\n==== Final Accuracy (FLAVA) on {total} samples: {acc:.4f} ====")
    
    

# Load model and processor
model = FlavaModel.from_pretrained("facebook/flava-full").to(device)
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model.eval()

ckpt_path = "/fgclip/src/logs/flava_hn_far/checkpoints/epoch_3.pt"


try:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    # 키 이름이 "module." 등으로 시작하면 지워줘야 함 (DDP나 DataParallel로 학습한 경우)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded custom weights from {ckpt_path}")
    print(f"⚠️ Missing keys: {missing}")
    print(f"⚠️ Unexpected keys: {unexpected}")

    evaluate_flava_tsv("/fgclip/OCRbench-FG/ocrbench_fg.txt", verbose=True)

except Exception as e:
    print(f"❌ Failed to load custom weights: {e}")
    
    
# evaluate_flava_tsv("/fgclip/OCRbench-FG/ocrbench_fg.txt", verbose=True)
