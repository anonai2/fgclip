import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch sees", torch.cuda.device_count(), "GPUs")
print("current device:", torch.cuda.current_device())
print("device name:", torch.cuda.get_device_name(0))

# 모델과 processor 로딩
def create_blip_model_and_transforms(pretrained=True, precision="fp32", device="cuda"):
    base_model_name = "Salesforce/blip-itm-base-coco"
    model_name = pretrained if pretrained else base_model_name
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name)
    # model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", from_tf=True)
    

    custom_ckpt_path = None  # None이면 기본 모델 사용, 경로 지정 시 커스텀 체크포인트 로드
    
    custom_ckpt_path = "/fgclip/src/logs/blip_hn_far_lr_1e_4/checkpoints/epoch_3.pt"

    if custom_ckpt_path:
        print(f"🔄 Loading fine-tuned weights from: {custom_ckpt_path}")
        ckpt = torch.load(custom_ckpt_path, map_location=device, weights_only=False)

        if "state_dict" not in ckpt:
            raise ValueError("❌ 'state_dict' key not found in checkpoint.")
        
        original_state_dict = ckpt["state_dict"]

        # ✅ 'module.' prefix 제거
        state_dict = {k.replace("module.", ""): v for k, v in original_state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"✅ Loaded weights.")
        print(f"   • Total model keys     : {len(model.state_dict().keys())}")
        print(f"   • Missing keys         : {len(missing)}")
        print(f"   • Unexpected keys      : {len(unexpected)}")

    model = model.to(device)
        
    
    return model, processor, processor

model, processor, _ = create_blip_model_and_transforms(
    pretrained="Salesforce/blip-itm-base-coco",
    precision="amp",
    device=device
)
model.eval()

# 임베딩 추출 함수
def get_embeddings(image_path, text_candidates):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image {image_path}: {e}")

    inputs = processor(images=image, text=text_candidates, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        # Vision encoder + projection
        vision_out = model.vision_model(inputs["pixel_values"]).last_hidden_state[:, 0, :]  # CLS token
        image_features = F.normalize(model.vision_proj(vision_out), dim=-1)  # (1, dim)

        # Text encoder + projection
        PAD_TOKEN_ID = 0
        attention_mask = (inputs["input_ids"] != PAD_TOKEN_ID).long()
        text_out = model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # CLS token
        text_features = F.normalize(model.text_proj(text_out), dim=-1)  # (N, dim)

    return image_features, text_features

def evaluate_blip_itc(file_path, debug_samples=10):
    correct_predictions = 0
    total_samples = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue

        image_path = parts[0]
        text_candidates = parts[1].split(", ")
        ground_truth_index = int(parts[2])

        try:
            image_embedding, text_embeddings = get_embeddings(image_path, text_candidates)
        except Exception as e:
            print(f"[ERROR] Skipping {image_path}: {e}")
            continue

        # Cosine similarity
        similarities = F.cosine_similarity(image_embedding, text_embeddings)
        predicted_index = int(torch.argmax(similarities).item())

        # 디버깅 출력
        if total_samples < debug_samples:
            print(f"[DEBUG] Sample {total_samples+1}")
            print(f"  Image: {image_path}")
            print(f"  Candidates: {text_candidates}")
            print(f"  Ground Truth Index: {ground_truth_index}")
            print(f"  Predicted Index:    {predicted_index}")
            print(f"  Correct: {predicted_index == ground_truth_index}")
            print("")

        if predicted_index == ground_truth_index:
            correct_predictions += 1
        total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"\n✅ Top-1 Accuracy (BLIP-ITC): {accuracy:.4f}")
    print(f"📊 Total evaluated samples: {total_samples}")


# 사용 예시
evaluate_blip_itc("/fgclip/OCRbench-FG/ocrbench_fg.txt")
