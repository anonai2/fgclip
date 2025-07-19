# 📘 FGCLIP: Fine-Grained Text-Aware VLM via Hard Negatives

This repository contains the code for our anonymous submission to **[WACV 2026]** on improving fine-grained OCR understanding in vision-language models (VLMs) via contrastive learning with character-level hard negatives.

> 🔒 This repository is anonymized for double-blind review.

---

## 🔧 Setup

```bash
git clone https://github.com/anonai2/fgclip.git
cd fgclip
pip install -r requirements.txt
```
## Training datasets construction
1. Download SynthTIGER data and place Synthtiger folder under ./data : https://github.com/clovaai/synthtiger
2. Generate character-level hard negatives:
```bash
cd data
python ./train/make_hn.py
```

## 🚀 Training
```bash
cd src
fgclip/scripts/hn_far_clip.sh
```


## 📊 Evaluate with OCRBench-FG
1. Download OCRBench-V2 : https://github.com/Yuliang-Liu/MultimodalOCR
2. Run evaluation:
```bash
cd fgclip
python ./test/clip_ocrbench_fg.py

```

## 📦 Pretrained Models and Processed Datasets
We plan to release the following resources after the review process is completed.
- Model checkpoints
- Processed training dataset with character-level hard negatives (.json format)

## 📜 License
This project is licensed under the MIT License.  
It also uses the following datasets which are provided under the MIT License:

- [SynthTIGER](https://github.com/clovaai/synthtiger) by NAVER Corp.
- [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR) by Yuliang Liu

## 🙊 Anonymous Submission
This repository is anonymized for blind review. 
Please do not contact the authors untill the review process is complete.



