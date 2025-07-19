import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
import ast
import random
# import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
# import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
# from webdataset.filters import _shuffle
# from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize


class CsvDataset(Dataset):

    def __init__(self, input_filename, transforms, img_key, caption_key, hard_captions_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.hard_captions = df[hard_captions_key].tolist()
        self.hard_images = df["neg_image"].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]

        chosen_caption = random.choice(self.hard_captions[idx])
        hard_captions = tokenize([str(chosen_caption)])[0]

        chose_image_index = random.choice(self.hard_images[idx])

        new_images = self.transforms(Image.open(str(self.images[chose_image_index])))
        new_texts = tokenize([str(self.captions[chose_image_index])])[0]

        chosen_caption = random.choice(self.hard_captions[chose_image_index])
        new_hard = tokenize([str(chosen_caption)])[0]

        return images, new_images, texts, new_texts, hard_captions, new_hard

class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, hard_captions_key, sep="<sep>", hn_index=None):
        logging.debug(f"Loading json data from {input_filename}.")
        with open(input_filename, "r") as f:
            self.data = json.load(f)

        self.images = [item[img_key] for item in self.data]
        self.captions = [item[caption_key] for item in self.data]
        self.hard_captions = [item[hard_captions_key].split(sep) for item in self.data]
        self.transforms = transforms
        self.root_dir = "fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0"
        self.hn_index = hn_index  # index로 사용할 값 (0~3)
        logging.debug("Done loading Json data.")
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        caption = self.captions[idx]
        hard_captions_list = self.hard_captions[idx]

        image = self.transforms(Image.open(image_path).convert("RGB"))
        texts = tokenize([caption])[0]

        if self.hn_index is not None:
            # hn_index가 범위를 벗어나지 않도록 안전하게 처리
            hn_idx = self.hn_index % len(hard_captions_list)
            hard_negative = hard_captions_list[hn_idx]
        else:
            # 기존처럼 랜덤 선택
            hard_negative = random.choice(hard_captions_list)

        hard_negative = tokenize([hard_negative])[0]

        return image, texts, hard_negative 

class JsonDataset_Da(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, hard_captions_key, sep = "<sep>"):
        logging.debug(f"Loading json data from {input_filename}.")
        with open(input_filename, "r") as f:
            self.data = json.load(f)
        
        self.images = [item[img_key] for item in self.data]
        self.captions = [item[caption_key] for item in self.data]
        self.hard_captions = [item[hard_captions_key].split(sep) for item in self.data]
        self.transforms = transforms
        self.root_dir = "fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0"
        logging.debug("Done loading Json data.")
        
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        caption = self.captions[idx]
        hard_captions_list = self.hard_captions[idx]
        
        image = self.transforms(Image.open(image_path).convert("RGB"))
        # texts = tokenize([caption])[0]
        # hard_negatives = [tokenize([hn])[0] for hn in hard_captions_list]
        texts = tokenize([caption])[0].cpu()
        hard_negatives = [tokenize([hn])[0].cpu() for hn in hard_captions_list]
        
        valid_mask = torch.ones(len(hard_negatives), dtype=torch.long)
        return image, texts, hard_negatives, valid_mask 
        
class JsonDataset_Da_blip(Dataset):
    def __init__(self, input_filename, processor, img_key, caption_key, hard_captions_key, sep="<sep>"):
        logging.debug(f"Loading json data from {input_filename}.")
        with open(input_filename, "r") as f:
            self.data = json.load(f)

        self.images = [item[img_key] for item in self.data]
        self.captions = [item[caption_key] for item in self.data]
        self.hard_captions = [item[hard_captions_key].split(sep) for item in self.data]
        self.processor = processor
        self.root_dir = "fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0"
        logging.debug("Done loading Json data.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        caption = self.captions[idx]
        hard_captions_list = self.hard_captions[idx]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.processor.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)  # [3, H, W]

        caption_ids = self.processor.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=32)['input_ids'].squeeze(0)
        hard_negatives = [
            self.processor.tokenizer(hn, return_tensors="pt", padding="max_length", truncation=True, max_length=32)['input_ids'].squeeze(0)
            for hn in hard_captions_list
        ]
        valid_mask = torch.ones(len(hard_negatives), dtype=torch.long)

        return image_tensor, caption_ids, hard_negatives, valid_mask 

import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import logging

class JsonDataset_SigLIP(Dataset):
    def __init__(self, input_filename, processor, img_key, caption_key, hard_captions_key, sep="<sep>"):
        with open(input_filename, "r") as f:
            self.data = json.load(f)

        self.images = [item[img_key] for item in self.data]
        self.captions = [item[caption_key] for item in self.data]
        self.hard_captions = [item[hard_captions_key].split(sep) for item in self.data]
        self.processor = processor
        self.root_dir = "fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0"

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")

        caption = self.captions[idx]
        hard_captions_list = self.hard_captions[idx]

        # 1️⃣ 이미지 전처리 → pixel tensor
        pixel_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # 2️⃣ 캡션 토크나이즈 → Tensor
        text_tensor = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )["input_ids"].squeeze(0)  # [seq_len]

        # 3️⃣ hard negatives 토크나이즈 → List[Tensor]
        hard_negatives = [
            self.processor.tokenizer(
                hn_caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )["input_ids"].squeeze(0)
            for hn_caption in hard_captions_list
        ]

        # 4️⃣ valid_mask
        valid_mask = torch.ones(len(hard_negatives), dtype=torch.long)

        return pixel_tensor, text_tensor, hard_negatives, valid_mask


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def filter_no_caption(sample):
    return 'txt' in sample

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True
def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        hard_captions_key=args.csv_hard_captions_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_json_dataset(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    # print(f"hn_index: {args.hn_index}")
    dataset = JsonDataset(
        input_filename, 
        preprocess_fn,
        img_key = "image_path",
        caption_key = "gt_text",
        hard_captions_key = "hard_negatives",
        sep = '<sep>',
        hn_index=args.hn_index  # index for hard negatives, if specified
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory = True, 
        sampler = sampler, 
        drop_last = is_train
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)

def custom_collate_fn(batch):
    images, texts, hard_negatives_lists, hn_masks = zip(*batch)

    # 1. Stack images: [B, C, H, W]
    images = torch.stack(images)

    # 2. GT texts (tokenized): [B, token_dim]
    texts = torch.stack(texts)

    # 3. Hard negatives: flatten and stack
    hard_negatives = []
    for hn_list in hard_negatives_lists:
        for hn in hn_list:
            hard_negatives.append(hn)  # each hn is already a tensor

    hard_negatives = torch.stack(hard_negatives)  # [B*4, token_dim]

    # 4. Combine texts and hard negatives: [B + B*4, token_dim]
    all_texts = torch.cat([texts, hard_negatives], dim=0)

    # 5. hn_masks: [B, 4]
    hn_masks = torch.stack(hn_masks)

    return images, all_texts, hn_masks

def get_json_dataset_da(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset_Da(
        input_filename, 
        preprocess_fn,
        img_key = "image_path",
        caption_key = "gt_text",
        hard_captions_key = "hard_negatives",
        sep = '<sep>'
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory = True, 
        sampler = sampler, 
        drop_last = is_train,
        collate_fn = custom_collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)

def get_json_dataset_da_blip(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset_Da_blip(
        input_filename, 
        preprocess_fn,
        img_key = "image_path",
        caption_key = "gt_text",
        hard_captions_key = "hard_negatives",
        sep = '<sep>'
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory = True, 
        sampler = sampler, 
        drop_last = is_train,
        collate_fn = custom_collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)

def get_json_dataset_da_siglip(args, preprocess_fn, is_train, epoch=0):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = JsonDataset_SigLIP(
        input_filename, 
        preprocess_fn,
        img_key = "image_path",
        caption_key = "gt_text",
        hard_captions_key = "hard_negatives",
        sep = '<sep>'
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory = True, 
        sampler = sampler, 
        drop_last = is_train,
        collate_fn = custom_collate_fn
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    
    return DataInfo(dataloader, sampler)



def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    elif dataset_type == "clip_json" : # clip model, No hard negatives
        return get_json_dataset
    elif dataset_type == "clip_json_da" : #clip model, 4 hard negatives 
        return get_json_dataset_da
    elif dataset_type == "blip_json_da" : #blip model, 4 hard negatives
        return get_json_dataset_da_blip
    elif dataset_type == "siglip_json_da" : #siglip model, 4 hard negatives
        return get_json_dataset_da_siglip
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
def get_data(args, preprocess_fns, epoch=0):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
