# i hate to set things up, but here i am doing all these stuff before training the model dammnit
#have to run
import os, json, sys
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/yolo-lab"
DIRS = {
    "datasets":   os.path.join(BASE_DIR, "datasets"),
    "runs":       os.path.join(BASE_DIR, "runs"),
    "checkpoints":os.path.join(BASE_DIR, "checkpoints"),
    "configs":    os.path.join(BASE_DIR, "configs"),
    "outputs":    os.path.join(BASE_DIR, "outputs"),
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_NAME  = "yolov8_coco_proto"
RUN_NAME  = f"{timestamp}_{EXP_NAME}"
RUN_DIR   = os.path.join(DIRS["runs"], RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

DATASET_YAML = os.path.join(DIRS["configs"], "dataset.yaml")

META = {
  "BASE_DIR": BASE_DIR,
  "DIRS": DIRS,
  "RUN_DIR": RUN_DIR,
  "RUN_NAME": RUN_NAME,
  "EXP_NAME": EXP_NAME,
  "DATASET_YAML": DATASET_YAML,
  "created_at": timestamp,
}
with open(os.path.join(RUN_DIR, "run_meta.json"), "w") as f: json.dump(META, f, indent=2)

print("YOLO bootstrap ready. :))))))))))")
print("BASE_DIR:", BASE_DIR)
print("RUN_DIR :", RUN_DIR)
print("DATASET_YAML (to create):", DATASET_YAML)

#run
from pprint import pprint
import os, json

DATASET_YAML = os.path.join (DIRS ["configs"], "dataset.yaml")
yaml_text = f"""# custom yolo dataset
path: /content/yolo_data
train: images/train
val: images/val
nc: 80
"""
os.makedirs (DIRS ["configs"], exist_ok = True)
with open (DATASET_YAML, "w") as f:
    f.write (yaml_text)
print ("Wrote dataset.yaml ->", DATASET_YAML)

CFG = {
    "exp_name": EXP_NAME,
    "project_dir": DIRS ["runs"],
    "run_name": RUN_NAME,
    "save_period": 1,
    "resume": True,
    "seed": 42,

    "data_root": "/content/yolo_data",
    "train_img_dir": "images/train",
    "val_img_dir":   "images/val",
    "train_lbl_dir": "labels/train",
    "val_lbl_dir":   "labels/val",
    "num_classes": 80,
    "imgsz": 640,
    "letterbox_pad": 114,

    "hflip_p": 0.5,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "scale_range": [0.9, 1.1],
    "translate": 0.10,

    "strides": [8, 16, 32],
    "backbone": "resnet50_fpn",
    "head_hidden": 256,
    "box_param": "ltrb",
    "use_obj_branch": True,

    # optimization
    "epochs": 50,
    "batch_size": 16,
    "optimizer": "adamw",
    "lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 3,
    "cosine_schedule": True,
    "amp": True,
    "grad_clip_norm": 10.0,
    "ema_decay": 0.9998,

    "assigner": "center_prior",
    "iou_type": "ciou",
    "loss_weights": {"box": 2.5, "cls": 1.0, "obj": 1.0, "dfl": 0.0},
}
pprint(CFG)


import os, json, subprocess, glob
from collections import defaultdict, Counter
import json
import math, random, json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#run
%pip -q install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
import shutil

fo.config.default_dataset_dir = os.path.join(DIRS["datasets"], "fo_cache")
ds = fo.load_dataset("coco2017_index")   # fast if you’ve already run Cell 4a once
#lite version, run this on startup


#for downloading
print("FiftyOne:", fo.__version__)

FO_CACHE_DIR = os.path.join(DIRS["datasets"], "fo_cache")
fo.config.default_dataset_dir = FO_CACHE_DIR
os.makedirs(FO_CACHE_DIR, exist_ok=True)

DS_NAME = "coco2017_index"
MAX_SAMPLES = 2000

# This creates/loads the listing; it doesn't force-download all media
ds = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train","validation"],
    label_types=["detections"],
    max_samples=MAX_SAMPLES,
    shuffle=False,
    dataset_name=DS_NAME,
    drop_existing=False,
)
print(ds.count_values("tags"))


#after 4a heavy
SEEN_TXT = os.path.join(DIRS["datasets"], "seen_filepaths.txt")
CURR_CHUNK_TXT = os.path.join(DIRS["datasets"], "current_chunk_paths.txt")

def _load_seen():
    if not os.path.exists(SEEN_TXT): return set()
    with open(SEEN_TXT, "r") as f:
        return set(l.strip() for l in f if l.strip())

def _append_seen(paths):
    # merge-dedupe
    seen = _load_seen().union(paths)
    with open(SEEN_TXT, "w") as f:
        for p in sorted(seen): f.write(p + "\n")

def _save_current_chunk(paths):
    with open(CURR_CHUNK_TXT, "w") as f:
        for p in paths: f.write(p + "\n")

def get_chunk_by_index(ds, split="train", k=2000, chunk_index=0):
    base = ds.match_tags(split).sort_by("filepath")
    return base.skip(chunk_index * k).limit(k)

def get_next_unseen_chunk(ds, split="train", k=2000):
    seen = _load_seen()
    base = ds.match_tags(split).sort_by("filepath")
    ids, paths = [], []
    for s in base:
        if s.filepath not in seen:
            ids.append(s.id); paths.append(s.filepath)
            if len(ids) >= k: break
    if not ids:
        print(f"No unseen samples left in {split}")
        return None
    view = base.select(ids)
    _save_current_chunk(paths)   # remember this chunk's files so we can delete them after training
    return view


#after 4b
ROOT = "/content/yolo_data"
IMG_TRAIN = f"{ROOT}/images/train"
IMG_VAL   = f"{ROOT}/images/val"
LBL_TRAIN = f"{ROOT}/labels/train"
LBL_VAL   = f"{ROOT}/labels/val"
for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(p, exist_ok=True)

COCO_NAMES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
  "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
  "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
  "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
  "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
  "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
  "vase","scissors","teddy bear","hair drier","toothbrush"
]
name_to_idx = {n:i for i,n in enumerate(COCO_NAMES)}

def detections_field(dataset):
    for f, info in dataset.get_field_schema().items():
        if info is fo.core.labels.Detections:
            return f
    return "ground_truth"

def _clean_dir(pat_list):
    import os, glob
    for pat in pat_list:
        for p in glob.glob(pat):
            try: os.remove(p)
            except: pass

def symlink_and_write_labels(sample_collection, split: str):
    field = detections_field(sample_collection._dataset)
    img_dir = IMG_TRAIN if split=="train" else IMG_VAL
    lbl_dir = LBL_TRAIN if split=="train" else LBL_VAL
    _clean_dir([os.path.join(img_dir, "*"), os.path.join(lbl_dir, "*.txt")])

    wrote = 0; bad_boxes = 0
    paths_this_chunk = []

    for s in sample_collection:
        src  = s.filepath
        stem = Path(src).stem
        ext  = Path(src).suffix.lower()
        dst_img = os.path.join(img_dir, f"{split}__{stem}{ext}")
        if not os.path.exists(dst_img):
            os.symlink(src, dst_img)

        paths_this_chunk.append(src)

        dets = getattr(s, field, None)
        lines = []
        if dets and dets.detections:
            for d in dets.detections:
                x, y, w, h = d.bounding_box
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    bad_boxes += 1; continue
                cls = name_to_idx.get(d.label, None)
                if cls is None: continue
                cx = min(max(x + w/2, 0.0), 1.0)
                cy = min(max(y + h/2, 0.0), 1.0)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(lbl_dir, f"{split}__{stem}.txt"), "w") as f:
            f.write("\n".join(lines))
        wrote += 1

    print(f"✅ {split}: wrote {wrote} labels; bad boxes skipped: {bad_boxes}")
    # also remember this chunk's filepaths (for deletion)
    _save_current_chunk(paths_this_chunk)


# Cell 4d — select a chunk and export
#run after 4a lite
CHUNK_SIZE = 2000

train_chunk = get_next_unseen_chunk(ds, "train", k=CHUNK_SIZE)

val_chunk = get_chunk_by_index(ds, "validation", k=500, chunk_index=0)

if train_chunk is not None:
    symlink_and_write_labels(train_chunk, "train")

if val_chunk is not None:
    symlink_and_write_labels(val_chunk, "val")

#RUN AFTER TRAINING , RUN WITH CAUTION!!!!!!!!!!
import shutil

def dispose_current_chunk(delete_from_cache=True, mark_seen=True):
    shutil.rmtree("/content/yolo_data", ignore_errors=True)
    for p in [
        "/content/yolo_data/images/train",
        "/content/yolo_data/images/val",
        "/content/yolo_data/labels/train",
        "/content/yolo_data/labels/val",
    ]:
        os.makedirs(p, exist_ok=True)

    if delete_from_cache and os.path.exists(CURR_CHUNK_TXT):
        with open(CURR_CHUNK_TXT, "r") as f:
            paths = [l.strip() for l in f if l.strip()]
        deleted, missing = 0, 0
        for p in paths:
            try:
                os.remove(p)
                deleted += 1
            except FileNotFoundError:
                missing += 1
            except Exception as e:
                print("Delete error:", p, e)

        print(f"Cache cleanup: deleted {deleted}, missing {missing}")

        if mark_seen and paths:
            _append_seen(paths)

        open(CURR_CHUNK_TXT, "w").close()

    print("Disposed current chunk.")


Stage 3

#still preparing, let me cook
def set_seed (seed = 42):
    import random, numpy as np, torch
    random.seed (seed)
    np.random.seed (seed)
    torch.manual_seed (seed)
    torch.cuda.manual_seed_all (seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed (CFG ["seed"])
device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
print (f"device = {device}")

def yolo_to_xyxy (labels_normal, W, H):
    if labels_normal.size == 0:
        return np.zeros ((0, 4), dtype = np.float32), np.zeros ((0,), dtype = np.int64)

    cls = labels_normal [:, 0].astype (np.int64)
    cx = labels_normal [:, 1] * W
    cy = labels_normal [:, 2] * H
    w = labels_normal [:, 3] * W
    h = labels_normal [:, 4] * H
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack ([x1, y1, x2, y2], axis = 1).astype (np.float32)
    return boxes, cls

def clip_boxes_xyxy (boxes, w, h):
    if boxes.size == 0:
        return boxes
    boxes [:, 0] = np.clip (boxes [:, 0], 0, w - 1)
    boxes [:, 1] = np.clip (boxes [:, 1], 0, h - 1)
    boxes [:, 2] = np.clip (boxes [:, 2], 0, w - 1)
    boxes [:, 3] = np.clip (boxes [:, 3], 0, h - 1)
    return boxes

def valid_boxes (boxes, min_size = 2.0):
    if boxes.size == 0:
        return np.zeros ((0,), dtype = bool)
    w = boxes [:, 2] - boxes [:, 0]
    h = boxes [:, 3] - boxes [:, 1]
    return (w >= min_size) & (h >= min_size)

def letterbox (image, new_size = 640, pad_value = 114):
  H, W = image.shape [:2]
  r = min (new_size / H, new_size / W)
  new_unpad = (int (round (W * r)), int (round (H * r)))
  interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
  image_resized = cv2.resize (image, new_unpad, interpolation=interp)
  dw = new_size - new_unpad [0]
  dh = new_size - new_unpad [1]
  pad_x = dw // 2
  pad_y = dh // 2
  image_label = cv2.copyMakeBorder (
      image_resized, pad_y, dh - pad_y, pad_x, dw - pad_x,
      borderType = cv2.BORDER_CONSTANT, value = (pad_value, pad_value, pad_value)
  )
  return image_label, float (r), (pad_x, pad_y)

#color aug iykyk
def augment_hsv (image, hgain = 0.015, sgain = 0.7, vgain = 0.4):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return image
    if image.ndim != 3 or image.shape [2] != 3:
        return image
    hsv = cv2.cvtColor (image, cv2.COLOR_BGR2HSV).astype (np.float32)
    h, s, v = hsv [...,0], hsv [...,1], hsv [...,2]
    a = 1 + (random.random () * 2 - 1) * hgain
    b = 1 + (random.random () * 2 - 1) * sgain
    c = 1 + (random.random () * 2 - 1) * vgain
    h [:] = np.clip (h * a, 0, 179)
    s [:] = np.clip (s * b, 0, 255)
    v [:] = np.clip (v * c, 0, 255)
    return cv2.cvtColor (hsv.astype (np.uint8), cv2.COLOR_HSV2BGR)

#dataset class
#turning yolo labels to tensors

class YoloDataset (Dataset):
    def __init__ (self, image_dir, label_dir, imgsz = 640, augment = True, pad_value = 114, horizontal_flip_prob = 0.5, hsv_hgain = 0.015, hsv_sgain = 0.7, hsv_vgain = 0.4):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.imgsz = imgsz
        self.augment = augment
        self.pad_value = pad_value
        self.horizontal_flip_prob = horizontal_flip_prob
        self.hsv_hgain = hsv_hgain
        self.hsv_sgain = hsv_sgain
        self.hsv_vgain = hsv_vgain

        self.image_paths = sorted (glob.glob (os.path.join (image_dir, "*.jpg")))
        assert len (self.image_paths) > 0, f"No images found in {image_dir}"
        self.stems = [Path (p).stem for p in self.image_paths]
        self.label_paths = [os.path.join (label_dir, s + ".txt") for s in self.stems]

    def __len__ (self):
        return len (self.image_paths)

    def _read_image (self, path):
        image = cv2.imread (path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError (f"Fail to read: {path}")
        return image

    def _read_labels (self, path):
      if not os.path.exists (path):
        return np.zeros ((0, 5), dtype = np.float32)
      with open (path, "r") as f:
          lines = [l.strip () for l in f if l.strip ()]
      if not lines:
          return np.zeros ((0, 5), dtype = np.float32)

      out = []
      for ln in lines:
          parts = ln.split ()

          if len(parts) != 5:
              continue
          cls, cx, cy, w, h = parts
          out.append ([float (cls), float (cx), float (cy), float (w), float (h)])
      arr = np.asarray (out, dtype = np.float32)

      if arr.size:
          arr [:, 1 : 5] = np.clip (arr[:, 1 : 5], 0.0, 1.0)
      return arr

    def __getitem__ (self, index):
        image_path = self.image_paths [index]
        label_path = self.label_paths [index]
        stem = self.stems [index]

        image = self._read_image (image_path)
        H, W = image.shape [:2]
        labels = self._read_labels (label_path) #[N, 5] (cls, cx, cy, w, h) normalized

        if self.augment:
            image = augment_hsv (image, self.hsv_hgain, self.hsv_sgain, self.hsv_vgain)

            if random.random () < self.horizontal_flip_prob:
                image = image [:, ::-1, :]

                if labels.size > 0:
                    labels [:, 1] = 1.0 - labels [:, 1]

        boxes_px, classes = yolo_to_xyxy (labels, W, H)

        image_label, scale, (pad_x, pad_y) = letterbox (image, self.imgsz, self.pad_value)

        if boxes_px.size != 0:
            boxes_px [:, [0, 2]] = boxes_px [:, [0, 2]] * scale + pad_x
            boxes_px [:, [1, 3]] = boxes_px [:, [1, 3]] * scale + pad_y
            boxes_px = clip_boxes_xyxy (boxes_px, self.imgsz, self.imgsz)
            keep = valid_boxes (boxes_px, min_size = 2.0)
            boxes_px = boxes_px [keep]
            classes = classes [keep]

        image_rgb = cv2.cvtColor (image_label, cv2.COLOR_BGR2RGB)
        image_float = (image_rgb.astype (np.float32) / 255.0).transpose (2, 0, 1)
        image = torch.from_numpy (image_float)
        target = {
            "boxes": torch.from_numpy (boxes_px).float (),
            "labels": torch.from_numpy (classes).long (),
            "image_id": stem,
            "orig_size": (H, W),
            "scale": scale,
            "pad": (pad_x, pad_y),
        }
        return image, target

#colate function and dataloaders

def collate_fn (batch):
    images, targets = list (zip (*batch))
    B = len (images)
    images = torch.stack (images, dim = 0)

    all_boxes = []
    all_labels = []
    all_bidx = []
    image_ids = []
    scales = []
    pads = []

    for i, t in enumerate (targets):
        n = t ["boxes"].shape [0]

        if n:
            all_boxes.append (t ["boxes"])
            all_labels.append (t ["labels"])
            all_bidx.append (torch.full ((n,), i, dtype = torch.int64))

        image_ids.append (t ["image_id"])
        scales.append (t ["scale"])
        pads.append (t ["pad"])

    if len (all_boxes):
        boxes = torch.cat (all_boxes, 0)
        labels = torch.cat (all_labels, 0)
        bidx = torch.cat (all_bidx, 0)
    else:
        boxes = torch.zeros ((0, 4), dtype = torch.float32)
        labels = torch.zeros ((0,), dtype = torch.int64)
        bidx = torch.zeros ((0,), dtype = torch.int64)

    return images, {
        "boxes": boxes,
        "labels": labels,
        "batch_idx": bidx,
        "image_id": image_ids,
        "scale": scales,
        "pad": pads,
    }

def _wif (worker_id):
  seed = CFG ["seed"] + worker_id
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed (seed)

VAL_IMG_DIR = os.path.join (CFG ["data_root"], CFG ["val_img_dir"])
VAL_LBL_DIR = os.path.join (CFG ["data_root"], CFG ["val_lbl_dir"])
val_ds = YoloDataset (VAL_IMG_DIR, VAL_LBL_DIR, imgsz = CFG ["imgsz"], augment = False,
                      pad_value = CFG ["letterbox_pad"], horizontal_flip_prob = 0.0)
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    worker_init_fn=_wif,
    generator=torch.Generator().manual_seed(CFG["seed"]),
)
print ("val dataset:", len (val_ds))

TRAIN_IMG_DIR = os.path.join (CFG ["data_root"], CFG ["train_img_dir"])
TRAIN_LBL_DIR = os.path.join (CFG ["data_root"], CFG ["train_lbl_dir"])

if os.path.isdir (TRAIN_IMG_DIR) and os.path.isdir (TRAIN_LBL_DIR):
    train_ds = YoloDataset (TRAIN_IMG_DIR, TRAIN_LBL_DIR, imgsz = CFG ["imgsz"], augment = True,
                            pad_value = CFG ["letterbox_pad"], horizontal_flip_prob = CFG ["hflip_p"],
                            hsv_hgain = CFG ["hsv_h"], hsv_sgain = CFG ["hsv_s"], hsv_vgain = CFG ["hsv_v"])
    train_loader = DataLoader(
    train_ds,
    batch_size=CFG["batch_size"],
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    worker_init_fn=_wif,
    generator=torch.Generator().manual_seed(CFG["seed"]),  # master seed
    )
    if train_loader is not None:
      xb, tb = next(iter(train_loader))
      print("Train batch:", xb.shape, xb.dtype)
      print("Targets keys:", list(tb.keys()))
      print("Boxes:", tb["boxes"].shape, tb["boxes"].dtype)
      print("Labels:", tb["labels"].shape, tb["labels"].dtype)
      print("Batch idx:", tb["batch_idx"].shape)
    else:
      print("No train loader available")

    xbv, tbv = next(iter(val_loader))
    print("Val batch:", xbv.shape, xbv.dtype)
else:
    train_loader = None
    print ("no training, train dataset not found")
