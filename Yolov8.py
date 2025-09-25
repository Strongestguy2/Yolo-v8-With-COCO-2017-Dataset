# i hate to set things up, but here i am doing all these stuff before training the model dammnit
#have to run
# %pip -q install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
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
FO_CACHE_DIR = "/content/fo_cache"
fo.config.default_dataset_dir = FO_CACHE_DIR
os.makedirs(FO_CACHE_DIR, exist_ok=True)
DATASET_YAML = os.path.join(DIRS["configs"], "dataset.yaml")
VAL_DS_NAME  = "coco2017_val500_fixed"
VAL_TXT = os.path.join(DIRS["datasets"], "val_filepaths.txt")

if not os.path.exists(VAL_TXT):
    ds_val = foz.load_zoo_dataset(
        "coco-2017",
        splits=["validation"],
        label_types=["detections"],
        max_samples=500,
        shuffle=True,
        seed=1234,
        dataset_name=VAL_DS_NAME,
        drop_existing=True,
    )
    with open(VAL_TXT, "w") as f:
        for s in ds_val:
            f.write(s.filepath + "\n")
else:
    pass

META = {
  "BASE_DIR": BASE_DIR,
  "DIRS": DIRS,
  "RUN_DIR": RUN_DIR,
  "RUN_NAME": RUN_NAME,
  "EXP_NAME": EXP_NAME,
  "DATASET_YAML": DATASET_YAML,
  "created_at": timestamp,
}
with open(os.path.join(RUN_DIR, "run_meta.json"), "w") as f:
  json.dump(META, f, indent=2)

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
    "viz_every": 200,

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

import shutil

import time
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

DS_TRAIN_ROLL = "coco2017_train_roll2k"
TARGET = 2000

def load_next_train_chunk(seen_paths, tries=6):
    seen_paths = list(seen_paths)  # ensure iterable is a list
    for _ in range(tries):
        seed = int(time.time()) % 2_000_000_000
        ds = foz.load_zoo_dataset(
            "coco-2017",
            splits=["train"],
            label_types=["detections"],
            max_samples=TARGET,
            shuffle=True,
            seed=seed,
            dataset_name=DS_TRAIN_ROLL,
            drop_existing=True,
        )

        # keep samples whose filepath is NOT in seen_paths
        fresh = ds.match(~F("filepath").is_in(seen_paths))

        n = len(fresh)
        if n >= int(0.9 * TARGET):
            print(f"Using train chunk with {n} fresh samples (seed={seed})")
            return fresh

        print(f"Overlap too high ({n} fresh). Retrying...")

    print(f"Proceeding with {n} fresh after retries")
    return fresh



#after 4a heavy
SEEN_TXT = os.path.join(DIRS["datasets"], "seen_filepaths.txt")
CURR_CHUNK_TXT = os.path.join(DIRS["datasets"], "current_chunk_paths.txt")

def _load_seen():
    if not os.path.exists(SEEN_TXT): return set()
    with open(SEEN_TXT, "r") as f:
        return set(l.strip() for l in f if l.strip())

def _stage_image(src, dst):
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)

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
    _save_current_chunk(paths)
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

def detections_field (dataset):
    schema = dataset.get_field_schema ()
    for name, field in schema.items ():
        if hasattr (field, "document_type"):
            try:
                import fiftyone.core.labels as fol
                if field.document_type is fol.Detections:
                    return name
            except Exception:
                pass
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


def stage_train_chunk ():
    seen = _load_seen ()
    view = load_next_train_chunk (seen)
    symlink_and_write_labels (view, "train")
    print("[TRAIN] staged ~2000 training samples.")

def stage_val_once():

    ds_val = foz.load_zoo_dataset (
        "coco-2017",
        splits = ["validation"],
        label_types = ["detections"],
        max_samples = 500,
        shuffle = True,
        seed = 1234,
        dataset_name = "coco2017_val500_fixed",
        drop_existing = False,
    )
    # Stage to /content/yolo_data/{images,labels}/val
    symlink_and_write_labels(ds_val, "val")
    print("[VAL] staged ~500 validation samples.")

# call it here so Run-All always has val staged
stage_val_once()

#RUN AFTER TRAINING , RUN WITH CAUTION!!!!!!!!!!
import shutil, os

def _reset_working_set ():
    shutil.rmtree ("/content/yolo_data", ignore_errors=True)
    for p in [
        "/content/yolo_data/images/train",
        "/content/yolo_data/images/val",
        "/content/yolo_data/labels/train",
        "/content/yolo_data/labels/val",
    ]:
        os.makedirs(p, exist_ok=True)

def _safe_remove_media(paths, root=FO_CACHE_DIR):
    deleted = missing = skipped = 0
    root_abs = os.path.abspath(root) + os.sep
    for p in paths:
        ap = os.path.abspath(p)
        if not ap.startswith(root_abs):
            skipped += 1
            continue
        try:
            os.remove(ap); deleted += 1
        except FileNotFoundError:
            missing += 1
        except Exception as e:
            print("Delete error:", ap, e)
    print(f"Cache cleanup: deleted {deleted}, missing {missing}, skipped {skipped}")

def dispose_current_chunk(delete_from_cache=True, mark_seen=True):
    _reset_working_set()

    if os.path.exists(CURR_CHUNK_TXT):
        with open(CURR_CHUNK_TXT, "r") as f:
            paths = [l.strip() for l in f if l.strip()]

        if delete_from_cache and paths:
            _safe_remove_media(paths, root=FO_CACHE_DIR)

        if mark_seen and paths:
            _append_seen(paths)

        open(CURR_CHUNK_TXT, "w").close()

    print("Disposed current chunk.")


#Stage 3

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

        patterns = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
        paths = []
        for patt in patterns:
            paths.extend(glob.glob(os.path.join(image_dir, patt)))

        paths = sorted(paths)

        seen_stems = set()
        image_paths = []
        for p in paths:
            s = Path(p).stem
            if s in seen_stems:
                continue
            seen_stems.add(s)
            image_paths.append(p)

        self.image_paths = image_paths
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir} matching {patterns}")

        self.stems = [Path(p).stem for p in self.image_paths]
        self.label_paths = [os.path.join(label_dir, s + ".txt") for s in self.stems]

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

#-----
#colate function and dataloaders
import os
def collate_fn (batch):
    images, targets = list (zip (*batch))
    images = torch.stack (images, dim = 0)

    all_boxes = []
    all_labels = []
    all_bidx = []
    image_ids = []
    scales = []
    pads = []
    orig_sizes = []

    for i, target in enumerate (targets):
        num_boxes = target ["boxes"].shape [0]

        if num_boxes:
            all_boxes.append (target ["boxes"])
            all_labels.append (target ["labels"])
            all_bidx.append (torch.full ((num_boxes,), i, dtype = torch.int64))

        image_ids.append (target ["image_id"])
        scales.append (target ["scale"])
        pads.append (target ["pad"])
        orig_sizes.append (target ["orig_size"])

    if len (all_boxes):
        boxes = torch.cat (all_boxes, 0)
        labels = torch.cat (all_labels, 0)
        batch_index = torch.cat (all_bidx, 0)
    else:
        boxes = torch.zeros ((0, 4), dtype = torch.float32)
        labels = torch.zeros ((0,), dtype = torch.int64)
        batch_index = torch.zeros ((0,), dtype = torch.int64)

    return images, {
        "boxes": boxes,
        "labels": labels,
        "batch_index": batch_index,
        "image_id": image_ids,
        "scale": scales,
        "pad": pads,
        "orig_size": orig_sizes,
    }

def _wif (worker_id):
  seed = CFG ["seed"] + worker_id
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed (seed)

VAL_IMG_DIR = os.path.join (CFG ["data_root"], CFG ["val_img_dir"])
VAL_LBL_DIR = os.path.join (CFG ["data_root"], CFG ["val_lbl_dir"])
def _count_imgs(p):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
    import glob, os
    return sum(len(glob.glob(os.path.join(p, e))) for e in exts)

if _count_imgs(VAL_IMG_DIR) == 0:
    print("[VAL] No images detected under", VAL_IMG_DIR, "→ staging now...")
    # in case user ran cells out of order, try staging once here too:
    try:
        stage_val_once()
    except NameError:
        print("[VAL] stage_val_once() not defined yet — run the FO cell first.")
        raise

val_ds = YoloDataset(
    VAL_IMG_DIR, VAL_LBL_DIR,
    imgsz=CFG["imgsz"], augment=False,
    pad_value=CFG["letterbox_pad"], horizontal_flip_prob=0.0
)
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=4,
    collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    worker_init_fn=_wif,
    generator=torch.Generator().manual_seed(CFG["seed"]),
)
print("val dataset:", len(val_ds))

TRAIN_IMG_DIR = os.path.join (CFG ["data_root"], CFG ["train_img_dir"])
TRAIN_LBL_DIR = os.path.join (CFG ["data_root"], CFG ["train_lbl_dir"])

if os.path.isdir (TRAIN_IMG_DIR) and os.path.isdir (TRAIN_LBL_DIR):

  def _count_imgs(p):
      exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
      import glob, os
      return sum(len(glob.glob(os.path.join(p, e))) for e in exts)

  if _count_imgs(TRAIN_IMG_DIR) == 0:
      print("[TRAIN] No images detected under", TRAIN_IMG_DIR, "→ staging chunk now...")
      try:
          stage_train_chunk()
      except NameError:
          print("[TRAIN] stage_train_chunk() not defined yet — run FO cell first.")
          raise

  train_ds = YoloDataset(
      TRAIN_IMG_DIR, TRAIN_LBL_DIR,
      imgsz=CFG["imgsz"], augment=True,
      pad_value=CFG["letterbox_pad"], horizontal_flip_prob=CFG["hflip_p"],
      hsv_hgain=CFG["hsv_h"], hsv_sgain=CFG["hsv_s"], hsv_vgain=CFG["hsv_v"]
  )
  train_loader = DataLoader(
      train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4,
      collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
      persistent_workers=False,
      worker_init_fn=_wif,
      generator=torch.Generator().manual_seed(CFG["seed"]),
  )

  xb, tb = next(iter(train_loader))
  print("Train batch:", xb.shape, xb.dtype)
  print("Targets keys:", list(tb.keys()))
else:
    train_loader = None
    print ("no training, train dataset not found")

# ==== SANITY CHECK CELL ====
# Verifies: staging, label pairing, dataloader shapes, letterbox coords, and cache bookkeeping.

import os, glob, random, math
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# --- paths from earlier cells (assumes you've defined these) ---
print("FO_CACHE_DIR:", FO_CACHE_DIR)
print("ROOT:", ROOT)
print("IMG_TRAIN:", IMG_TRAIN)
print("LBL_TRAIN:", LBL_TRAIN)
print("IMG_VAL:", IMG_VAL)
print("LBL_VAL:", LBL_VAL)
print("CURR_CHUNK_TXT:", CURR_CHUNK_TXT)
print("SEEN_TXT:", SEEN_TXT)

# 1) Basic file inventory
def count_files(dir_, exts=(".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP")):
    n = 0
    for e in exts:
        n += len(glob.glob(os.path.join(dir_, e)))
    return n

n_tr_img = count_files(IMG_TRAIN)
n_tr_lbl = len(glob.glob(os.path.join(LBL_TRAIN, "*.txt")))
n_va_img = count_files(IMG_VAL)
n_va_lbl = len(glob.glob(os.path.join(LBL_VAL, "*.txt")))

print(f"[STAGED] train: {n_tr_img} images, {n_tr_lbl} labels")
print(f"[STAGED]   val: {n_va_img} images, {n_va_lbl} labels")

# 2) Check label<->image stems & YOLO label range
def stems(dir_imgs):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
    P=[]
    for e in exts:
        P += glob.glob(os.path.join(dir_imgs, e))
    return sorted([os.path.splitext(os.path.basename(p))[0] for p in P])

def bad_yolo_lines(lbl_path):
    bad = 0
    with open(lbl_path, "r") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts=line.split()
            if len(parts)!=5:
                bad+=1; continue
            try:
                cls,cx,cy,w,h = parts
                cx,cy,w,h = map(float,(cx,cy,w,h))
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    bad += 1
            except:
                bad += 1
    return bad

tr_stems = stems(IMG_TRAIN)
va_stems = stems(IMG_VAL)

missing_train_lbl = [s for s in tr_stems if not os.path.exists(os.path.join(LBL_TRAIN, s + ".txt"))]
missing_val_lbl   = [s for s in va_stems if not os.path.exists(os.path.join(LBL_VAL,   s + ".txt"))]

print(f"[CHECK] missing train labels: {len(missing_train_lbl)}")
print(f"[CHECK] missing val   labels: {len(missing_val_lbl)}")

sample_lbls = glob.glob(os.path.join(LBL_TRAIN, "*.txt"))[:20]
range_issues = sum(bad_yolo_lines(p) for p in sample_lbls)
print(f"[CHECK] sampled train label lines out of [0,1] or malformed: {range_issues}")

# 3) Quick dataloader smoke test (uses your Dataset & collate_fn)
try:
    _train_ds = YoloDataset(IMG_TRAIN, LBL_TRAIN, imgsz=CFG["imgsz"], augment=False, pad_value=CFG["letterbox_pad"], horizontal_flip_prob=0.0)
    _val_ds   = YoloDataset(IMG_VAL,   LBL_VAL,   imgsz=CFG["imgsz"], augment=False, pad_value=CFG["letterbox_pad"], horizontal_flip_prob=0.0)

    _train_loader = torch.utils.data.DataLoader(
        _train_ds, batch_size=min(4, max(1, len(_train_ds))), shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
    )
    _val_loader = torch.utils.data.DataLoader(
        _val_ds, batch_size=min(4, max(1, len(_val_ds))), shuffle=False, num_workers=2,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
    )

    xb, tb = next(iter(_train_loader))
    print("[DL] train batch images:", tuple(xb.shape), xb.dtype)
    for k in ("boxes","labels","batch_index"):
        print(f"[DL] train targets[{k}]:", tuple(tb[k].shape) if hasattr(tb[k], "shape") else type(tb[k]))

    xbv, tbv = next(iter(_val_loader))
    print("[DL]   val batch images:", tuple(xbv.shape), xbv.dtype)

    # 4) Assert boxes are within letterboxed frame [0, imgsz]
    def box_bounds_ok(boxes, size):
        if boxes.numel()==0: return True
        x1,y1,x2,y2 = boxes.unbind(-1)
        return bool( (x1.min() >= 0) and (y1.min() >= 0) and (x2.max() <= size) and (y2.max() <= size) )

    ok_train = box_bounds_ok(tb["boxes"], CFG["imgsz"])
    ok_val   = box_bounds_ok(tbv["boxes"], CFG["imgsz"]) if isinstance(tbv["boxes"], torch.Tensor) else True
    print(f"[CHECK] boxes within letterbox bounds (train/val): {ok_train}/{ok_val}")

except AssertionError as e:
    print("Dataset assertion:", e)
    xb, tb, xbv, tbv = None, None, None, None

# 5) Visualize a few samples (letterboxed images with letterboxed boxes)
def show_batch(images, targets, num=4, size=CFG["imgsz"], title="train"):
    if images is None:
        print(f"[VIS] No {title} batch available");
        return
    B = images.shape[0]
    num = min(num, B)
    cols = min(2, num)
    rows = math.ceil(num/cols)
    plt.figure(figsize=(cols*5, rows*5))
    for i in range(num):
        img = images[i].numpy().transpose(1,2,0)  # CHW->HWC
        img = (np.clip(img,0,1)*255).astype(np.uint8)
        # collect boxes for this image
        mask = (targets["batch_index"]==i)
        boxes = targets["boxes"][mask].cpu().numpy() if torch.is_tensor(targets["boxes"]) else np.zeros((0,4))
        labels = targets["labels"][mask].cpu().numpy() if torch.is_tensor(targets["labels"]) else np.zeros((0,),dtype=int)

        # draw boxes
        canvas = img.copy()
        for (x1,y1,x2,y2), c in zip(boxes, labels):
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(canvas,str(int(c)),(x1,max(0,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)
        plt.subplot(rows, cols, i+1)
        plt.title(f"{title} #{i}  boxes:{len(boxes)}")
        plt.imshow(canvas)
        plt.axis("off")
    plt.show()

show_batch(xb,  tb,  num=4, title="train")
show_batch(xbv, tbv, num=4, title="val")

# 6) Current chunk bookkeeping
if os.path.exists(CURR_CHUNK_TXT):
    with open(CURR_CHUNK_TXT, "r") as f:
        curr_paths = [l.strip() for l in f if l.strip()]
    exists_in_cache = sum(os.path.exists(p) for p in curr_paths)
    print(f"[CHUNK] current chunk paths listed: {len(curr_paths)}; exist on disk: {exists_in_cache}")
else:
    print("[CHUNK] no CURR_CHUNK_TXT found yet")

if os.path.exists(SEEN_TXT):
    with open(SEEN_TXT,"r") as f:
        seen_count = sum(1 for _ in f)
    print(f"[SEEN] seen_filepaths.txt lines: {seen_count}")

print("✅ Sanity check done.")
# ==== END SANITY CHECK ====


#Stage 4: Finally the actual model, Backbone


#the fun part - backbone and fpn finally!!!!!!!!!!
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet50Backbone (nn.Module):
    #finally the fun part :))) - return c3, c4, c5 features map (strides 8, 16, 32)

    def __init__ (self, pretrained = False, freeze_bn = False):
        super ().__init__ ()

        try:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = torchvision.models.resnet50 (weights=weights)
        except AttributeError:
            model = torchvision.models.resnet50 (pretrained=pretrained)

        self.stem = nn.Sequential (
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1  #c2 stride 4
        self.layer2 = model.layer2  #c3 stride 8
        self.layer3 = model.layer3  #c4 stride 16
        self.layer4 = model.layer4  #c5 stride 32

        if freeze_bn:
            self._freeze_bn ()

    def _freeze_bn (self):
        for m in self.modules ():
            if isinstance (m, nn.BatchNorm2d):
                m.eval ()

                for p in m.parameters ():
                    p.requires_grad = False

    def forward (self, x):
        x = self.stem (x)
        x1 = self.layer1 (x)
        c3 = self.layer2 (x1)
        c4 = self.layer3 (c3)
        c5 = self.layer4 (c4)
        return c3, c4, c5

class ConvBNAct (nn.Module):
    def __init__ (self, c_in, c_out, k = 3, s = 1, p = None):
        super ().__init__ ()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d (c_in, c_out, k, s, p, bias = False)
        self.bn = nn.BatchNorm2d (c_out)
        self.act = nn.SiLU (inplace  = True)

    def forward (self, x):
        return self.act (self.bn (self.conv (x)))

class FPN (nn.Module):
    #minimal(yes i know minimal) produce p3, p4, p5 (all out_ch)
    #input (c3, c4, c5)

    def __init__ (self, c3, c4, c5, out_ch = 256):
        super ().__init__ ()

        self.l3 = nn.Conv2d (c3, out_ch, 1, 1, 0)
        self.l4 = nn.Conv2d (c4, out_ch, 1, 1, 0)
        self.l5 = nn.Conv2d (c5, out_ch, 1, 1, 0)

        self.p3 = ConvBNAct (out_ch, out_ch, 3, 1)
        self.p4 = ConvBNAct (out_ch, out_ch, 3, 1)
        self.p5 = ConvBNAct (out_ch, out_ch, 3, 1)

    def forward (self, c3, c4, c5):
        p5 = self.l5 (c5)
        p4 = self.l4 (c4) + F.interpolate (p5, size = c4.shape [-2:], mode = "nearest")
        p3 = self.l3 (c3) + F.interpolate (p4, size = c3.shape [-2:], mode = "nearest")

        p5 = self.p5 (p5)
        p4 = self.p4 (p4)
        p3 = self.p3 (p3)

        return p3, p4, p5


#models/head + wrapper
class YoloV8LiteHead (nn.Module):
    #decoupled head, anchor free, no objectness
    #box predicts LTRB distances (>= 0). Will add DFL Later

    def __init__ (self, in_ch = 256, num_classes = 80, hidden = 256, num_levels = 3):
        super ().__init__ ()
        self.num_classes = num_classes
        self.num_levels = num_levels

        def make_tower ():
            return nn.Sequential (
                ConvBNAct (in_ch, hidden, 3, 1),
                ConvBNAct (hidden, hidden, 3, 1),
            )

        self.cls_towers = nn.ModuleList ([make_tower () for _ in range (num_levels)])
        self.reg_towers = nn.ModuleList ([make_tower () for _ in range (num_levels)])

        self.cls_preds = nn.ModuleList ([nn.Conv2d (hidden, num_classes, 1) for _ in range (num_levels)])
        self.box_preds = nn.ModuleList ([nn.Conv2d (hidden, 4, 1) for _ in range (num_levels)])

    def forward (self, features):
        #features: list of [p3, p4, p5], each is [B, C, H, W]
        #return: cls_outs and box outs

        cls_outs = []
        box_outs = []

        for i, f in enumerate (features):
            cls_tower = self.cls_towers [i] (f)
            reg_tower = self.reg_towers [i] (f)
            cls_outs.append (self.cls_preds [i] (cls_tower))
            box_outs.append (nn.functional.softplus (self.box_preds [i] (reg_tower)))
            #incase jason doesn't know, softplus is ln (1 + exp(x)), always > 0
        return cls_outs, box_outs

class YoloModel (nn.Module):
    def __init__ (self, num_classes = 80, backbone = "resnet50_fpn", head_hidden = 256, fpn_out = 256, criterion = None):
        super ().__init__ ()
        assert backbone == "resnet50_fpn", "only resnet50_fpn wired in this mvp"
        self.backbone = ResNet50Backbone (pretrained = False)
        #channel dims for resnet50, c3, c4, c5
        self.neck = FPN (c3 = 512, c4 = 1024, c5 = 2048, out_ch = fpn_out)
        self.head = YoloV8LiteHead (in_ch = fpn_out, num_classes = num_classes, hidden = head_hidden, num_levels  = 3)
        self.strides = [8, 16, 32]
        self.criterion = criterion

    def forward(self, x, targets=None):
      c3, c4, c5 = self.backbone(x)
      p3, p4, p5 = self.neck(c3, c4, c5)
      cls_outs, box_outs = self.head([p3, p4, p5])
      head_out = {"features": [p3, p4, p5], "cls": cls_outs, "box": box_outs, "strides": self.strides}

      # Training path: compute & return standardized loss dict
      if self.training and targets is not None and hasattr(self, "criterion") and self.criterion is not None:
          losses, stats = self.criterion(head_out, targets)
          # Normalize shapes of (losses, stats) so train_step can rely on keys.

          # losses can be either a scalar tensor OR a dict; handle both
          if torch.is_tensor(losses):
              total_loss = losses
              # If stats might not contain components, give safe defaults:
              loss_box = stats.get("loss_box", total_loss.detach()*0)
              loss_cls = stats.get("loss_cls", total_loss.detach()*0)
              num_pos  = stats.get("num_pos",  0)
          elif isinstance(losses, dict):
              # try common keys
              total_loss = losses.get("loss") or losses.get("total_loss") or losses.get("overall") or sum([v for v in losses.values() if torch.is_tensor(v)])
              loss_box   = losses.get("loss_box",  stats.get("loss_box", total_loss.detach()*0))
              loss_cls   = losses.get("loss_cls",  stats.get("loss_cls", total_loss.detach()*0))
              num_pos    = losses.get("num_pos",   stats.get("num_pos",  0))
          else:
              raise TypeError("criterion must return Tensor or (dict, stats) with a total loss")

          return {
              "loss": total_loss,
              "loss_box": loss_box,
              "loss_cls": loss_cls,
              "num_pos": float(num_pos),   # keep as float for logging
              # keep raw head for optional extra logging if you want
              # "head_out": head_out,
          }

      # Inference path: return raw predictions
      return head_out


#Stage 5


#forward pass
import torch

#grid and decode utilities
def make_grid (features_h, features_w, stride, device):
    #return grid centers in input/letterbox ccoords
    #centes x and centers y both already multiplied by stride

    ys = torch.arange (features_h, device = device)
    xs = torch.arange (features_w, device = device)
    yy, xx = torch.meshgrid (ys, xs, indexing = "ij") #hxw
    cx = (xx + 0.5) * stride
    cy = (yy + 0.5) * stride
    return cx.reshape (-1), cy.reshape (-1)

def decode_letterbox_to_xyzy (letterbox, centers_x, centers_y):
    #(n, 4), ltrb in letterbox coords
    #(n,) centers x and y in input coords
    #returns xyxy: (n, 4)

    l, t, r, b = letterbox.unbind (-1)
    x1 = centers_x - l
    y1 = centers_y - t
    x2 = centers_x + r
    y2 = centers_y + b
    return torch.stack ([x1, y1, x2, y2], dim = -1)

def flatten_head_outputs (cls_outs, box_outs, strides, image_size):
    #cls_outs and box_outs are list of [B, C, H, W] and [B, 4, H, W]
    #returns per iamge flatteded tensors
    #all_scores: list of (N, C)
    #all_boxes: list of (N, 4)
    B = cls_outs [0].shape [0]
    device = cls_outs [0].device
    all_scores = [[] for _ in range (B)]
    all_boxes = [[] for _ in range (B)]

    for level, (cl, bx, s) in enumerate (zip (cls_outs, box_outs, strides)):
        B_, C, H, W = cl.shape
        assert B_ == B
        cl = cl.permute (0, 2, 3, 1).reshape (B, H * W, C)
        bx = bx.permute (0, 2, 3, 1).reshape (B, H * W, 4)

        cx, cy = make_grid (H, W, s, device)
        #exp to (B, HW)
        cx = cx.unsqueeze (0).expand (B, -1)
        cy = cy.unsqueeze (0).expand (B, -1)

        #decode
        for i in range (B):
            xyxy = decode_letterbox_to_xyzy (bx [i], cx [i], cy [i])

            xyxy [..., 0::2].clamp_ (0, image_size)
            xyxy [..., 1::2].clamp_ (0, image_size)
            #(hw,c)
            all_boxes [i].append (xyxy)
            #(hw, 4)
            all_scores [i].append (cl [i])

    #concat levels
    for i in range (B):
        all_boxes [i] = torch.cat (all_boxes [i], dim = 0)
        all_scores [i] = torch.cat (all_scores [i], dim = 0)

    return all_scores, all_boxes


import torch
import torchvision.ops as nms

#simple nms and postprocess

@torch.no_grad ()
def post_process_one (img_boxes, img_scores, score_thresh = 0.25, iou_thresh = 0.50, max_det = 300):
    #img_boxes: (N, 4) in xyxy format
    #img_scores: (N, C)
    #return: boxes (M, 4), scores (M,), labels (M,)

    C = img_scores.shape [1]
    probs = img_scores.sigmoid ()
    conf, cls = probs.max (dim = 1)
    keep = conf > score_thresh

    if keep.sum () == 0:
        return img_boxes.new_zeros ((0, 4)), conf.new_zeros ((0,)), cls.new_zeros ((0,), dtype = torch.long)

    boxes = img_boxes [keep]
    conf = conf [keep]
    cls = cls [keep]

    offsets = cls.to (boxes) * 4096
    offset_boxes = boxes + offsets [:, None]
    keep_index = nms.nms (offset_boxes, conf, iou_thresh)

    if len (keep_index) > max_det:
        keep_index = keep_index [:max_det]
    return boxes [keep_index], conf [keep_index], cls [keep_index]

@torch.no_grad ()
def model_inference_step (model, images, image_size = 640, score_thresh = 0.25, iou_thresh = 0.50, max_det = 300):
    #images is (B, 3, H, W)
    #return list of detections of each images

    out = model (images)
    cls_outs = out ["cls"]
    box_outs = out ["box"]
    strides = out ["strides"]

    scores_list, boxes_list = flatten_head_outputs (cls_outs, box_outs, strides, image_size)

    results = []
    for boxes, scores in zip (boxes_list, scores_list):
        b, s, c = post_process_one (boxes, scores, score_thresh, iou_thresh, max_det)
        results.append ({"boxes": b, "scores": s, "classes": c})
    return results

model.eval ()
try:
  # letterboxed to CFG["imgsz"]
    xbv, tbv = next (iter (val_loader))
except StopIteration:
    raise RuntimeError ("val_loader is empty; stage_val_once() may not have run")

xbv = xbv.to (device)
with torch.no_grad ():
    dets = model_inference_step (model, xbv, image_size=CFG["imgsz"], score_thresh=0.25, iou_thresh=0.50, max_det=100)

print (f"[SMOKE] got {len(dets)} detection lists for batch size {xbv.shape [0]}")
for i, d in enumerate (dets [:2]):
    print(f" img{i}: boxes {tuple (d ['boxes'].shape)}  scores {tuple (d ['scores'].shape)}  classes {tuple(d['classes'].shape)}")

#Stage 6


#box ops (IoU)

def box_iou_xyxy (a, b):
    #a: (N, 4), b (N, 4) pair IoU in xyxy
    #return IoU (N, )

    x1 = torch.max (a [:, 0], b [:, 0])
    y1 = torch.max (a [:, 1], b [:, 1])
    x2 = torch.min (a [:, 2], b [:, 2])
    y2 = torch.min (a [:, 3], b [:, 3])
    inter = (x2 - x1).clamp (min = 0) * (y2 - y1).clamp (min = 0)
    area_a = (a [:, 2] - a [:, 0]).clamp (min = 0) * (a [:, 3] - a [:, 1]).clamp (min = 0)
    area_b = (b [:, 2] - b [:, 0]).clamp (min = 0) * (b [:, 3] - b [:, 1]).clamp (min = 0)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def iou_loss (predict_xyxy, target_xyxy):
    return 1.0 - box_iou_xyxy (predict_xyxy, target_xyxy)

#center-prior assigner (top-k) + target builder

def _make_level_grids (image_size, strides, device):
    levels = []

    for s in strides:
        H = image_size // s
        W = image_size // s
        ys = torch.arange (H, device = device)
        xs = torch.arange (W, device = device)
        yy, xx = torch.meshgrid (ys, xs, indexing = "ij")
        cx = (xx + 0.5) * s
        cy = (yy + 0.5) * s

        levels.append ({
            "H": H, "W": W, "stride": s,
            "cx": cx.reshape (-1),
            "cy": cy.reshape (-1),
        })
    return levels

@torch.no_grad ()
def build_targets_center_prior (targets, num_classes: int, image_size: int, strides: list, topk: int = 10, center_radius: float = 2.5, device = None):
    #returns per-image flatted targets for all locations across levels
    #ret t_cls (N, c) multi hot(most 0, 1 for pos class)
    #ret t_box_letterbox (N, 4) dis to GT edges (pos); 0 for negative
    #position_mask: (N, ) bool where pos are true
    #matched_gt_inds: (N, ) long (indx into per-img gt lst or -1)
    #n = sum_l (Hl * Wl)

    B = len (targets ["image_id"])
    if device is None:
        device = targets ["boxes"].device if torch.is_tensor (targets ["boxes"]) else torch.device ("cpu")

    levels = _make_level_grids (image_size, strides, device)
    per_image = []

    M = targets ["boxes"].shape [0]
    bidx = targets ["batch_index"].to (device) if torch.is_tensor (targets ["batch_index"]) else torch.tensor ([], dtype = torch.long, device = device)
    boxes_all = targets ["boxes"].to (device).float () if M else torch.zeros ((0, 4), device = device)
    labels_all = targets ["labels"].to (device).long () if M else torch.zeros ((0,), dtype = torch.long, device = device)

    for i in range (B):
        sel = (bidx == i)
        gt = boxes_all [sel]
        gc = labels_all [sel]
        Gi = gt.shape [0]

        centers_x = torch.cat ([L ["cx"] for L in levels], dim = 0)
        centers_y = torch.cat ([L ["cy"] for L in levels], dim = 0)
        N = centers_x.numel ()

        position_mask = torch.zeros (N, dtype = torch.bool, device = device)
        t_box_letterbox = torch.zeros (N, 4, device = device)
        t_cls = torch.zeros (N, num_classes, device = device)
        matched_gt_index = torch.full ((N,), -1, dtype = torch.long, device = device)

        if Gi == 0:
            per_image.append ((t_cls, t_box_letterbox, position_mask, matched_gt_index))
            continue

        start = 0
        for L in levels:
            radius = center_radius * L ["stride"]
            HW = L ["H"] * L ["W"]
            #(HW, )
            cx = L ["cx"]
            cy = L ["cy"]

            gx1 = (gt [:, 0]- radius).clamp_ (0, image_size)
            gy1 = (gt [:, 1]- radius).clamp_ (0, image_size)
            gx2 = (gt [:, 2]+ radius).clamp_ (0, image_size)
            gy2 = (gt [:, 3]+ radius).clamp_ (0, image_size)

            in_x = (cx.unsqueeze (0) >= gx1.unsqueeze (1)) & (cx.unsqueeze (0) <= gx2.unsqueeze (1))
            in_y = (cy.unsqueeze (0) >= gy1.unsqueeze (1)) & (cy.unsqueeze (0) <= gy2.unsqueeze (1))
            inside = in_x & in_y

            if inside.any ():
                gcx = (gt [:, 0] + gt [:, 2]) / 2
                gcy = (gt [:, 1] + gt [:, 3]) / 2
                dx = torch.abs (cx.unsqueeze (0) - gcx.unsqueeze (1))
                dy = torch.abs (cy.unsqueeze (0) - gcy.unsqueeze (1))
                distance = dx + dy

                distance_masked = torch.where (inside, distance, torch.full_like (distance, 1e9))
                k = min (topk, HW)
                _, indexs = torch.topk (-distance_masked, k = k, dim = 1)
                flat_index = indexs + start
                position_mask [flat_index.reshape (-1)] = True

                matched_gt_index [flat_index.reshape (-1)] = torch.arange (Gi, device = device).unsqueeze (1).expand (-1, k).reshape (-1)

            start += HW

        pos_index = torch.nonzero (position_mask, as_tuple = False).flatten ()

        if pos_index.numel ():
            mg: torch.Tensor = matched_gt_index [pos_index]
            gt_sel = gt [mg]

            centers_x_all = torch.cat ([L ["cx"] for L in levels], dim = 0)
            centers_y_all = torch.cat ([L ["cy"] for L in levels], dim = 0)
            cxp = centers_x_all [pos_index]
            cyp = centers_y_all [pos_index]

            l = (cxp - gt_sel [:, 0]).clamp_min (0)
            t = (cyp - gt_sel [:, 1]).clamp_min (0)
            r = (gt_sel [:, 2] - cxp).clamp_min (0)
            b = (gt_sel [:, 3] - cyp).clamp_min (0)
            t_box_letterbox [pos_index] = torch.stack ([l, t, r, b], dim = 1)

            t_cls [pos_index, gc [mg]] = 1.0

        per_image.append ((t_cls, t_box_letterbox, position_mask, matched_gt_index))

    return per_image, levels

#loss writing

class DetectionLoss (nn.Module):
    def __init__ (self, num_classes, image_size, strides, lambda_box = 2.5, lambda_cls = 1.0):
        super ().__init__ ()
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls

    def forward (self, model_out, targets):
        #out: {"cls": [L], "box"...}
        #cls [l] = (B,C,H,W) logits
        #box [l] = (B,4,H,W) LTRB dist >= 0
        #targets: dict from collate_fn

        B = model_out ["cls"][0].shape[0]
        device = model_out ["cls"][0].device

        scores_list, boxes_list = flatten_head_outputs (model_out ["cls"], model_out ["box"], self.strides, self.image_size)

        per_image_targets, _ = build_targets_center_prior (targets, num_classes = self.num_classes, image_size = self.image_size, strides = self.strides, device = device)

        total_cls = torch.tensor (0.0, device = device)
        total_box = torch.tensor (0.0, device = device)
        total_pos = 0

        for i in range (B):
            logits = scores_list [i]
            predict_ltrb = None
            predict_xyxy = boxes_list [i]

            t_cls, t_ltrb, pos_mask, matched_index = per_image_targets [i]
            N = logits.shape [0]

            cls_loss = F.binary_cross_entropy_with_logits (logits, t_cls, reduction = "none") / max (N, 1)
            total_cls += cls_loss.sum ()

            if pos_mask.any ():
                predict_xy = predict_xyxy [pos_mask]

                levels = _make_level_grids (self.image_size, self.strides, device)
                centers_x_all = torch.cat ([L ["cx"] for L in levels], dim = 0)
                centers_y_all = torch.cat ([L ["cy"] for L in levels], dim = 0)
                pos_index = torch.nonzero (pos_mask, as_tuple = False).flatten ()
                cxp = centers_x_all [pos_index]
                cyp = centers_y_all [pos_index]
                ltrb = t_ltrb [pos_mask]
                l, t, r, b = ltrb.unbind (-1)
                target_xy = torch.stack ([cxp - l, cyp - t, cxp + r, cyp + b], dim = -1)

                box_loss = iou_loss (predict_xy, target_xy).mean ()
                total_box += box_loss
                total_pos += ltrb.shape [0]

        total = self.lambda_cls * total_cls + self.lambda_box * total_box

        stats = {
            "loss": float (total.detach ().item ()),
            "loss_cls": float (total_cls.detach ().item ()),
            "loss_box": float (total_box.detach ().item ()),
            "num_pos": int (total_pos),
        }
        return total, stats

from torch.amp import GradScaler, autocast


#optimizer + 1 training step with AMP

def build_optimizer (model, cfg):
    if cfg ["optimizer"].lower () == "adamw":
        optimizer = torch.optim.AdamW (model.parameters (), lr = cfg ["lr"], weight_decay = cfg ["weight_decay"])
    else:
        optimizer = torch.optim.SGD (model.parameters (), lr = cfg ["lr"], momentum = 0.9, weight_decay = cfg ["weight_decay"], nesterov = True)

    if cfg.get ("cosine_schedule", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR (optimizer, T_max = cfg ["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR (optimizer, milestones = [int (0.7 * cfg ["epochs"])], gamma = 0.1)

    return optimizer, scheduler

scaler = GradScaler (enabled = CFG ["amp"])

def build_model(cfg = CFG, device_arg = device):
    """Construct a ``YoloModel`` along with its loss criterion.

    Keeping the helper makes it easy for external scripts (like ``YoloMain``)
    to import this module and request a fresh model instance without relying
    on globals that may or may not have executed yet.
    """

    criterion = DetectionLoss (
        num_classes = cfg ["num_classes"],
        image_size = cfg ["imgsz"],
        strides = [8, 16, 32],
        lambda_box = cfg ["loss_weights"] ["box"],
        lambda_cls = cfg ["loss_weights"] ["cls"],
    )

    model = YoloModel (
        num_classes = cfg ["num_classes"],
        backbone = cfg ["backbone"],
        head_hidden = cfg ["head_hidden"],
        fpn_out = 256,
        criterion = criterion,
    ).to (device_arg)

    return model, criterion


model, criterion = build_model ()

#forward smoke test
x = torch.randn (2, 3, CFG ["imgsz"], CFG ["imgsz"], device = device)
with torch.no_grad ():
    out = model (x)

print ("Levels:", len (out ["features"]))

for i, (c, b) in enumerate (zip (out ["cls"], out ["box"])):
    print (f"Level {i}: cls: {tuple (c.shape)}, box: {tuple (b.shape)}, stride: {model.strides [i]}")

model = YoloModel (
    num_classes = CFG ["num_classes"],
    backbone = CFG ["backbone"],
    head_hidden = CFG ["head_hidden"],
    fpn_out = 256,
    criterion = criterion,
).to (device)

#forward smoke test
x = torch.randn (2, 3, CFG ["imgsz"], CFG ["imgsz"], device = device)
with torch.no_grad ():
    out = model (x)

print ("Levels:", len (out ["features"]))

for i, (c, b) in enumerate (zip (out ["cls"], out ["box"])):
    print (f"Level {i}: cls: {tuple (c.shape)}, box: {tuple (b.shape)}, stride: {model.strides [i]}")

optimizer, scheduler = build_optimizer (model, CFG)

def train_step (model, batch, device):
    model.train ()
    images, targets = batch
    images = images.to (device, non_blocking = True)

    optimizer.zero_grad (set_to_none = True)

    with torch.cuda.amp.autocast (enabled = CFG.get ("amp", True)):
        out = model (images, targets)
        loss = out ["loss"]
        loss_box = out.get ("loss_box", 0.0)
        loss_cls = out.get ("loss_cls", 0.0)
        num_pos = out.get ("num_pos", 0.0)

        if not torch.is_tensor (loss_box):
            loss_box = torch.as_tensor (loss_box, device = loss.device, dtype = loss.dtype)
        if not torch.is_tensor (loss_cls):
            loss_cls = torch.as_tensor (loss_cls, device = loss.device, dtype = loss.dtype)
        if not torch.is_tensor (num_pos):
            num_pos = torch.as_tensor (num_pos, device = loss.device, dtype = loss.dtype)

    scaler.scale (loss).backward ()

    if CFG.get ("grad_clip_norm", None):
        scaler.unscale_ (optimizer)
        torch.nn.utils.clip_grad_norm_ (model.parameters (), CFG ["grad_clip_norm"])

    scaler.step (optimizer)
    scaler.update ()

    stats = {
        "loss": float (loss.detach ().item ()),
        "loss_box": float (loss_box.detach ().item ()),
        "loss_cls": float (loss_cls.detach ().item ()),
        "num_pos": float (num_pos.detach ().item ())
    }
    return stats

#ema
import copy

class ModelEMA:
    def __init__ (self, model, decay = 0.9998, device = None):
        self.module = copy.deepcopy (model).eval ()
        for p in self.module.parameters ():
            p.requires_grad_ (False)
        self.decay = decay
        if device:
            self.module.to (device)

    @torch.no_grad ()
    def update (self, model):
        msd = model.state_dict ()
        esd = self.module.state_dict ()
        d = self.decay
        for k in esd.keys():
          if esd[k].dtype.is_floating_point:
              esd[k].mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
          else:
              esd[k].copy_(msd[k])

ema = ModelEMA (model, decay = CFG ["ema_decay"], device = device)
print ("EMA ready with decay", CFG ["ema_decay"])

#overfit a tiny subset to check learning

from collections import deque
import time

tiny_count = min (400, len (train_ds))
indices = torch.randperm (len (train_ds)) [:tiny_count].tolist ()
tiny_subset = torch.utils.data.Subset (train_ds, indices)
tiny_loader = DataLoader (
    tiny_subset, batch_size = min (CFG ["batch_size"], 8), shuffle = True, num_workers = 2, pin_memory = torch.cuda.is_available (), collate_fn = collate_fn
)

print (f"[SANITY] Overfitting tiny subset: {len (tiny_subset)} iamges")

history = deque (maxlen = 50)

for epoch in range (3):
    t0 = time.time ()

    for it, batch in enumerate (tiny_loader):
        stats = train_step (model, batch, device)
        ema.update (model)
        history.append (stats ["loss"])

        if (it + 1) % 10 == 0:
            print (f"epoch {epoch+1} iter {it+1}, loss = {stats ['loss']:.4f}, box = {stats ['loss_box']:.4f}, cls = {stats ['loss_cls']:.4f}, pos = {stats ['num_pos']}, time = {time.time () - t0:.2f}s")

    scheduler.step ()
    print (f"epoch {epoch+1} done, avg loss = {sum (history) / len (history):.4f}, time = {time.time () - t0:.2f}s")

#Stage 7


#loader builders

def build_loader (image_dir, label_dir, cfg, shuffle):
    ds = YoloDataset (
        image_dir, label_dir,
        imgsz = cfg ["imgsz"],
        augment = shuffle,
        pad_value = cfg ["letterbox_pad"],
        horizontal_flip_prob = (cfg ["hflip_p"] if shuffle else 0.0),
        hsv_hgain = cfg.get ("hsv_h", 0.0),
        hsv_sgain = cfg.get ("hsv_s", 0.0),
        hsv_vgain = cfg.get ("hsv_v", 0.0),
    )

    dl = DataLoader (
        ds, batch_size = cfg ["batch_size"], shuffle = shuffle, num_workers = 4,
        collate_fn = collate_fn, pin_memory = torch.cuda.is_available (),
        persistent_workers = torch.Generator ().manual_seed (cfg ["seed"]),
    )
    return ds, dl

VAL_IMG_DIR = os.path.join (CFG ["data_root"], CFG ["val_img_dir"])
VAL_LBL_DIR = os.path.join (CFG ["data_root"], CFG ["val_lbl_dir"])
val_ds, val_loader = build_loader (VAL_IMG_DIR, VAL_LBL_DIR, CFG, shuffle = False)
print ("val size:", len (val_ds))

#-----
#coco mAP eval
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _image_size_from_stem (stem):
    #try common image extensions

    for e in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        p = os.path.join (VAL_IMG_DIR, stem + e)

        if os.path.exists (p):
            im = val_ds._read_image (p)
            return im.shape [:2]
    raise FileNotFoundError (f"no image for {stem}")

@torch.no_grad ()
def evaluate_map_coco (model, val_loader, device, imgsz = 640, score_thresh = 0.001):
    categories = [{"id": i, "name": n} for i, n in enumerate (COCO_NAMES)]
    coco_gt = {"images" : [], "annotations": [], "categories": categories}
    ann_id = 1

    for idx, stem in enumerate (val_loader.dataset.stems):
        H, W = _image_size_from_stem (stem)
        coco_gt ["images"].append ({"id": idx, "height": H, "width": W, "file_name": stem})
        lp = os.path.join (VAL_LBL_DIR, stem + ".txt")

        if os.path.exists (lp):
            arr = val_loader.dataset._read_labels (lp)
            if arr.size:
                boxes, cls = yolo_to_xyxy (arr, W, H)
                for j , (x1, y1, x2, y2) in enumerate (boxes):
                    coco_gt ["annotations"].append ({
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": int (cls [j]),
                        "bbox": [float (x1), float (y1), float (x2 - x1), float (y2 - y1)],
                        "area": float (max (0.0, (x2 - x1)) * max (0.0, (y2 - y1))),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    cocoGt = COCO ()
    cocoGt.dataset = coco_gt
    cocoGt.createIndex ()

    #model inference
    dets_json = []
    model.eval ()
    for images, targets in val_loader:
        images = images.to (device, non_blocking = True)
        outs = model_inference_step (
            model,
            images,
            image_size = imgsz,
            score_thresh = score_thresh,
            iou_thresh = 0.7,
            max_det = 300,
        )

        for batch_index, det in enumerate (outs):
            stem = targets ["image_id"][batch_index]
            image_index = val_loader.dataset.stems.index (stem)
            H, W = targets ["orig_size"][batch_index]
            scale = targets ["scale"][batch_index]
            pad = targets ["pad"][batch_index]

            if det ["boxes"].numel () == 0:
                continue

            boxes_original = undo_letterbox_to_orig (
                det ["boxes"].clone (), pad, scale, (H, W)
            )
            scores = det ["scores"].tolist ()
            classes = det ["classes"].tolist ()

            for prediction_index in range (boxes_original.shape [0]):
                x1, y1, x2, y2 = boxes_original [prediction_index].tolist ()
                dets_json.append ({
                    "image_id": image_index,
                    "category_id": int (classes [prediction_index]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float (scores [prediction_index]),
                })

    if len (dets_json):
        cocoDt = cocoGt.loadRes (dets_json)
    else:
        cocoDt = cocoGt.loadRes ([])
    cocoEval = COCOeval (cocoGt, cocoDt, iouType = "bbox")
    cocoEval.params.useCats
    cocoEval.evaluate ()
    cocoEval.accumulate ()
    cocoEval.summarize ()
    return float (cocoEval.stats [0]) if cocoEval.stats is not None else 0.0

#resume helper

def load_checkpoint (path, model, ema = None, optimizer = None, scheduler = None, device = "cuda"):
    checkpoint = torch.load (path, map_location = device)
    model.load_state_dict (checkpoint ["model"])
    if ema and "ema" in checkpoint:
        ema.module.load_state_dict (checkpoint ["ema"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict (checkpoint ["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict (checkpoint ["scheduler"])
    print ("Loaded checkpoint:", path)

# === cell7_train_with_viz: EMA + loaders + realtime viz + training + eval ===
import os, copy, glob, math, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display, clear_output

VIZ_EVERY = int(CFG.get("viz_every", 200))   # inline viz frequency (steps). Set 0 to disable
VIZ_MAX_IMAGES = int(CFG.get("viz_max_images", 4))
SAVE_EVERY_EPOCH = True                      # always save per-epoch ckpt
EVAL_EVERY_EPOCH = True                      # run mAP eval each epoch (skips if pycocotools missing)

# ---------- 1) EMA ----------
class ModelEMA:
    def __init__(self, model, decay=0.9998, device=None):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        if device:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.module.state_dict()
        d = self.decay
        for k in esd.keys():
          if esd[k].dtype.is_floating_point:
              esd[k].mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
          else:
              esd[k].copy_(msd[k])

# instantiate EMA
ema = ModelEMA(model, decay=CFG.get("ema_decay", 0.9998), device=device)

# ---------- 2) Dataloader builders (uses your YoloDataset, collate_fn, _wif) ----------
def build_loader(img_dir, lbl_dir, cfg, shuffle):
    ds = YoloDataset(
        img_dir, lbl_dir,
        imgsz=cfg["imgsz"], augment=shuffle,
        pad_value=cfg["letterbox_pad"],
        horizontal_flip_prob=(cfg.get("hflip_p", 0.5) if shuffle else 0.0),
        hsv_hgain=cfg.get("hsv_h", 0.0),
        hsv_sgain=cfg.get("hsv_s", 0.0),
        hsv_vgain=cfg.get("hsv_v", 0.0),
    )
    dl = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=4,
        collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
        persistent_workers=False, worker_init_fn=_wif,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    return ds, dl

VAL_IMG_DIR = os.path.join(CFG["data_root"], CFG["val_img_dir"])
VAL_LBL_DIR = os.path.join(CFG["data_root"], CFG["val_lbl_dir"])
val_ds, val_loader = build_loader(VAL_IMG_DIR, VAL_LBL_DIR, CFG, shuffle=False)

# ---------- 3) Letterbox undo + small helpers ----------
def undo_letterbox_to_orig(xyxy, pad, scale, orig_size):
    # xyxy in letterbox space -> original image coords
    px, py = pad
    x1 = (xyxy[:,0] - px) / scale
    y1 = (xyxy[:,1] - py) / scale
    x2 = (xyxy[:,2] - px) / scale
    y2 = (xyxy[:,3] - py) / scale
    H, W = orig_size
    x1.clamp_(0, W-1); x2.clamp_(0, W-1)
    y1.clamp_(0, H-1); y2.clamp_(0, H-1)
    return torch.stack([x1,y1,x2,y2], dim=1)

def _find_img_path(stem, img_dir):
    for e in (".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"):
        p = os.path.join(img_dir, stem + e)
        if os.path.exists(p):
            return p
    return None

def _draw_boxes(ax, boxes, labels=None, scores=None, title=None):
    ax.set_axis_off()
    if title: ax.set_title(title, fontsize=10)
    for i,(x1,y1,x2,y2) in enumerate(boxes):
        rect = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        tag = None
        if labels is not None:
            cid = int(labels[i])
            if 0 <= cid < len(COCO_NAMES):
                tag = COCO_NAMES[cid]
            else:
                tag = str(cid)
        if scores is not None:
            s = f"{scores[i]:.2f}"
            tag = f"{tag} {s}" if tag else s
        if tag:
            ax.text(x1, max(0, y1-2), tag, fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# ---------- 4) Live inline visualization (Pred LEFT vs GT RIGHT on ORIGINAL image) ----------
def show_pred_vs_gt_inline(model, batch, dataset, img_dir, lbl_dir, max_images=4):
    model.eval()
    images, targets = batch
    dev = next(model.parameters()).device
    images_dev = images.to(dev, non_blocking=True)

    outs = model_inference_step(model, images_dev, image_size=CFG["imgsz"],
                                score_thresh=0.001, iou_thresh=0.7, max_det=200)

    B = images.shape[0]
    nshow = min(B, max_images)

    fig_w = 10
    fig_h = max(6, nshow * 3)
    fig, axes = plt.subplots(nshow, 2, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(nshow):
        stem = targets["image_id"][i]
        H, W = targets["orig_size"][i]
        scale = targets["scale"][i]
        pad = targets["pad"][i]

        # load original image
        p = _find_img_path(stem, img_dir)
        if p is None:
            # fallback to letterboxed tensor
            img_disp = (images[i].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        else:
            img_disp = dataset._read_image(p)

        # preds -> original coords
        if outs[i]["boxes"].numel() > 0:
            pred_boxes_orig = undo_letterbox_to_orig(outs[i]["boxes"].detach().cpu(), pad, scale, (H, W)).numpy()
            pred_scores = outs[i]["scores"].detach().cpu().numpy().tolist()
            pred_classes = outs[i]["classes"].detach().cpu().numpy().tolist()
        else:
            pred_boxes_orig = np.zeros((0,4), dtype=np.float32)
            pred_scores, pred_classes = [], []

        # gt from txt
        gt_boxes_orig = np.zeros((0,4), dtype=np.float32)
        gt_classes = []
        lblp = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(lblp):
            arr = dataset._read_labels(lblp)
            if arr.size:
                bxyxy, cls = yolo_to_xyxy(arr, W, H)
                gt_boxes_orig = np.array(bxyxy, dtype=np.float32)
                gt_classes = cls.astype(np.int32).tolist()

        # LEFT: preds
        axL = axes[i,0]; axL.imshow(img_disp)
        _draw_boxes(axL, pred_boxes_orig, labels=pred_classes, scores=pred_scores, title=f"{stem} — Pred")
        # RIGHT: GT
        axR = axes[i,1]; axR.imshow(img_disp)
        _draw_boxes(axR, gt_boxes_orig, labels=gt_classes, scores=None, title=f"{stem} — GT")

    plt.tight_layout()
    clear_output(wait=True)
    display(fig)
    plt.close(fig)

# ---------- 5) (Optional) COCO mAP evaluation (skips gracefully if pycocotools missing) ----------
def evaluate_map_coco_safe(model, val_loader, device):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("[EVAL] pycocotools not available, skipping mAP eval.")
        return None

    categories = [{"id": i, "name": n} for i, n in enumerate(COCO_NAMES)]
    coco_gt = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    for idx, stem in enumerate(val_loader.dataset.stems):
        im_path = _find_img_path(stem, VAL_IMG_DIR)
        if im_path is None:
            continue
        im = val_loader.dataset._read_image(im_path)
        H, W = im.shape[:2]
        coco_gt["images"].append({"id": idx, "file_name": stem, "height": H, "width": W})
        lp = os.path.join(VAL_LBL_DIR, stem + ".txt")
        if os.path.exists(lp):
            arr = val_loader.dataset._read_labels(lp)
            if arr.size:
                boxes, cls = yolo_to_xyxy(arr, W, H)
                for j,(x1,y1,x2,y2) in enumerate(boxes):
                    coco_gt["annotations"].append({
                        "id": ann_id, "image_id": idx, "category_id": int(cls[j]),
                        "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        "area": float(max(0.0, (x2-x1))*max(0.0, (y2-y1))),
                        "iscrowd": 0
                    })
                    ann_id += 1
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cocoGt = COCO()
    cocoGt.dataset = coco_gt
    cocoGt.createIndex()

    dets_json = []
    model.eval()
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        outs = model_inference_step(model, images, image_size=CFG["imgsz"],
                                    score_thresh=0.001, iou_thresh=0.7, max_det=300)
        for b, det in enumerate(outs):
            stem = targets["image_id"][b]
            try:
                img_idx = val_loader.dataset.stems.index(stem)
            except:
                continue
            H, W = targets["orig_size"][b]
            scale = targets["scale"][b]; pad = targets["pad"][b]
            if det["boxes"].numel() == 0:
                continue
            bx = undo_letterbox_to_orig(det["boxes"].detach().cpu(), pad, scale, (H, W)).numpy()
            sc = det["scores"].detach().cpu().numpy().tolist()
            cl = det["classes"].detach().cpu().numpy().tolist()
            for k in range(bx.shape[0]):
                x1,y1,x2,y2 = bx[k].tolist()
                dets_json.append({
                    "image_id": img_idx,
                    "category_id": int(cl[k]),
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "score": float(sc[k]),
                })

    cocoDt = cocoGt.loadRes(dets_json) if len(dets_json) else cocoGt.loadRes([])
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.useCats = 1
    cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
    return float(cocoEval.stats[0]) if cocoEval.stats is not None else None

global_step = 0

def one_epoch(model, loader, device, log_every=50):
    global global_step
    model.train()
    meters = {"loss":0.0, "loss_box":0.0, "loss_cls":0.0, "num_pos":0, "n":0}

    steps = len(loader)
    warmup_epochs = int(CFG.get("warmup_epochs", 0))
    for it, batch in enumerate(loader):
        stats = train_step(model, batch, device)
        ema.update(model)
        global_step += 1

        # inline viz
        if VIZ_EVERY and ((it + 1) % VIZ_EVERY == 0):
            # show predictions vs GT from the *current batch*
            show_pred_vs_gt_inline(ema.module, (batch[0].cpu(), batch[1]), loader.dataset,
                                   TRAIN_IMG_DIR, TRAIN_LBL_DIR, max_images=VIZ_MAX_IMAGES)

        for k in ("loss","loss_box","loss_cls","num_pos"):
            meters[k] += stats[k]
        meters["n"] += 1
        if (it+1) % log_every == 0:
            n = meters["n"]
            print(f'   it {it+1:>4}/{steps} | loss {meters["loss"]/n:.4f}  '
                  f'box {meters["loss_box"]/n:.4f}  cls {meters["loss_cls"]/n:.4f}  '
                  f'pos {meters["num_pos"]/n:.1f}')

    n = max(1, meters["n"])
    return {k:(meters[k]/n) for k in meters if k!="n"}

# ---------- 7) Main training loop (chunk → train → dispose → eval → save) ----------
best_map = -1.0
os.makedirs(DIRS["checkpoints"], exist_ok=True)

for epoch in range(CFG["epochs"]):
    print(f"\n=== EPOCH {epoch+1}/{CFG['epochs']} ===")

    # Ensure a train chunk is staged
    if _count_imgs(TRAIN_IMG_DIR) == 0:
        stage_train_chunk()

    # Build train loader fresh each epoch (new chunk)
    train_ds, train_loader = build_loader(TRAIN_IMG_DIR, TRAIN_LBL_DIR, CFG, shuffle=True)

    # light LR warmup across first few epochs
    if epoch < CFG.get("warmup_epochs", 0):
        for g in optimizer.param_groups:
            g["lr"] = CFG["lr"] * (epoch + 1) / max(1, CFG["warmup_epochs"])

    stats = one_epoch(model, train_loader, device)
    # step epoch scheduler (if you're using epoch-level Cosine)
    if 'scheduler' in globals() and scheduler is not None:
        try:
            scheduler.step()
        except Exception as e:
            pass

    # rotate chunk to keep disk usage low
    dispose_current_chunk(delete_from_cache=True, mark_seen=True)

    # Evaluate (mAP)
    map5095 = None
    if EVAL_EVERY_EPOCH:
        map5095 = evaluate_map_coco_safe(ema.module, val_loader, device)
        if map5095 is not None:
            print(f"[EVAL] mAP50-95: {map5095:.4f}")

    # Save ckpts
    ckpt_base = os.path.join(DIRS["checkpoints"], f"{RUN_NAME}_e{epoch+1:03d}.pt")
    if SAVE_EVERY_EPOCH:
        torch.save({
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if ('scheduler' in globals() and scheduler is not None) else None),
            "cfg": CFG
        }, ckpt_base)
        print("Saved:", ckpt_base)

    if map5095 is not None and map5095 > best_map:
        best_map = map5095
        best_path = os.path.join(DIRS["checkpoints"], f"{RUN_NAME}_best.pt")
        torch.save({
            "model": model.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if ('scheduler' in globals() and scheduler is not None) else None),
            "cfg": CFG
        }, best_path)
        print("↑ Saved BEST:", best_path)

print("\nTraining loop finished.")
