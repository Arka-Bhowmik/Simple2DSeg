from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
#
MISSING_TEXT = {"", "nan", "<missing>", "none", "null"}
#
def clean_path(value) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in MISSING_TEXT else s
#
@dataclass
class Case:
    row_index: int
    image_path: Path
    mask_save_path: Path
    segmented_path: Path
    existing_mask_path: Optional[Path]
#
class CSVWorkflow:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.df = pd.read_csv(self.csv_path)
        # File_path is the only mandatory column.
        if "File_path" not in self.df.columns:
            raise ValueError('Required CSV column "File_path" was not found.')
        #
        csv_updated = False
        # If Segment is missing, add it as empty.
        if "Segment" not in self.df.columns:
            self.df["Segment"] = pd.NA
            csv_updated = True
        #
        # If Mask_path is missing, add it as empty.
        if "Mask_path" not in self.df.columns:
            self.df["Mask_path"] = pd.NA
            csv_updated = True
        #
        # Save back immediately if we added any missing columns.
        if csv_updated:
            self.df.to_csv(self.csv_path, index=False)
        #
        # Now Mask_path is guaranteed to exist.
        self.has_mask_path = True
        # Convert Segment values to numeric:
        # blank/invalid -> NaN -> unprocessed
        self.df["Segment"] = pd.to_numeric(self.df["Segment"], errors="coerce")
    #
    def next_unprocessed(self, start: int = 0) -> Optional[Case]:
        for i in range(max(0, start), len(self.df)):
            if not pd.isna(self.df.at[i, "Segment"]):
                continue
            image_text = clean_path(self.df.at[i, "File_path"])
            if not image_text:
                print(f"Warning: row {i+1} has an empty File_path. Skipping this row.")
                continue
            image_path = Path(image_text).expanduser()
            if not image_path.is_file():
                print(f"Warning: original image does not exist: {image_path}. Skipping row {i+1}.")
                continue
            #
            ext = image_path.suffix
            default_mask = image_path.with_name(image_path.stem + "_mask" + ext)
            segmented = image_path.with_name(image_path.stem + "_sub" + ext)
            existing = None
            save_path = default_mask
            if self.has_mask_path:
                mask_text = clean_path(self.df.at[i, "Mask_path"])
                if mask_text:
                    save_path = Path(mask_text).expanduser()
                    # Keep the supplied path even if the file is absent: main.m saves the new/edited mask there.
                    existing = save_path
            return Case(i, image_path, save_path, segmented, existing)
        return None
    #
    def load_case(self, case: Case) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
        image = np.array(Image.open(case.image_path))
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        existing = None
        warning = None
        if case.existing_mask_path is not None:
            if case.existing_mask_path.is_file():
                raw = np.array(Image.open(case.existing_mask_path).convert("L"))
                m = raw.astype(np.float32) / 255.0 > 0.5
                if m.shape == image.shape[:2]:
                    existing = m
                else:
                    warning = (f"Existing mask size {m.shape[1]}x{m.shape[0]} does not match "
                               f"image size {image.shape[1]}x{image.shape[0]}; starting with an empty mask.")
            else:
                warning = (f"Mask_path is present in the CSV, but the mask file was not found: "
                           f"{case.existing_mask_path}. Starting a new segmentation; saving will use this path.")
        return image, existing, warning
    #
    def save_case(self, case: Case, mask: np.ndarray, segmented: np.ndarray) -> None:
        if mask.shape != segmented.shape[:2]:
            raise ValueError("Mask size does not match current image.")
        #
        case.mask_save_path.parent.mkdir(parents=True, exist_ok=True)
        case.segmented_path.parent.mkdir(parents=True, exist_ok=True)
        #
        Image.fromarray((np.asarray(mask, bool).astype(np.uint8) * 255), mode="L").save(case.mask_save_path)
        Image.fromarray(_image_to_writable(segmented)).save(case.segmented_path)
        # Save absolute mask path into the CSV.
        self.df.at[case.row_index, "Mask_path"] = str(case.mask_save_path.expanduser().resolve())
        self.df.at[case.row_index, "Segment"] = 1
        self.df.to_csv(self.csv_path, index=False)
    #
#
def _image_to_writable(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating):
        b = a.astype(np.float32)
        mn, mx = float(np.nanmin(b)), float(np.nanmax(b))
        if mn >= 0 and mx <= 1:
            b = b * 255.0
        elif mx > mn:
            b = (b - mn) / (mx - mn) * 255.0
        return np.clip(b, 0, 255).astype(np.uint8)
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        return ((a.astype(np.float32) - info.min) / (info.max - info.min) * 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)
#