from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from .mask_utils import polygon_area, polygon_centroid, polygon_to_mask, points_to_json


ALLOWED_TAGS = {"bean", "coin", "color-card", "ruler"}


@dataclass
class AnnotationRow:
    image_filename: str
    image_stem: str
    image_width: int
    image_height: int
    object_index: int
    tag: str
    center_x: float
    center_y: float
    manual_area_px: float
    polygon_points_json: str


class OpenCVPolygonAnnotator:
    def __init__(
        self,
        image_paths: List[Path],
        output_xlsx: str | Path,
        annotated_dir: str | Path | None = None,
    ):
        self.annotated_dir = Path(annotated_dir) if annotated_dir else None
        if self.annotated_dir:
            self.annotated_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = image_paths
        self.output_xlsx = Path(output_xlsx)
        self.rows: List[AnnotationRow] = []

        # Load existing annotations if present
        if self.output_xlsx.exists():
            existing_df = pd.read_excel(self.output_xlsx)
            for _, r in existing_df.iterrows():
                self.rows.append(
                    AnnotationRow(
                        image_filename=r["image_filename"],
                        image_stem=r["image_stem"],
                        image_width=r["image_width"],
                        image_height=r["image_height"],
                        object_index=r["object_index"],
                        tag=r["tag"],
                        center_x=r["center_x"],
                        center_y=r["center_y"],
                        manual_area_px=r["manual_area_px"],
                        polygon_points_json=r["polygon_points_json"],
                    )
                )
            print(f"Loaded {len(self.rows)} existing annotations from {self.output_xlsx}")

        self._current_points: List[Tuple[int, int]] = []
        self._completed_polygons: List[dict] = []
        self._img = None
        self._display = None
        self._display_scale = 1.0
        self._display_offset = (0, 0)
        self._current_index = self._find_resume_index()
        self._window_name = "FabaBean Annotator"

    def _find_resume_index(self) -> int:
        return 0

    # def _find_resume_index(self) -> int:
    #     completed_images = {row.image_filename for row in self.rows}
    #     for idx, img_path in enumerate(self.image_paths):
    #         if img_path.name not in completed_images:
    #             return idx
    #     return len(self.image_paths)

    def _save_annotated_image(self, image_path: Path):
        """Save current image with completed polygons drawn."""
        if self.annotated_dir is None or self._img is None:
            return

        canvas = self._img.copy()

        for poly_idx, item in enumerate(self._completed_polygons):
            pts = np.array(item["points"], dtype=np.int32)

            # polygon
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # label
            label = f'{item["tag"]} #{poly_idx + 1}'
            x0, y0 = pts[0]
            cv2.putText(
                canvas,
                label,
                (x0 + 10, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        output_path = self.annotated_dir / image_path.name
        cv2.imwrite(str(output_path), canvas)

    def _load_image(self, image_path: Path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        self._img = img
        h, w = img.shape[:2]

        max_w, max_h = 1600, 1100
        scale = min(max_w / w, max_h / h, 1.0)
        self._display_scale = scale
        if scale < 1.0:
            display = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            display = img.copy()
        self._display = display
        self._display_offset = (0, 0)
        self._current_points = []
        self._completed_polygons = []

    def _to_image_coords(self, x: int, y: int) -> Tuple[int, int]:
        scale = self._display_scale if self._display_scale > 0 else 1.0
        ix = int(round((x - self._display_offset[0]) / scale))
        iy = int(round((y - self._display_offset[1]) / scale))
        h, w = self._img.shape[:2]
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        return ix, iy

    def _draw(self) -> np.ndarray:
        canvas = self._display.copy()

        for poly_idx, item in enumerate(self._completed_polygons):
            pts = np.array(
                [(int(round(x * self._display_scale)), int(round(y * self._display_scale))) for x, y in item["points"]],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
            label = f'{item["tag"]} #{poly_idx + 1}'
            x0, y0 = pts[0]
            cv2.putText(canvas, label, (x0 + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(self._current_points) > 0:
            pts = np.array(self._current_points, dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(0, 0, 255), thickness=1)
            for p in self._current_points:
                cv2.circle(canvas, p, 1, (0, 0, 255), -1)

        info_lines = [
            f"Image {self._current_index + 1}/{len(self.image_paths)}: {self.image_paths[self._current_index].name}",
            "Left click add point | f finish polygon | c clear | n next | p prev | q quit",
            f"Completed objects on image: {len(self._completed_polygons)}",
        ]
        y = 25
        for line in info_lines:
            cv2.putText(canvas, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 28
        return canvas

    def _save_rows(self, image_path: Path):
        if not self.rows:
            return
        df = pd.DataFrame([row.__dict__ for row in self.rows])
        self.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(self.output_xlsx, index=False)

    def _finalize_polygon(self, image_path: Path):
        if len(self._current_points) < 3:
            print("Need at least 3 points to form a polygon.")
            return

        while True:
            tag = input("Tag this polygon [bean / coin / color-card / ruler]: ").strip().lower()
            if tag in ALLOWED_TAGS:
                break
            print("Invalid tag. Try again.")

        # tag = "bean"  # default to bean for now since we only have that class

        img = self._img
        h, w = img.shape[:2]
        area = polygon_area(self._current_points)
        cx, cy = polygon_centroid(self._current_points)

        obj_idx = sum(1 for row in self.rows if row.image_filename == image_path.name) + 1

        row = AnnotationRow(
            image_filename=image_path.name,
            image_stem=image_path.stem,
            image_width=w,
            image_height=h,
            object_index=obj_idx,
            tag=tag,
            center_x=float(cx),
            center_y=float(cy),
            manual_area_px=float(area),
            polygon_points_json=points_to_json(self._current_points),
        )
        self.rows.append(row)
        self._completed_polygons.append({"points": self._current_points.copy(), "tag": tag})
        self._current_points = []
        self._save_rows(image_path)
        print(f"Saved annotation #{obj_idx} for {image_path.name} ({tag}).")
        self._save_annotated_image(image_path)

    def run(self):
        if not self.image_paths:
            raise ValueError("No images were provided.")

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = self._to_image_coords(x, y)
                dx = int(round(ix * self._display_scale))
                dy = int(round(iy * self._display_scale))
                self._current_points.append((dx, dy))

        cv2.setMouseCallback(self._window_name, mouse_callback)

        while 0 <= self._current_index < len(self.image_paths):
            
            image_path = self.image_paths[self._current_index]
            self._load_image(image_path)

            # restore already completed rows for this image if going back
            for row in self.rows:
                if row.image_filename == image_path.name:
                    pts = json.loads(row.polygon_points_json)
                    original = [(int(p["x"]), int(p["y"])) for p in pts]
                    self._completed_polygons.append({"points": original, "tag": row.tag})

            while True:
                frame = self._draw()
                cv2.imshow(self._window_name, frame)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    self._save_rows(image_path)
                    self._save_annotated_image(image_path)
                    cv2.destroyAllWindows()
                    return

                if key == ord('c'):
                    self._current_points = []

                elif key == ord('f'):
                    # convert display points to original image coordinates for storage
                    original_pts = []
                    for px, py in self._current_points:
                        ix, iy = self._to_image_coords(px, py)
                        original_pts.append((ix, iy))
                    self._current_points = original_pts
                    self._finalize_polygon(image_path)

                elif key == ord('n'):
                    self._save_annotated_image(image_path)
                    self._current_points = []
                    self._current_index += 1
                    break

                elif key == ord('p'):
                    self._save_annotated_image(image_path)
                    self._current_points = []
                    self._current_index = max(0, self._current_index - 1)
                    break

            self._save_rows(image_path)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-csv",
        type=str,
        required=True,
        help="CSV file created by the sampling step"
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default="work/manual_annotations.xlsx",
        help="Excel file to save annotations"
    )
    args = parser.parse_args()

    selected_csv = Path(args.selected_csv)
    if not selected_csv.exists():
        raise FileNotFoundError(f"Selected CSV not found: {selected_csv}")

    df = pd.read_csv(selected_csv)

    # Adjust this depending on your CSV column name
    if "image_path" in df.columns:
        image_paths = [Path(p) for p in df["image_path"].tolist()]
    elif "filename" in df.columns:
        # if the CSV only stores filenames, you must rebuild full paths here
        raise ValueError("CSV has filename only; please store full image_path in selected_images.csv")
    else:
        raise ValueError(f"Unsupported CSV columns: {list(df.columns)}")

    annotator = OpenCVPolygonAnnotator(
        image_paths=image_paths,
        output_xlsx=args.output_xlsx
    )
    annotator.run()