from __future__ import annotations

from visiontrack.geometry.polygon import bbox_area, bbox_center, intersection_area, point_in_polygon
from visiontrack.perception.raw_observation import RawObservation


class PalletMapper:
    def __init__(self, min_overlap_ratio: float = 0.10):
        self.min_overlap_ratio = min_overlap_ratio

    def find_parent_pallet(
        self,
        box: RawObservation,
        pallets: list[RawObservation],
    ) -> RawObservation | None:
        best_pallet = None
        best_score = 0.0
        box_center = bbox_center(box.bbox)

        for pallet in pallets:
            px1, py1, px2, py2 = pallet.bbox
            pallet_polygon = [(px1, py1), (px2, py1), (px2, py2), (px1, py2)]
            center_inside = point_in_polygon(box_center, pallet_polygon)
            overlap = intersection_area(box.bbox, pallet.bbox)
            ratio = overlap / max(float(bbox_area(box.bbox)), 1.0)
            score = ratio + (0.5 if center_inside else 0.0)
            if score > best_score:
                best_score = score
                best_pallet = pallet

        if best_score >= self.min_overlap_ratio:
            return best_pallet
        return None
