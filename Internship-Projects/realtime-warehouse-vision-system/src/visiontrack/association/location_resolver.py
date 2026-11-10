from __future__ import annotations

from dataclasses import dataclass

from visiontrack.association.pallet_mapper import PalletMapper
from visiontrack.geometry.polygon import (
    bbox_area,
    bbox_center,
    intersection_area,
    point_in_polygon,
)
from visiontrack.perception.raw_observation import RawObservation


REGION_CLASSES = frozenset(
    {
        "conveyor_belt",
        "dock",
        "zone",
    }
)


@dataclass(frozen=True)
class ResolvedLocation:
    pallet_code: str | None
    region_code: str | None
    pallet: RawObservation | None
    region: RawObservation | None


class DynamicLocationResolver:
    """
    Resolves locations only from current YOLO detections.

    No fixed polygons, chart coordinates, or camera-specific positions
    are used.
    """

    def __init__(
        self,
        min_pallet_overlap_ratio: float,
        min_region_overlap_ratio: float,
    ):
        self.pallet_mapper = PalletMapper(
            min_overlap_ratio=min_pallet_overlap_ratio
        )
        self.min_region_overlap_ratio = float(
            min_region_overlap_ratio
        )

    def resolve_box(
        self,
        box: RawObservation,
        pallets: list[RawObservation],
        regions: list[RawObservation],
    ) -> ResolvedLocation:
        parent_pallet = self.pallet_mapper.find_parent_pallet(
            box,
            pallets,
        )

        # When a box is on a pallet, use the pallet bbox to determine
        # the operational region carrying that box.
        location_subject = parent_pallet or box
        region = self._find_best_region(
            location_subject,
            regions,
        )

        return ResolvedLocation(
            pallet_code=self._identity(
                parent_pallet,
                "PALLET",
            ),
            region_code=self._region_identity(region),
            pallet=parent_pallet,
            region=region,
        )

    def resolve_pallet(
        self,
        pallet: RawObservation,
        regions: list[RawObservation],
    ) -> ResolvedLocation:
        region = self._find_best_region(
            pallet,
            regions,
        )

        return ResolvedLocation(
            pallet_code=self._identity(
                pallet,
                "PALLET",
            ),
            region_code=self._region_identity(region),
            pallet=pallet,
            region=region,
        )

    def _find_best_region(
        self,
        subject: RawObservation,
        regions: list[RawObservation],
    ) -> RawObservation | None:
        subject_center = bbox_center(subject.bbox)
        subject_area = max(
            float(bbox_area(subject.bbox)),
            1.0,
        )

        best_region = None
        best_score = None

        for region in regions:
            if region.class_name not in REGION_CLASSES:
                continue

            x1, y1, x2, y2 = region.bbox
            region_polygon = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]

            center_inside = point_in_polygon(
                subject_center,
                region_polygon,
            )
            overlap_ratio = (
                intersection_area(
                    subject.bbox,
                    region.bbox,
                )
                / subject_area
            )

            if (
                not center_inside
                and overlap_ratio
                < self.min_region_overlap_ratio
            ):
                continue

            # Prefer centre containment, greater overlap, and then the
            # smaller/more-specific detected region.
            score = (
                int(center_inside),
                overlap_ratio,
                -float(bbox_area(region.bbox)),
                float(region.confidence),
            )

            if best_score is None or score > best_score:
                best_score = score
                best_region = region

        return best_region

    @staticmethod
    def _identity(
        observation: RawObservation | None,
        prefix: str,
    ) -> str | None:
        if observation is None:
            return None
        if observation.qr_value:
            return observation.qr_value
        if observation.track_id is not None:
            return f"{prefix}_TRACK_{observation.track_id}"
        return observation.identity_hint.upper()

    @staticmethod
    def _region_identity(
        observation: RawObservation | None,
    ) -> str | None:
        if observation is None:
            return None
        if observation.qr_value:
            return observation.qr_value

        prefix = observation.class_name.upper()
        if observation.track_id is not None:
            return f"{prefix}_TRACK_{observation.track_id}"
        return observation.identity_hint.upper()
