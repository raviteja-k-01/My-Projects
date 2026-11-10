from __future__ import annotations

from dataclasses import dataclass

from visiontrack.geometry.polygon import BBox, Polygon, bbox_center, point_in_polygon


@dataclass(frozen=True)
class ZoneDefinition:
    key: str
    zone_code: str
    display_name: str
    db_zone_name: str
    polygon: Polygon
    enabled: bool = True


class ZoneMap:
    def __init__(self, zones_config: dict):
        self.zones = self._load_zones(zones_config)

    def zone_for_bbox(self, bbox: BBox) -> ZoneDefinition | None:
        center = bbox_center(bbox)
        for zone in self.zones:
            if zone.enabled and point_in_polygon(center, zone.polygon):
                return zone
        return None

    def zone_code_for_bbox(self, bbox: BBox) -> str | None:
        zone = self.zone_for_bbox(bbox)
        return zone.zone_code if zone else None

    @staticmethod
    def _load_zones(config: dict) -> list[ZoneDefinition]:
        zones = []
        for key, data in config.get("zones", {}).items():
            polygon = [tuple(point) for point in data.get("polygon", [])]
            zones.append(
                ZoneDefinition(
                    key=key,
                    zone_code=str(data["zone_code"]),
                    display_name=str(data.get("display_name", key)),
                    db_zone_name=str(data.get("db_zone_name", data.get("display_name", key))),
                    polygon=polygon,
                    enabled=bool(data.get("enabled", True)),
                )
            )
        return zones
