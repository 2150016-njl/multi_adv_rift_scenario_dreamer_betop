import itertools
from typing import Dict, List

import cv2
import numpy as np
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from nuplan_plugin.actor_state.state_representation import Point2D
from nuplan_plugin.actor_state.static_object import StaticObject
from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
from scipy import ndimage
from shapely import Point, Polygon

DA = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]


class CostMapManager:
    def __init__(
        self,
        origin: np.ndarray,
        angle: float,
        map_api: CarlaMap,
        height: int = 500,
        width: int = 500,
        resolution: float = 0.2,
    ) -> None:
        self.map_api = map_api
        self.height = height
        self.width = width
        self.resolution = resolution
        self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
        self.origin = origin
        self.angle = angle
        self.offset = np.array([height / 2, width / 2], dtype=np.float32)
        self.rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )

    def build_cost_maps(
        self,
        static_objects: List[StaticObject],
        agents: Dict[str, np.ndarray] = None,
        agents_polygon: List[Polygon] = None,
    ):
        drivable_area_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        speed_limit_mask = np.zeros((self.height, self.width), dtype=np.float32)

        radius = max(self.height, self.width) * self.resolution / 2
        objects_dict = self.map_api.query_proximal_map_data(
            Point(*self.origin), radius
        )
        da_objects_dict = {da: objects_dict[da] for da in DA}

        da_objects = itertools.chain.from_iterable(da_objects_dict.values())

        for obj in da_objects:
            self.fill_polygon(drivable_area_mask, obj.polygon, value=1)

            speed_limit_mps = obj.speed_limit_mps if obj.speed_limit_mps else 50
            self.fill_polygon(speed_limit_mask, obj.polygon, value=speed_limit_mps)

        for static_ojb in static_objects:
            if np.linalg.norm(static_ojb.center.array - self.origin, axis=-1) > radius:
                continue
            self.fill_convex_polygon(
                drivable_area_mask, static_ojb.box.geometry, value=0
            )

        if agents is not None:
            # parking vehicles as static obstacles
            position = agents["position"]
            valid_mask = agents["valid_mask"]
            for pos, mask, polygon in zip(position, valid_mask, agents_polygon):
                if mask.sum() < 50:
                    continue
                pos = pos[mask]
                displacement = np.linalg.norm(pos[-1] - pos[0])
                if displacement < 1.0:
                    self.fill_convex_polygon(drivable_area_mask, polygon, value=0)

        distance = ndimage.distance_transform_edt(drivable_area_mask)
        inv_distance = ndimage.distance_transform_edt(1 - drivable_area_mask)
        drivable_area_sdf = distance - inv_distance
        drivable_area_sdf *= self.resolution

        return {
            "cost_maps": drivable_area_sdf[:, :, None].astype(np.float16),  # (H, W. C)
        }

    def global_to_pixel(self, coord: np.ndarray):
        coord = np.matmul(coord - self.origin, self.rot_mat)
        coord = coord / self.resolution_hw + self.offset
        return coord

    def fill_polygon(self, mask, polygon, value=1):
        polygon = self.global_to_pixel(np.stack(polygon.exterior.coords.xy, axis=1))
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], value)

    def fill_convex_polygon(self, mask, polygon, value=1):
        polygon = self.global_to_pixel(np.stack(polygon.exterior.coords.xy, axis=1))
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), value)

    def fill_polyline(self, mask, polyline, value=1):
        polyline = self.global_to_pixel(polyline)
        cv2.polylines(
            mask,
            [np.round(polyline.reshape(-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=value,
            thickness=1,
        )
