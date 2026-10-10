# Carla Map Doc

## Get HD Map Data

* Regenerate HD Map Data

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Generate HD Map Data for Specific Carla Town
python data/gen_hdmap.py -town Town05 # Generate Town05 HD Map Data
```

* or download from [Google Drive](https://drive.google.com/drive/folders/1CWdcO2Gd-Qd9cF-bBTC1oK6jVZtFGDj3?usp=drive_link) and put into [here](../data/map_data)

## Carla Map Annotation

[HD Map Anno](../data/map_data/anno/HD-Map-Anno.md)

## Carla Map API (Required HD Map Data)

[Carla Map API](../rift/cbv/planning/pluto/utils/nuplan_map_utils.py)

**Core function**

* query_proximal_map_data (Lane, Lane Connector, Crosswalks)

```python
query_proximal_map_data(point: Point, radius: float) -> Dict[SemanticMapLayer, List[CarlaMapObject]]:
    """
    Query map elements within a specified radius of a given geographic coordinate,
    and return them as a dictionary mapping SemanticMapLayer to lists of CarlaMapObjects.

    Parameters:
    - x, y: Geographic coordinates.
    - radius: Query radius, in the same unit as the coordinates.

    Returns:
    - proximal_map_data: Dictionary mapping SemanticMapLayer to lists of CarlaMapObjects.
    """
```

* query_reference_lines

```python
query_reference_lines(current_waypoints, current_state: CarlaAgentState, route_ids: Dict[str, List[int]], max_length: float=120):
    """
    Query all the possible reference lanes under given max_length.

    Parameters:
    - current_waypoints: the current carla waypoint list.
    - current_state: the current state of CBV
    - route_road_ids: the road id set of the CBV global route
    - max_length: the max length of the reference lane.

    Returns:
    - reference_lanes: List of LineStrings representing the reference lanes.
    """
```



