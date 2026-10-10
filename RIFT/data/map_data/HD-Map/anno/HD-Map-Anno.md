## HD-Map Data Structure (in right hand system)
``` shell
  # Each HD-Map file contains road information of a certain town (under right-hand system)
  - road_id  # CARLA road id
    # Each road_id corresponds to a dict, where each element formed as:
    -lane_id
        - 'LaneType'  # the type of the lane, can be 'Driving'
        - 'LaneWidth' # the width of the lane
        - 'LaneMark'  # dict {'Left': [LaneElement], 'Center': [LaneElement], 'Right': [LaneElement]}, each LaneElement is a dict formed as:
            - Points  # Location-rotation array formed as ((location_x (m), location_y (m), location_z (m)), (roll (rad), pitch (rad), yaw (rad)), waypoint.is_junction)
            - Type # String, can be 'Broken', 'Solid', 'SolidSolid', 'Other', 'NONE', 'Center'
            - Color # Color, can be 'Blue', 'White', 'Yellow'. (color of Type-'Center' is 'White') 
            - Topology # String array contains the 'road_id' and 'lane_id' of the current road adjacent to, formed as ((road_id, lane_id), ..)
            # If the Type == 'Center', there will be other three keys named 'TopologyType', 'Left' and 'Right'
            - TopologyType # The current lane's topology status, can be 'Junction', 'Normal', 'EnterNormal', 'EnterJunction', 'PassNormal', 'PassJunction', 'StartJunctionMultiChange', or 'StartNormalMultiChange'
            - Left # The road_id and lane_id of the left lane of the current lane，formed as (road_id, lane_id), None if the left lane does not exist
            - Right # The road_id and lane_id of the right lane of the current lane，formed as (road_id, lane_id), None if the right lane does not exist
    # If current 'road_id' contains trigger volumes, there will be a special dict with 'TriggerVolumes' as key:
    - 'Trigger_Volumes'
        # Each 'TriggerVolumes' corresponds to a list, where each element is a dict formed as:
        - 'Points' # Vertexs location array of current trigger volume
        - 'Type' # The parent actor type of current trigger volume, can be 'StopSign' or 'TrafficLight'
        - 'ParentActor_Location' # The parent actor's location of current trigger volume, formed as (location.x, location.y, location.z)
  - 'Crosswalks'
    # 'Crosswalks' is a list, where each element is a dict formed as:
    - 'Polygon'  # shapely.geometry.Polygon object, representing the polygon of the crosswalk
    - 'Location'  # the center location of the crosswalk
```