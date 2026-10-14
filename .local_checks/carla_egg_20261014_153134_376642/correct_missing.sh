set -eu
CARLA_PYTHONAPI="C:/Users/Lenovo/Desktop/pycharm_project/multi_adv_rift_scenario_dreamer_betop/.local_checks/carla_egg_20261014_153134_376642/PythonAPI/carla/missing"
CARLA_EGG="C:/Users/Lenovo/Desktop/pycharm_project/multi_adv_rift_scenario_dreamer_betop/.local_checks/carla_egg_20261014_153134_376642/PythonAPI/carla/dist/carla-0.9.15-py2.7-linux-x86_64.egg"
  CARLA_EGG="$CARLA_PYTHONAPI/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
  export CARLA_EGG
  [ -f "$CARLA_EGG" ] || { echo "Missing previously verified CARLA egg: $CARLA_EGG" >&2; exit 2; }
  echo "CARLA_EGG=$CARLA_EGG"
