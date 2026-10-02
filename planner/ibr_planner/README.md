# IBR Planner

Iterative Best Response (IBR) planner, ported from the [AirSim NeurIPS 2019 Drone Racing](https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing) baseline (`baselines/gtp.py`).
In the two-car simulator, the IBR car (the sim's 2nd car, `opp_racecar`) races the spliner car (the sim's 1st car, `car_state`).

## Install

```bash
pip install --user "cvxpy>=1.4,<1.6"   # cvxpy 1.7+ pulls numpy 2 and breaks ROS Humble
cd ~/ws
colcon build --packages-select ibr_planner stack_master
source install/setup.bash
```

## Configure the second car

In `stack_master/config/SIM/sim.yaml`:

```yaml
num_agent: 2
sx1: <x>        # start pose of the IBR car, must be on the track
sy1: <y>
stheta1: <yaw>
```

Rebuild after every change to `sim.yaml`:

```bash
colcon build --packages-select stack_master && source install/setup.bash
```

## Run (IBR vs spliner)

Run each command in its own terminal, after `source ~/ws/install/setup.bash`:

```bash
ros2 launch stack_master base_system_launch.xml sim:=True racecar_version:=SIM map_name:=<map>
ros2 launch stack_master head_to_head_launch.xml ctrl_algo:=PP      # spliner car
ros2 launch ibr_planner ibr_planner_launch.xml                       # IBR car
```

Nothing moves until all three are running.

Optional: drop static obstacles the spliner car's tracker remembers after the IBR car has left:

```bash
ros2 param set /tracking noMemoryMode true
```

## Run (spliner vs virtual opponent, no IBR)

Set `num_agent: 1` in `sim.yaml` and rebuild `stack_master`, then:

```bash
ros2 launch stack_master base_system_launch.xml sim:=True racecar_version:=SIM map_name:=<map>
ros2 launch opponent_publisher opponent_publisher_launch.xml type:=virtual speed_scaler:=0.5 start_s:=5.0
ros2 launch stack_master head_to_head_launch.xml ctrl_algo:=PP perception:=False
```

## Monitor

```bash
ros2 topic echo /ibr/solve_time                                 # should stay < 1/plan_rate_hz
ros2 topic hz /opp_drive                                        # IBR control rate, should be ~40 Hz
ros2 topic echo /opp_racecar/odom --field twist.twist.linear.x  # IBR car speed
ros2 topic echo /car_state/odom --field twist.twist.linear.x    # spliner car speed
```

In RViz, add `/ibr/path` to see the IBR plan.

## Debug: IBR car does not move

```bash
grep num_agent ~/ws/install/stack_master/share/stack_master/config/SIM/sim.yaml   # must be 2
ros2 topic hz /opp_racecar/odom                                                   # must be publishing
ros2 topic echo /ibr/solve_time --once                                            # must be < plan_timeout_s
```

## Parameters

All parameters, with comments, are in `config/ibr_planner_params.yaml`. Rebuild `ibr_planner` after editing it.
