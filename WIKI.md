cd ~/ws
rm -rf build/* install/* log

# 1. f110_gym alone, without --symlink-install
colcon build --packages-up-to f110_gym

# 2. everything else, through the repo's script
#    (it adds --symlink-install and --continue-on-error)
source /opt/ros/humble/setup.bash
~/ws/src/race_stack/.vscode/build.sh Release

source install/setup.bash



# Clean up leftover nodes
pkill -9 -f <name>


# Launch two car sim
ros2 launch stack_master sim_two_car_launch.xml map_name:=GLC_smile_small ego_planner:=spliner opp_planner:=spliner


















