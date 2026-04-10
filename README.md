# Installation

(TODO: Make sure to install CUDA and toolkits first)
This repo was run for:
CUDA 13.0
PyTorch for 13.0

(Make sure nvidia toolkit and CUDA are properly setup first)

This repo uses Poetry for dependency management. To set up this project, first install
[Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10
installed on your system.

Then, configure poetry to set up a virtual environment that uses Python 3.10:
```
poetry env use python3.10
```

Next, install all the required dependencies to the virtual environment with the
following command:
```bash
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

Ensure you have built colmap (cloned via submodule) with CUDA 13 and specify the correct architecture: [Colmap Installation](https://colmap.github.io/install.html)

Then locally install pycolmap after building Colmap's C++ code: [PyColmap Installation](https://colmap.github.io/pycolmap/index.html#pycolmap-index)

We must install a couple files manually as well (this is for gsplat):
```
poetry run pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim.git@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
```

For local Drake and manipulation installations, insert the following at the end of the
`.venv/bin/activate` and `.venv/bin/activate.nu` files, modifying the paths and python
version as required:
```bash
export PYTHONPATH=""
export PYTHONPATH=~/drake-build/install/lib/python3.10/site-packages:${PYTHONPATH}
export PYTHONPATH=~/manipulation:${PYTHONPATH}
```

Activate the environment:
```bash
poetry shell
```

Install `git-lfs`:

```bash
git-lfs install
git-lfs pull
```

## Other Dependencies
You may also need to install the following:
```
sudo apt update
sudo apt install python3-tk
```

### iiwa Driver

[Drake's iiwa driver](https://github.com/RobotLocomotion/drake-iiwa-driver) must be
installed manually to use the real iiwa robot. NOTE that
[Drake's pre-requisites](https://drake.mit.edu/from_source.html) must be installed
before installing the driver.

The FRI source can be downloaded from
[here](https://mitprod-my.sharepoint.com/:u:/g/personal/nepfaff_mit_edu/EdUdfStUexZKqlfwKLTKOyUBmpoI3H1ylzit-813TMV1Eg?e=HRWaIv)
and installed using the following instructions (from the driver repo):
```bash
cd kuka-fri
unzip /path/to/your/copy/of/FRI-Client-SDK_Cpp-1_7.zip
patch -p1 < ../fri_udp_connection_file_descriptor.diff
```

Once build, the driver can be run using `./bazel-bin/kuka-driver/kuka_driver` or using
`bazel run //kuka-driver:kuka_driver`.

#### Networking troubleshooting

If the driver doesn't connect to the kuka, check that the sunrise cabinet is reachable
on the network using `nmap -sP 192.170.10.2/24`. Both the local computer and a second
computer (the sunrise cabinet) should show up.

If it doesn't show up, check the following:
1. There must be an ethernet network connecting the local computer and the sunrise
sunrise cabinet KONI port (ideally through a switch). This network must have the static
IP `192.170.10.200` with netmask `255.255.255.0`.
2. The sunrise cabinet KONI port must be owned by RTOS and not by Windows. Connect a
monitor, mouse, and keyboard to the sunrise cabinet. Start the cabinet and login. Press
`WIN+R` to open the command window. Type
`C:\KUKA\Hardware\Manager\KUKAHardwareManager.exe -query OptionNIC -os RTOS`. Everything
is in order if the popup says `BusType OptionNIC found`. If the popup says
`BusTypeOptionNIC not present`, change the port ownership using
`C:\KUKA\Hardware\Manager\KUKAHardwareManager.exe -assign OptionNIC -os RTOS`. Unplug
the monitor and restart the sunrise cabinet before re-checking the network with `nmap`.

#### Port troubleshooting

Make sure that the sunrise cabinet port matches the kuka driver port. If not, then
modify the kuka driver source code to change the port
(`kuka-driver/kuka_driver.cc/kDefaultPort`). It is also possible to start the kuka
driver with a specific port over the command line. However, it is easier to hardcode the
port as the cabinet port won't change.

#### Robot limit exceeded errors

1. Enter the KRF mode
2. Manually operate the robot out of the limits using the tablet
3. Re-enter automatic mode

If no KRF mode exists, then do the following:
1. Unmaster the joint whose limits are exceeded
2. Use the teach pendant in T1 mode to move the joint back inside its allowed range
3. Master the joint

#### "Voltage of intermediate circuit too low" error

One of the fuses is blown. You need to open the control box and replace them.
In most cases, it is sufficient to replace the 5A fuse.

The two fuses are:
- 5A/80V Automotive Fuse ([buying link](https://www.newark.com/multicomp-pro/mp008147/fuse-automotive-5a-80vdc-rohs/dp/69AJ0523))
- 7.5A/80V Automotive Fuse ([buying link](https://www.newark.com/multicomp-pro/mp008148/fuse-automotive-7-5a-80vdc-rohs/dp/69AJ0524))

### Schunk WSG 50 Gripper Driver (Optional)

Connect the WSG gripper to the same switch that is connecting the local computer with
the sunrise cabinet. Add the IP `192.168.1.200` with netmask `255.255.255.0` as an
additional static IP to the network (the first IP should still be `192.170.10.200`).
The WSG is connected properly if the `WSG 50 Control Panel` web interface can be
accessed through http://192.168.1.20/. Try to control the gripper through the web
interface. If this doesn't work, then controlling it through the driver also won't work.

[Drake's Schunk driver](https://github.com/RobotLocomotion/drake-schunk-driver) must be
installed manually to use the WSG programatically. Once built, the driver can be run
using `bazel run //src:schunk_driver`. The driver requires Bazel 6. Multiple Bazel
versions can be managed by installing `bazelisk` from [here](https://github.com/bazelbuild/bazelisk/releases).
The Bazel version will then be read from the `.bazeliskrc` file in the repo.

#### Cable

The WSG requires a Male M8 4-Pin A Coding to RJ45 Connector.

#### Networking troubeshooting

Check that one host shows up when using `nmap -sP 192.168.1.201`. and that the website
is accessible at http://192.168.1.20/. If not, then check that you followed the IP
instructions and that the gripper's ethernet cable is plugged into the switch.

#### Error while moving: The device is not initialized

1. Navigate to the website http://192.168.1.20/
2. Motion -> Manual Control
3. Click on `Home` and wait until the homing sequence is finished
4. Re-try commanding the gripper via the web interface

#### Network error during movement

This might be due to cable issues. Check for cable issues by opening the webpage and
locating the "Link Active" blinking/ switching light indicator. The blue light
continuously switches between the left and right circle while the gripper is connected.
Pull/ twist one of the cables and see whether the light stops switching. If this is the
case, then there is probably a cable error and the cable might need replacing.

Note that the gripper takes a few minutes to reconnect after the connection ist lost.
The connection is re-established once the webpage loads again.

#### Getting system info failed

Follow the [wsg driver setup instructions](https://github.com/RobotLocomotion/drake-schunk-driver?tab=readme-ov-file#configuring-the-gripper).
In particular, the gripper might be set to ICP instead of UDP.

#### Can't connect to the gripper/ nothing else works

Reset the gripper's config on its MicroSD card:
1. Power off the gripper
2. Remove the MicroSD card (behind the black plate with two small screws on the gipper side)
3. Insert it into your computer
4. Navigate to the config folder
5. Rename the file `config/system.cfg` to `config/system.old`
6. Re-insert the MicroSD card into the gripper
7. The gripper should now be discoverable on its default IP address

### Optitrack Driver (Optional)

[Drake's Optitrack driver](https://github.com/RobotLocomotion/optitrack-driver) must be
installed manually to use the [Optitrack](https://optitrack.com/) functionality.

Build and install the wheel as described
[here](https://github.com/RobotLocomotion/optitrack-driver#to-build-a-wheel). Make sure
to install the wheel from inside the poetry virtual environment.

### FT 300-S Driver (Optional)

The [FT 300-S LCM driver](https://github.com/nepfaff/ft-300s-driver) must be installed
according to its README instructions.

The included LCM messages must be added to the python path (after building):
```
export PYTHONPATH=~/path_to_parent_dir/ft_300s_driver/bazel-bin/lcmtypes/ft_300s/:${PYTHONPATH}
```

## Executing code on the real robot

1. Start the `DrakeFRIPositionDriver` or `DrakeDRITorqueDriver` on the teach pendant.
2. Run the iiwa driver by running `bazel run //kuka-driver:kuka_driver` from
`drake-iiwa-driver`.
3. If using the WSG, run the schunk driver using `bazel run //src:schunk_driver` from
`drake-schunk-driver`.
4. Run the desired script with the `--use_hardware` flag.

### Note about timesteps
The `torque_only` driver runs at 1000Hz while all other controllers run at 200Hz. The
specified timesteps should match the controllers, i.e. 0.001 for the `torque_only`
driver and 0.005 for all other drivers.

### Controlling the robot in `torque_only` mode

**NOTE:** It is recommended to calibrate the joint torque sensors before running the
robot in `torque_only` mode. This can be achieved by running the
`PositionAndGMSReferencing` application on the teach pendant.

1. Start the `DrakeFRITorqueOnlyDriver` on the teach pendant.
2. Optional: Make sure that the iiwa driver is build by running `bazel build //...` from
`drake-iiwa-driver`.
3. Run the iiwa driver by running
`sudo ./bazel-bin/kuka-driver/kuka_driver --torque_only=true --time_step 0.001 --realtime`
from `drake-iiwa-driver` (`sudo` is required for `--realtime` which helps but is not
required).
4. Run the desired script with the `--use_hardware` flag.

Note that you might need to increase the conservative default torque limits in the driver
code. See [here](https://github.com/nepfaff/drake-iiwa-driver/tree/increase_default_torque_limits)
for how to do this. This is necessary if the motions appear very jerky and the driver
terminates with "Robot is in an unsafe state".

#### Obtaining slightly better performance

You might be able to achieve slightly better performance in `torque_only` mode by
pinning the processes to the same core and increasing their priority.

1. Make sure that you can run `chrt` without sudo privileges:
`sudo setcap cap_sys_nice=eip /usr/bin/chrt`. This is only required once.
2. Run the desired script using `taskset -c 1,29 chrt -r 90 python {...} --use_hardware`.

# Usage

To run an example to test that everything works, you can download their sample dataset. First, navigate to the `reconstruction` folder.

```
python datasets/download_dataset.py
```

Then, run `simple_trainer.py` with one of the datsets:
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
```