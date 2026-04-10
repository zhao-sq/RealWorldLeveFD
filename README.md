### Guidelines
#### Communication
Use the ethernet to connect the robot to your computer. The IP address is 192.168.1.101. If you can ping this address, you can just turn to Python controller.
```bash
ping 192.168.1.101
```
The success output is like:
```bash
PING 192.168.1.101 (192.168.1.101) 56(84) bytes of data.
64 bytes from 192.168.1.101: icmp_seq=1 ttl=60 time=3.81 ms
...
```

or you can type `http://192.168.1.101` in your web browser and the "robot homepage" should appear.

If not, please set the IPV4 configuration. Mannual configuration. Address: 192.168.1.200. Netmask: 255.255.255.0 Gateway: 192.168.1.1

After restart or reconnecting the cable:
```bash
sudo ip addr add 192.168.1.10/24 dev enx6c1ff71d6359
```

#### Environment & dependencies

1. **Create Python environment (py 3.7)** — once per machine/project:
    ```bash
    conda create --name crx python=3.7
    conda activate crx
    ```
    Afterwards simply `conda activate crx` on future sessions.

2. **Install required packages:**
    ```bash
    cd crx_rmi_utils
    pip install -e .
    pip install pyqtgraph zmq pyspacemouse pyserial opencv-python pynput pyrealsense2
    ```

---

#### Python controller

3. **Install gripper driver (if using gripper):**

    * linux: https://github.com/WCHSoftGroup/ch343ser_linux  or https://www.wch.cn/downloads/CH341SER_LINUX_ZIP.html
    * mac: https://www.wch.cn/downloads/CH341SER_MAC_ZIP.html
    * windows: https://www.wch.cn/downloads/CH343SER_EXE.html

4. **Detect your serial port:**
    ```bash
    ls /dev/ttyUSB* /dev/ttyACM*
    ```

5. **Configure and run the controller:**

    Open `teleoperation/crx_spacemouse_gripper.py` and adjust:

    * `device_name` argument to `connect_to_spacemouse()` (default: "3Dconnexion Universal Receiver").
    * `ENABLE_GRIPPER` constant at the top to enable/disable the gripper.

    Then start the script:
    ```bash
    python teleoperation/crx_spacemouse_gripper.py
    ```

    This invocation lets you operate the robot manually. **No data will be recorded** when running the controller alone.
    To collect demos you must launch the collector script described below.

6. **Control manual:**

    * **Translation mode** – move the joystick to translate the gripper.
    * **Orientation mode** – hold the left button and move the joystick to rotate the tool.
    * **Gripper operation** – press the right button and move the joystick to send open/close commands.
    * **Exit** – hold both left and right buttons for ~3 s. (only used in teleoperation/crx_spacemouse_gripper.py)

#### Data Collection and Data Viewer

3. **Set up data directory and number of demos.**
    
    _You must invoke the collector script (`teleoperation/data_collection.py`) to record data – simply running the controller will not save anything._

    The script now collects a specified number of **new** demos and asks you to classify each one as success or failure. Existing demo folders are skipped automatically and after labeling the demo directory is moved into a `success/` or `failure/` subfolder with a `label.txt` file.

    To change where data are saved, edit the `folder` variable in `teleoperation/data_collection.py` (near the top of `__main__`):
    ```python
    folder = "collected_data/place_kettle_on_potholder/"
    ```

    Adjust `exp_name` to change the base directory name for each demo:
    ```python
    exp_name = "demo_"
    ```

    Set how many *new* success demos you want to collect in this run with `num_demos`:
    ```python
    num_demos = 10
    ```
    The collector will ignore existing `demo_<n>` folders and continue incrementing the index until the requested number has been recorded. After each demo finishes you will be prompted to press **s** (success) or **f** (failure); the folder will then be moved into the corresponding subdirectory and labeled.

4. **Robot Control Manual:**
    
    Control of the robot during data collection is handled via the spacemouse; the behaviour matches the description given earlier in the "Python controller" section.  While collecting demos the script will automatically start the next run when you press **q** (or you can terminate entirely with Ctrl‑C).

5. **Data collection Manual:**

    Use the following command to start collecting data with the spacemouse-based collector:
    ```
    python teleoperation/data_collection.py
    ```
    <!-- ```
    sudo <full_path_of_python> MSC_Robot/crx/crx_rmi_utils/tests/data_collection.py
    ```
    You can get the full path of python by the command:
    ```
    which python
    ``` -->
    Press `q` to end a demos and start the next demo.

6. **View Data:**

    A separate utility script can export and inspect the recordings produced by the
    collector.  It lives at `teleoperation/data_viewer.py` and understands the
    `.npz` format written by the recorder.

    - **Plot robot trajectories** – 3‑D scatter plots of any of the saved state
      vectors (`sent_state`, `real_robot_state`, `spacemouse_state`).  Points are
      coloured by gripper open/closed and saved as PNG files under
      `<save_dir>/plot/`.

    - **Dump camera frames** – all colour frames recorded during a trial are
      written as PNGs to `<save_dir>/image/demo_<n>/`.  Optionally animated GIFs
      can be produced by setting `make_gifs=True`.

    - **Interactive playback** – `view_images()` will show the colour streams in
      an OpenCV window, stepping through trials with Enter and quitting with `q`.

    Example usage (adjust paths to match your data):
    ```python
    from teleoperation.data_viewer import dataViewer

    viewer = dataViewer("collected_data/mid_pour_water")

    # plot the sent_state trajectory
    viewer.plot("view_dataset/mid_pour_water", state_key="sent_state")

    # export all colour frames and make GIFs
    viewer.download_image("view_dataset/mid_pour_water", make_gifs=True)

    # play back interactively
    viewer.view_images()
    ```

    Or simply run the script directly:
    ```bash
    python teleoperation/data_viewer.py
    ```
    adjusting the hard‑coded `data_dir`/`out_dir` variables at the bottom of the
    file before launching.

    **Outputs produced by the viewer:**

    * `plot/` – PNG trajectory plots, one per trial (e.g. `0.png`, `1.png`).
    * `image/demo_<n>/` – raw camera frames from each trial.
    * `gif/<camera>/` – animated GIFs of each stream if `make_gifs=True`.
    * On‑screen windows when using `view_images()` (no files written).

#### Trouble Shooting
If you get the error message, "HIDException: unable to open device" or "spamouse not connected", try configuring udev:
```
sudo mkdir -p /etc/udev/rules.d/
echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0666", TAG+="uaccess", TAG+="udev-acl"' | sudo tee /etc/udev/rules.d/92-viia.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```
Reference: https://bbs.archlinux.org/viewtopic.php?id=278341

If you get the error message, "[Errno 2] could not open port /dev/ttyACM0", try
```
sudo chmod a+rw /dev/ttyACM0 
```
Reference: https://askubuntu.com/questions/1219498/could-not-open-port-dev-ttyacm0-error-after-every-restart


If you get "No Intel Device connected" even if you can access your camera with ```realsense-viewer```, try rebuilding librealsense for RSUSB backend (first uninstall pyrealsense2):

1. Activate ```crx```.
    ```bash
    conda activate crx
    ```

2. Since you already have the source:
    ```bash
    cd ~/Documents/librealsense
    rm -rf build
    mkdir build && cd build

    cmake .. \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=/home/msc/miniconda3/envs/crx/bin/python \
    -DCMAKE_BUILD_TYPE=Release

    make -j$(nproc)
    sudo make install
    ```

3. Copy the Compiled Binding to ```crx```.

    Find the built .so:
    ```bash
    find ~/Documents/librealsense/build -name "pyrealsense2*.so"
    ```

    For example,
    ```bash
    /home/msc/Documents/librealsense/build/Release/pyrealsense2.cpython-310-x86_64-linux-gnu.so
    ```

    Copy it into the crx site-packages:
    ```bash
    cp /home/msc/Documents/librealsense/build/Release/pyrealsense2.cpython-310-x86_64-linux-gnu.so \
    /home/msc/miniconda3/envs/crx/lib/python3.10/site-packages/
    ```

4. Fix the GLIBCXX / CXXABI Issue:
    ```bash
    conda install -c conda-forge libstdcxx-ng
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' > $CONDA_PREFIX/etc/conda/activate.d/ld_preload.sh
    ```

    Then reactivate:
    ```bash
    conda deactivate
    conda activate crx
    ```