import pyspacemouse
import time
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    states = []
    times = []  # store timestamps
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    success = pyspacemouse.open(device="3Dconnexion Universal Receiver")
    if success:
        print("SpaceMouse connected")
    else:
        print("SpaceMouse connection failed")
        exit()

    start_time = time.time()
    for i in range(1000000):
        state = pyspacemouse.read()
        states.append([
            state.x, state.y, state.z,          # translation axes
            state.roll, state.pitch, state.yaw, # rotation axes
            state.buttons[0], state.buttons[1]  # button states
        ])
        times.append(time.time() - start_time)  # time since start (s)
        print(i)

    pyspacemouse.close()

    states = np.array(states)
    times = np.array(times)
    labels = ["x", "y", "z", "roll", "pitch", "yaw", "button1", "button2"]

    plt.figure(figsize=(10, 6))
    for i in range(states.shape[1]):
        plt.plot(times, states[:, i], label=labels[i])

    plt.xlabel("Time (s)")  # use seconds
    plt.ylabel("Value")
    plt.title("SpaceMouse State Over Time (pitch)")
    plt.legend()
    plt.grid(True)

    # Cleaner X-axis ticks every 1 second
    # import matplotlib.ticker as ticker
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))

    save_path = os.path.join(save_dir, "spacemouse_state_time_plot_pitch.png")
    plt.savefig(save_path)
    # plt.show()

    print(f"Plot saved at: {save_path}")
