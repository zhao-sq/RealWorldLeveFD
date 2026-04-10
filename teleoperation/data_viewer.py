import numpy as np
import cv2
import os
import keyboard
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

class dataViewer:
    def __init__(self, directory):
        self.directory = directory
        self.trials = None

    def countTrials(self):
        self.trials = len(self.get_trial_names())

    def get_trial_names(self):
        trial_names = []
        for name in os.listdir(self.directory):
            trial_path = os.path.join(self.directory, name)
            if not os.path.isdir(trial_path) or not name.startswith("demo_"):
                continue
            try:
                trial_idx = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            trial_names.append((trial_idx, name))
        trial_names.sort()
        return [name for _, name in trial_names]

    def get_data_files(self, trial_path):
        data_files = []
        for name in os.listdir(trial_path):
            full_path = os.path.join(trial_path, name)
            if not os.path.isfile(full_path) or not name.endswith(".npz"):
                continue
            stem = os.path.splitext(name)[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                frame_idx = float("inf")
            data_files.append((frame_idx, name))
        data_files.sort(key=lambda item: (item[0], item[1]))
        return [name for _, name in data_files]

    def plot(self, save_dir, state_key='sent_state'):
        """Plot a 3‑D scatter of a chosen robot state (``sent_state`` by default).

        * ``save_dir`` – parent directory where ``plot/`` will be created.
        * ``state_key`` – one of the keys saved by ``data_collection``
          (``sent_state`` | ``real_robot_state`` | ``spacemouse_state``).
        """

        self.countTrials()
        save_dir = os.path.join(save_dir, "plot")
        os.makedirs(save_dir, exist_ok=True)

        for trial_name in self.get_trial_names():
            trial_path = os.path.join(self.directory, trial_name)

            data_files = self.get_data_files(trial_path)

            positions = []
            colors = []

            for name in data_files:
                data = np.load(os.path.join(trial_path, name))
                if state_key not in data:
                    raise KeyError(f"state '{state_key}' not found in {name}")
                positions.append(data[state_key])
                # color by gripper open/closed, fall back to 0 if missing
                gr = int(data.get('gripper_state', 0))
                colors.append(1 if gr == 0 else 2)

            positions = np.array(positions)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Robot position ({state_key}) trial {trial_name}')
            plt.savefig(os.path.join(save_dir, f"{trial_name}.png"))
            plt.close(fig)

    def view_images(self):
        """Interactively display the colour streams for every trial.

        Arrow through trials with ENTER, quit with 'q'.
        """

        self.countTrials()
        for trial_name in self.get_trial_names():
            trial_path = os.path.join(self.directory, trial_name)
            data_files = self.get_data_files(trial_path)

            cams = ['fixed_left', 'fixed_right', 'in_hand']
            frames = {cam: [] for cam in cams}

            for name in data_files:
                data = np.load(os.path.join(trial_path, name))
                for cam in cams:
                    key = f"{cam}_color"
                    if key in data:
                        frames[cam].append(data[key])

            # display each frame set-by-set
            for idx in range(len(data_files)):
                combined = None
                # stack horizontally
                for cam in cams:
                    if idx < len(frames[cam]):
                        f = cv2.cvtColor(frames[cam][idx], cv2.COLOR_RGB2BGR)
                        if combined is None:
                            combined = f
                        else:
                            combined = cv2.hconcat([combined, f])
                if combined is not None:
                    cv2.imshow(f"trial_{trial_name}", combined)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return
            keyboard.wait('enter')
            cv2.destroyAllWindows()

    def download_image(self, save_dir, make_gifs=False):
        """Dump RGB frames to ``save_dir/demo_X/images/``.

        If ``make_gifs`` is True the fixed‑left/fixed‑right/in‑hand streams
        will also be stitched into GIFs under ``save_dir/demo_X/gifs/``.
        """

        self.countTrials()
        os.makedirs(save_dir, exist_ok=True)

        for trial_name in self.get_trial_names():
            trial_path = os.path.join(self.directory, trial_name)
            save_trial_dir = os.path.join(save_dir, trial_name)
            save_image_dir = os.path.join(save_trial_dir, "images")
            save_gif_dir = os.path.join(save_trial_dir, "gifs")
            os.makedirs(save_image_dir, exist_ok=True)
            if make_gifs:
                os.makedirs(save_gif_dir, exist_ok=True)

            data_files = self.get_data_files(trial_path)

            # buffers for gif creation
            frames_buf = { 'fixed_left': [], 'fixed_right': [], 'in_hand': [] }

            for name in data_files:
                data = np.load(os.path.join(trial_path, name))
                for cam in ('fixed_left', 'fixed_right', 'in_hand'):
                    key = f"{cam}_color"
                    if key in data:
                        frame = data[key]
                        # resize for consistency and convert when saving
                        frame = cv2.resize(frame, (424, 240), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(
                            os.path.join(save_image_dir, f"{name.replace('.npz','')}_{cam}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        )
                        if make_gifs:
                            frames_buf[cam].append(frame)

            if make_gifs:
                for cam, frames in frames_buf.items():
                    if frames:
                        imageio.mimsave(
                            os.path.join(save_gif_dir, f"{cam}.gif"),
                            frames,
                            duration=0.1,
                            loop=1,
                        )


if __name__ == "__main__":
    # sample usage – adjust the paths to suit your dataset
    data_dir = "/home/msc/Documents/crx_rmi_utils/teleoperation/collected_data/trash_disposal/success"
    out_dir = "/home/msc/Documents/crx_rmi_utils/teleoperation/view_dataset/trash_disposal/success"

    viewer = dataViewer(data_dir)

    # plot the sent_state trajectory for the first few trials
    # viewer.plot(out_dir, state_key='sent_state')

    # export all colour frames + make GIFs
    viewer.download_image(out_dir, make_gifs=True)

    # interactively inspect videos
    # viewer.view_images()
    
