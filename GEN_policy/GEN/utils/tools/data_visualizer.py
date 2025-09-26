import h5py
import pdb
import numpy as np
import matplotlib.pyplot as plt

def visualize_global_plan_and_cmd(hdf5_path):
    with h5py.File(hdf5_path, 'r') as root:
        for sample_name in root['samples']:
            cmd_linear = root[f'samples/{sample_name}/target_linear'][:].astype(np.float32)  # Left shape: (3,)
            linear_speed = cmd_linear[0].item()
            cmd_angular = root[f'samples/{sample_name}/target_angular'][:].astype(np.float32)  # Left shape: (3,)
            angular_speed = cmd_angular[2].item()
            global_plan = root[f'samples/{sample_name}/global_plan'][:].astype(np.float32)  # Left shape: (n, 7)
            
            print(f"linear_speed: {linear_speed}, angular_speed: {angular_speed}")
            plt.scatter(global_plan[:,0], global_plan[:,1], c=np.arange(len(global_plan)),cmap='viridis',s=20)
            plt.xlim(-1.5,1.5);plt.ylim(-1.5,1.5)
            plt.gca().set_aspect('equal')
            plt.show()

if __name__ == '__main__':
    hdf5_path = '/home/cvte/twilight/data/gen_nav/warehouse/h5py/batch_00000.hdf5'
    visualize_global_plan_and_cmd(hdf5_path)