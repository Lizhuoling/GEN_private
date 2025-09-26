import open3d as o3d
import numpy as np
import pdb

def load_colored_pointcloud(npy_path):
    pc_xyzrgb = np.load(npy_path)
    pc_xyzrgb = pc_xyzrgb[pc_xyzrgb[:, 2] < 100]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:])
    return pcd

def visualize_pointcloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="colored point cloud")
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    colored_pcd = load_colored_pointcloud(npy_path = '/home/cvte/twilight/code/GEN_private/src/gen_base_nav/pc_xyzrgb.npy')
    visualize_pointcloud(pcd = colored_pcd)