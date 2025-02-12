import time
import open3d as o3d

class PointCloudVisualizer:
    def __init__(self):
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()

    def add_pointclouds(self, pcd_list):
        for pcd in pcd_list:
            self.viz.add_geometry(pcd)

    def render(self):
        while True:
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.005)  # makes display a little smoother

