from scipy.io import loadmat
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image 
# Load the data
camera_data = loadmat('calib_asus.mat')
depth_10 = loadmat('depth_10.mat')
depth14 = loadmat('depth_14.mat')
depth17 = loadmat('depth_17.mat')

depth_cam = camera_data['Depth_cam']
rgb_cam = camera_data['RGB_cam']
R_d_to_rgb = camera_data['R_d_to_rgb']
T_d_to_rgb = camera_data['T_d_to_rgb']

depth_10_array = depth_10['depth_array'].astype(np.float64)
depth_14_array = depth14['depth_array'].astype(np.float64)
depth_17_array = depth17['depth_array'].astype(np.float64)

k_depth = depth_cam['K'][0][0]
R_depth = np.eye(3)
T_depth = np.zeros(shape=(3,1))

k_rgb = rgb_cam['K'][0][0]
R_rgb = R_d_to_rgb
T_rgb = T_d_to_rgb

def get_z(u,v):
    return (depth_14_array[u,v]/1000)

def compute_x_y(u,v):
    z = get_z(u,v)
    RT = np.hstack([R_depth,T_depth])
    K = k_depth
    K_inv = linalg.pinv(K)
    RT_inv = linalg.pinv(RT)
    uv_1 = np.array([u,v,1])
    uv_1 = uv_1.reshape(3,1)
    xy = z * np.matmul(np.matmul(RT_inv, K_inv),uv_1)
    return xy[0],xy[1], xy[2]

def compute_x_y_rgb(u,v):
    z = get_z(u,v)
    RT = np.hstack([R_rgb,T_rgb])
    K = k_rgb
    K_inv = linalg.inv(K)
    RT_inv = linalg.pinv(RT)
    KRT = np.matmul(K,RT)
    KRT_inv = linalg.pinv(KRT)
    uv_1 = np.array([u,v,1])
    uv_1 = uv_1.reshape(3,1)
    xy = np.matmul(KRT_inv,uv_1)
    xy[2] = z * xy[2]
    return xy[0],xy[1], xy[2]

rgb_image = Image.open('rgb_image_14.png')
rgb_image = np.array(rgb_image)
points = []
colors = [] 
for u in range(depth_14_array.shape[0]):
    for v in range(depth_14_array.shape[1]):
        # z = get_z(u, v)
        x, y, z = compute_x_y(u, v)
        point = np.array([x,y,z]).flatten()
        points.append(point)

        # Extract RGB values from the RGB image at the computed coordinates
        if 0 <= u < rgb_image.shape[0] and 0 <= v < rgb_image.shape[1]:
            color = rgb_image[u, v]  # Assuming the image is in HxWx3 format
            colors.append(color / 255.0)  # Normalize color values to [0, 1]
        else:
            colors.append([0, 0, 0])    

points = np.array(points, dtype=np.float64)
colors = np.array(colors, dtype=np.float64)

# Create and visualize the RGB point cloud
pcd = o3d.geometry.PointCloud()
if points.ndim == 2 and points.shape[1] == 3:
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors.shape[0] == points.shape[0] and colors.shape[1] == 3:
        pcd.colors = o3d.utility.Vector3dVector(colors)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        o3d.visualization.draw_geometries([downsampled_pcd])
    else:
        print("Color array shape is incorrect:", colors.shape)
else:
    print("Points array shape is incorrect:", points.shape)

    