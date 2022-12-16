'''
Common camera utilities
'''

import math
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer.implicit.raysampling import _xy_to_ray_bundle

class RelativeCameraLoader(nn.Module):
    def __init__(self,
            query_batch_size=1,
            rand_query=True,
            relative=True,
            center_at_origin=False,
        ):
        super().__init__()

        self.query_batch_size = query_batch_size
        self.rand_query = rand_query
        self.relative = relative
        self.center_at_origin = center_at_origin

    def plot_cameras(self, cameras_1, cameras_2):
        '''
        Helper function to plot cameras

        Args:
            cameras_1 (PyTorch3D camera): cameras object to plot
            cameras_2 (PyTorch3D camera): cameras object to plot
        '''
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
        import plotly.graph_objects as go
        plotlyplot = plot_scene(
                {
                    'scene_batch': {
                        'cameras': cameras_1.to('cpu'),
                        'rel_cameras': cameras_2.to('cpu'),
                    }
                },
                camera_scale=.5,#0.05,
                pointcloud_max_points=10000,
                pointcloud_marker_size=1.0,
                raybundle_max_rays=100
            )
        plotlyplot.show()

    def concat_cameras(self, camera_list):
        '''
        Returns a concatenation of a list of cameras

        Args:
            camera_list (List[PyTorch3D camera]): a list of PyTorch3D cameras
        '''
        R_list, T_list, f_list, c_list, size_list = [], [], [], [], []
        for cameras in camera_list:
            R_list.append(cameras.R)
            T_list.append(cameras.T)
            f_list.append(cameras.focal_length)
            c_list.append(cameras.principal_point)
            size_list.append(cameras.image_size)

        camera_slice = PerspectiveCameras(
            R = torch.cat(R_list), 
            T = torch.cat(T_list), 
            focal_length = torch.cat(f_list),
            principal_point = torch.cat(c_list),
            image_size = torch.cat(size_list),
            device = camera_list[0].device,
        )
        return camera_slice

    def get_camera_slice(self, scene_cameras, indices):
        '''
        Return a subset of cameras from a super set given indices

        Args:
            scene_cameras (PyTorch3D Camera): cameras object
            indices (tensor or List): a flat list or tensor of indices

        Returns:
            camera_slice (PyTorch3D Camera) - cameras subset
        '''
        camera_slice = PerspectiveCameras(
            R = scene_cameras.R[indices], 
            T = scene_cameras.T[indices], 
            focal_length = scene_cameras.focal_length[indices],
            principal_point = scene_cameras.principal_point[indices],
            image_size = scene_cameras.image_size[indices],
            device = scene_cameras.device,
        )
        return camera_slice


    def get_relative_camera(self, scene_cameras:PerspectiveCameras, query_idx, center_at_origin=False):
        """
        Transform context cameras relative to a base query camera

        Args:
            scene_cameras (PyTorch3D Camera): cameras object
            query_idx (tensor or List): a length 1 list defining query idx

        Returns:
            cams_relative (PyTorch3D Camera): cameras object relative to query camera
        """

        query_camera = self.get_camera_slice(scene_cameras, query_idx)
        query_world2view = query_camera.get_world_to_view_transform()
        all_world2view = scene_cameras.get_world_to_view_transform()
        
        if center_at_origin:
            identity_cam = PerspectiveCameras(device=scene_cameras.device, R=query_camera.R, T=query_camera.T)
        else:
            T = torch.zeros((1, 3))
            identity_cam = PerspectiveCameras(device=scene_cameras.device, R=query_camera.R, T=T)
         
        identity_world2view  = identity_cam.get_world_to_view_transform()

        # compose the relative transformation as g_i^{-1} g_j
        relative_world2view = identity_world2view.inverse().compose(all_world2view)
        
        # generate a camera from the relative transform
        relative_matrix = relative_world2view.get_matrix()
        cams_relative = PerspectiveCameras(
                            R = relative_matrix[:, :3, :3],
                            T = relative_matrix[:, 3, :3],
                            focal_length = scene_cameras.focal_length,
                            principal_point = scene_cameras.principal_point,
                            image_size = scene_cameras.image_size,
                            device = scene_cameras.device,
                        )
        return cams_relative

    def forward(self, scene_cameras, scene_rgb=None, scene_masks=None, query_idx=None, context_size=3, context_idx=None, return_context=False):
        '''
        Return a sampled batch of query and context cameras (used in training)

        Args:
            scene_cameras (PyTorch3D Camera): a batch of PyTorch3D cameras
            scene_rgb (Tensor): a batch of rgb
            scene_masks (Tensor): a batch of masks (optional)
            query_idx (List or Tensor): desired query idx (optional)
            context_size (int): number of views for context

        Returns:
            query_cameras, query_rgb, query_masks: random query view
            context_cameras, context_rgb, context_masks: context views
        '''

        if query_idx is None:
            query_idx = [0]
            if self.rand_query:
                rand = torch.randperm(len(scene_cameras))
                query_idx = rand[:1]

        if context_idx is None:
            rand = torch.randperm(len(scene_cameras))
            context_idx = rand[:context_size]

        
        if self.relative:
            rel_cameras = self.get_relative_camera(scene_cameras, query_idx, center_at_origin=self.center_at_origin)
        else:
            rel_cameras = scene_cameras

        query_cameras = self.get_camera_slice(rel_cameras, query_idx)
        query_rgb = None
        if scene_rgb is not None:
            query_rgb = scene_rgb[query_idx]
        query_masks = None
        if scene_masks is not None:
            query_masks = scene_masks[query_idx]

        context_cameras = self.get_camera_slice(rel_cameras, context_idx)
        context_rgb = None
        if scene_rgb is not None:
            context_rgb = scene_rgb[context_idx]
        context_masks = None
        if scene_masks is not None:
            context_masks = scene_masks[context_idx]
        
        if return_context:
            return query_cameras, query_rgb, query_masks, context_cameras, context_rgb, context_masks, context_idx
        return query_cameras, query_rgb, query_masks, context_cameras, context_rgb, context_masks


def get_interpolated_path(cameras: PerspectiveCameras, n=50, method='circle', theta_offset_max=0.0):
    '''
    Given a camera object containing a set of cameras, fit a circle and get 
    interpolated cameras

    Args:
        cameras (PyTorch3D Camera): input camera object
        n (int): length of cameras in new path
        method (str): 'circle'
        theta_offset_max (int): max camera jitter in radians

    Returns:
        path_cameras (PyTorch3D Camera): interpolated cameras
    '''
    device = cameras.device
    cameras = cameras.cpu()

    if method == 'circle':

        #@ https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
        #@ Fit plane
        P = cameras.get_camera_center().cpu()
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U,s,V = torch.linalg.svd(P_centered)
        normal = V[2,:]
        if (normal*2 - P_mean).norm() < (normal - P_mean).norm():
            normal = - normal
        d = -torch.dot(P_mean, normal)  # d = -<p,n>    

        #@ Project pts to plane
        P_xy = rodrigues_rot(P_centered, normal, torch.tensor([0.0,0.0,1.0]))
        
        #@ Fit circle in 2D
        xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
        t = torch.linspace(0, 2*math.pi, 100)
        xx = xc + r*torch.cos(t)
        yy = yc + r*torch.sin(t)

        #@ Project circle to 3D
        C = rodrigues_rot(torch.tensor([xc,yc,0.0]), torch.tensor([0.0,0.0,1.0]), normal) + P_mean
        C = C.flatten()

        #@ Get pts n 3D
        t = torch.linspace(0, 2*math.pi, n)
        u = P[0] - C
        new_camera_centers = generate_circle_by_vectors(t, C, r, normal, u)

        #@ OPTIONAL THETA OFFSET
        if theta_offset_max > 0.0:
            aug_theta = (torch.rand((new_camera_centers.shape[0])) * (2*theta_offset_max)) - theta_offset_max
            new_camera_centers = rodrigues_rot2(new_camera_centers, normal, aug_theta)

        #@ Get camera look at
        new_camera_look_at = get_nearest_centroid(cameras)

        #@ Get R T
        up_vec = -normal
        R, T = look_at_view_transform(eye=new_camera_centers, at=new_camera_look_at.unsqueeze(0), up=up_vec.unsqueeze(0), device=cameras.device)
    else:
        raise NotImplementedError
    
    c = (cameras.principal_point).mean(dim=0, keepdim=True).expand(R.shape[0],-1)
    f = (cameras.focal_length).mean(dim=0, keepdim=True).expand(R.shape[0],-1)
    image_size = cameras.image_size[:1].expand(R.shape[0],-1)


    path_cameras = PerspectiveCameras(R=R,T=T,focal_length=f,principal_point=c,image_size=image_size, device=device)
    cameras = cameras.to(device)
    return path_cameras

def np_normalize(vec, axis=-1):
    vec = vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)
    return vec


#@ https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
#-------------------------------------------------------------------------------
# Generate points on circle
# P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
#-------------------------------------------------------------------------------
def generate_circle_by_vectors(t, C, r, n, u):
    n = n/torch.linalg.norm(n)
    u = u/torch.linalg.norm(u)
    P_circle = r*torch.cos(t)[:,None]*u + r*torch.sin(t)[:,None]*torch.cross(n,u) + C
    return P_circle

#@ https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
#-------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
#-------------------------------------------------------------------------------
def fit_circle_2d(x, y, w=[]):
    
    A = torch.stack([x, y, torch.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = torch.diag(w)
        A = torch.dot(W,A)
        b = torch.dot(W,b)
    
    # Solve by method of least squares
    c = torch.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = torch.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

#@ https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[None,...]
    
    # Get vector of rotation k and angle theta
    n0 = n0/torch.linalg.norm(n0)
    n1 = n1/torch.linalg.norm(n1)
    k = torch.cross(n0,n1)
    k = k/torch.linalg.norm(k)
    theta = torch.arccos(torch.dot(n0,n1))
    
    # Compute rotated points
    P_rot = torch.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*torch.cos(theta) + torch.cross(k,P[i])*torch.sin(theta) + k*torch.dot(k,P[i])*(1-torch.cos(theta))

    return P_rot

def rodrigues_rot2(P, n1, theta):
    '''
    Rotate points P wrt axis k by theta radians
    '''
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[None,...]
    
    k = torch.cross(P, n1.unsqueeze(0))
    k = k/torch.linalg.norm(k)
    
    # Compute rotated points
    P_rot = torch.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*torch.cos(theta[i]) + torch.cross(k[i],P[i])*torch.sin(theta[i]) + k[i]*torch.dot(k[i],P[i])*(1-torch.cos(theta[i]))

    return P_rot

#@ https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return torch.arctan2(torch.linalg.norm(torch.cross(u,v)), torch.dot(u,v))
    else:
        return torch.arctan2(torch.dot(n,torch.cross(u,v)), torch.dot(u,v))

#@ https://www.crewes.org/Documents/ResearchReports/2010/CRR201032.pdf
def get_nearest_centroid(cameras: PerspectiveCameras):
    '''
    Given PyTorch3D cameras, find the nearest point along their principal ray
    '''

    #@ GET CAMERA CENTERS AND DIRECTIONS
    camera_centers = cameras.get_camera_center()

    c_mean = (cameras.principal_point).mean(dim=0)
    xy_grid = c_mean.unsqueeze(0).unsqueeze(0)
    ray_vis = _xy_to_ray_bundle(cameras, xy_grid.expand(len(cameras),-1,-1), 1.0, 15.0, 20, True)
    camera_directions = ray_vis.directions

    #@ CONSTRUCT MATRICIES
    A = torch.zeros((3*len(cameras)), len(cameras)+3)
    b = torch.zeros((3*len(cameras), 1))
    A[:,:3] = torch.eye(3).repeat(len(cameras),1)
    for ci in range(len(camera_directions)):
        A[3*ci:3*ci+3, ci+3] = -camera_directions[ci]
        b[3*ci:3*ci+3, 0] = camera_centers[ci]
    #' A (3*N, 3*N+3)   b (3*N, 1)

    #@ SVD
    U, s, VT = torch.linalg.svd(A)
    Sinv = torch.diag(1/s)
    if len(s) < 3*len(cameras):
        Sinv = torch.cat((Sinv, torch.zeros((Sinv.shape[0], 3*len(cameras) - Sinv.shape[1]), device=Sinv.device)), dim=1)
    x = torch.matmul(VT.T, torch.matmul(Sinv,torch.matmul(U.T, b)))
    
    centroid = x[:3,0]
    return centroid


def get_angles(target_camera: PerspectiveCameras, context_cameras: PerspectiveCameras, centroid=None):
    '''
    Get angles between cameras wrt a centroid

    Args:
        target_camera (Pytorch3D Camera): a camera object with a single camera
        context_cameras (PyTorch3D Camera): a camera object

    Returns:
        theta_deg (Tensor): a tensor containing angles in degrees
    '''
    a1 = target_camera.get_camera_center()
    b1 = context_cameras.get_camera_center()

    a = a1 - centroid.unsqueeze(0)
    a = a.expand(len(context_cameras), -1)
    b = b1 - centroid.unsqueeze(0)

    ab_dot = (a*b).sum(dim=-1)
    theta = torch.acos((ab_dot)/(torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)))
    theta_deg = theta * 180 / math.pi
    
    return theta_deg