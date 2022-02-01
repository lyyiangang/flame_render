import numpy as np
import cv2
import sys
import time

import torch
sys.path.append('.')
import mesh
from FLAME import FLAME

def display(raw_rgb_img):
    img = (raw_rgb_img * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('bgr_img', bgr_img)
    cv2.waitKey(0)

def run():
    """
    world coordinate system:
         y
         |
         |
         |
         |
         |
         O-------------------x
        /
       /
      /
     /
    /
   z
    -------------
    """
    h, w = 256, 256
    flame_model_path = '../flame_head_model/FLAME2020/generic_model.pkl'

    # Creating a batch of different global poses
    # pose_params_numpy[:, :3] : global rotaation
    # pose_params_numpy[:, 3:] : jaw rotaation
    # 0.0, 30.0*np.pi/180, 0.0, 0.0, 0.0, 0.0
    # 6 params.[pitch, yaw,roll, jaw, unknown, unknown]
    pose_params_numpy = np.array([[np.pi / 4, np.pi/3, 0.0, 0.0, 0, 0],
                                    ], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
    neck_pose = np.array([[0, 0, 0]], np.float32)
    # pitch, yaw, roll
    # neck_pose = np.array([[0, np.pi/4, np.pi/4]], np.float32)
    neck_pose = torch.from_numpy(neck_pose).cuda()
    flame_model = FLAME(flame_model_path)
    flame_model.cuda()

    shape_params = torch.zeros(1, 100).cuda()
    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
    vertices = flame_model(shape_params, expression_params, pose_params, neck_pose).cpu().numpy().squeeze() # For RingNet project
    # torch.Size([1, 5023, 3])
    colors = (np.ones([vertices.shape[0], 3], np.float32) * [176, 191, 192])/255
    triangles = flame_model.faces
    # colors = colors/np.max(colors)# map to [0, 1]
    vertices -= np.mean(vertices, 0)[None, :] # move face coordinate system center to world cs center

    # convert obj to world cs 
    obj = {}
    # scale face model to real size
    obj['s'] = h /(np.max(vertices[:,1]) - np.min(vertices[:,1])) 
    obj['angles'] = [0, 0, 0]
    obj['t'] = [0, 0, 0]
    # convert face to world coordinate system
    R = mesh.transform.angle2matrix(obj['angles'])
    # s*X*R + t
    world_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])

    # define camera in world cs
    camera = {}
    camera['proj_type'] = 'orthographic' # perspective or orthographic
    camera['eye'] = [0, 0, 500] # eye location in world coordinate system
    camera['at'] = [0, 0, 0] # gaze at position
    camera['up'] = [0, 1, 0]
    # for perspective case
    camera['near'] = 1000
    camera['far'] = -100
    camera['fovy'] = 30

    # add 1 point light
    light_intensities = np.array([[1, 1, 1]], dtype = np.float32)
    light_positions = np.array([[0, 0, 300]])

    t_start = time.time()
    # add 2 point lights
    light_intensities = np.array([[1, 1, 1], [1, 1, 1]], dtype = np.float32)
    light_positions = np.array([[300, 0, 300], [-300, 0, 300]])
    colors = mesh.light.add_light(world_vertices, triangles, colors, light_positions, light_intensities, flip_normal= True)

    if camera['proj_type'] == 'orthographic':
        camera_vertices = mesh.transform.orthographic_project(world_vertices) # not normalized
        projected_vertices = mesh.transform.lookat_camera(camera_vertices, camera['eye'], camera['at'], camera['up'])
    else:
        # to camera coordinate system
        camera_vertices = mesh.transform.lookat_camera(world_vertices, camera['eye'], camera['at'], camera['up'])
        # perspective project and convert to NDC. positon is normlized to [-1, 1]
        projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
    # to image coords(position in image), if perspectvie, the coordinate will be map to image's w/h
    image_vertices = mesh.transform.to_image(projected_vertices, h, w, camera['proj_type'] == 'perspective')
    rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
    raw_rgb_img = np.minimum((np.maximum(rendering, 0)), 1)
    print(f'takes {(time.time() - t_start) * 1000}ms')
    mesh.io.write_obj_with_colors('tmp.obj', vertices, triangles, colors)
    display(raw_rgb_img)

if __name__ == '__main__':
    run()