import bpy
from mathutils import Matrix, Vector
import bpy_extras
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import json
import os
import random
# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_3x3_intrinsic_matrix(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    
    # Square pixel: pixel_aspect_ratio = 1
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

def get_3x4_extrinsic_matrix(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))
        
    location, rotation = cam.matrix_world.decompose()[0:2]
    
    R_world2bcam = rotation.to_matrix().transposed()   
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def projection_matrix(cam):
    K = get_3x3_intrinsic_matrix(cam.data)
    RT = get_3x4_extrinsic_matrix(cam)
    return K@RT, K, RT

def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    # Returns the camera space coords for a 3d point.
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)

    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )

    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))

def get_vertex_normal():
    directory = os.path.dirname(bpy.data.filepath)
    vertice_selected = np.load(f'{directory}/../../data/key_component/vertice.npy')
    normal_selected = np.load(f'{directory}/../../data/key_component/normal.npy')
    vertice_selected = vertice_selected.tolist()
    normal_selected = normal_selected.tolist()
    return vertice_selected, normal_selected

def get_angle(normal1, normal2):
    normal1, normal2 = array(normal1), array(normal2)
    cosine = dot(normal1, normal2) / norm(normal1) / norm(normal2)
    angle = arccos(clip(cosine, -1, 1))
    return angle 

def load_setting_file():
    directory = os.path.dirname(bpy.data.filepath)
    setting = json.load(open(f'{directory}/setting.json'))
    return setting

def load_kc_data():
    directory = os.path.dirname(bpy.data.filepath)
    object_data = json.load(open(f'{directory}/../../data/key_component/data.json'))
    return object_data

def get_image_with_background(setting, cnt):
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)
        

    # Blender Node Setting
    directory = os.path.dirname(bpy.data.filepath)
    image_path_lst = os.listdir(f'{directory}/../../data/yolo/background/')
    if len(image_path_lst) == 0:
        print('Please save your background data into dir.')
        return
    idx = np.random.randint(len(image_path_lst))
    image_path = f'{directory}/../../data/yolo/background/{image_path_lst[idx]}'


    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    composite_node = tree.nodes.new('CompositorNodeComposite')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')

    alpha_over_node.use_premultiply = True
    alpha_over_node.premul = 1
    composite_node.use_alpha = False
    scale_node.space = 'RENDER_SIZE'
    image_node.image = bpy.data.images.load(image_path)

    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])  # link Renger Image to Viewer Image
    links.new(render_layer_node.outputs[2], composite_node.inputs[1])  # link Render Z to Viewer Alpha
    links.new(alpha_over_node.outputs[0], composite_node.inputs[0])

    # render
    bpy.context.scene.render.resolution_percentage = 100 # make sure scene height and width are ok (edit)
    path = setting['yolo_image_path']
    obj_name = setting['object_name']

    bpy.context.scene.render.filepath = f'{path}{obj_name}_{cnt}.jpg'
    bpy.ops.render.render(write_still = True)