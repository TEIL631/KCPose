import bpy, sys
from mathutils import Vector
import random
import subprocess
sys.modules['util'] = bpy.data.texts['util.py'].as_module()
from util import *

setting = load_setting_file()
kc_data = load_kc_data()

def snap(t, r, render_number):
    object_name = setting['object_name']
    kc_name_lst = kc_data['kc_name']

    # Insert your camera here
    cam = bpy.data.objects['Camera']
    cam.location = (0, 0, 1.1)
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1))

    # _, _, RT = projection_matrix(cam)
    object = bpy.data.objects[object_name]
    object.location = (t[0], t[1], t[2])
    object.rotation_euler = (r[0], r[1], r[2])
    bpy.context.view_layer.update()
    mat = object.matrix_world

    vertex_selected, normal_selected = get_vertex_normal()
    save_vertex_idx = []

    for idx in range(len(vertex_selected)):
        global_normal = mat.inverted().transposed() @ Vector(normal_selected[idx])
        angle = get_angle(cam_direction, global_normal)
        if angle > 1.7:
            save_vertex_idx.append(idx)

    objs_lst = []
    for idx in save_vertex_idx:
        # transforms points in the Object space into World space.
        v_co_world = mat @ Vector(vertex_selected[idx])
        pixel = project_by_object_utils(cam, v_co_world)
        objs_lst.append([kc_name_lst[idx], pixel[0], pixel[1]])
    
    with open(f'{os.path.dirname(bpy.data.filepath)}/../../tmp/kc_pixel_lst.txt', 'w') as f:
        json.dump(objs_lst, f, indent = 2)
    python_bin = setting['python_bin']

    # Path to the script that must run under the virtualenv
    script_file = f"{os.path.dirname(bpy.data.filepath)}/../util/xml/write_xml.py"
    subprocess.Popen([python_bin, script_file, str(render_number)])    
    get_image_with_background(setting, render_number)

render_number = 0
cad_rt = setting['cad_movement']
training_data_number = int(setting['training_data_number'])

for _ in range(training_data_number):
    t_range = cad_rt['translation']
    r_range = cad_rt['rotation']
    t = [random.uniform(t_range['x-axis'][0], t_range['x-axis'][1]), random.uniform(t_range['y-axis'][0], t_range['x-axis'][1]), random.uniform(t_range['z-axis'][0], t_range['z-axis'][1])]
    r = [random.uniform(r_range['x-axis'][0], r_range['x-axis'][1]), random.uniform(r_range['y-axis'][0], r_range['x-axis'][1]), random.uniform(r_range['z-axis'][0], r_range['z-axis'][1])]
    snap(t, r, render_number)
    render_number += 1
