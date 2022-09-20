import xml.etree.ElementTree as ET
import json
import sys
import os

def get_obj(setting, kc_pixel_lst):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    object = json.load(open(f'{dirname}/annotateObject.json'))
    width = setting['bnd_width']
    object['object']['name'] = kc_pixel_lst[0]
    object['object']['bndbox']['xmin'] = str(kc_pixel_lst[1] - int(width) / 2)
    object['object']['bndbox']['ymin'] = str(kc_pixel_lst[2] - int(width) / 2)
    object['object']['bndbox']['xmax'] = str(kc_pixel_lst[1] + int(width) / 2)
    object['object']['bndbox']['ymax'] = str(kc_pixel_lst[2] + int(width) / 2)
    return object

def get_header(setting, render_number):

    dirname, _ = os.path.split(os.path.abspath(__file__))
    header = json.load(open(f'{dirname}/annotateHeader.json'))
    header['filename'] = setting['object_name']
    header['path'] = setting['yolo_image_path'] + setting['object_name'] + f'_{render_number}.jpg'
    header['size']['width'] = setting['px_width']
    header['size']['height'] = setting['px_height']
    return header

def dict_to_xml(setting, kc_pixel_lst, render_number):
    header = get_header(setting, render_number)

    # write xml  
    annotation = ET.Element('annotation')
    for key, val in header.items():
        child = ET.SubElement(annotation, key)
        if key == 'size' or  key == 'source':
            for subkey in val:
                subchild = ET.SubElement(child, subkey)
                subchild.text = val[subkey]
        else:
            child.text = str(val)

    for kc in kc_pixel_lst:
        object = get_obj(setting, kc)
        child = ET.SubElement(annotation, 'object')
        for key in object['object']:
            subchild = ET.SubElement(child, key)
            if key == 'bndbox':
                for subkey in object['object'][key]:
                    thrchild = ET.SubElement(subchild, subkey)
                    thrchild.text = str(object['object'][key][subkey])
            else:        
                subchild.text = str(object['object'][key])
    return annotation

def main(render_number, kc_pixel_lst):
    setting = json.load(open(f'{dirname}/../../blender/setting.json'))
    annotation = dict_to_xml(setting, kc_pixel_lst, render_number)
    ET.indent(annotation)

    path = setting['yolo_xml_path']
    obj_name = setting['object_name']
    tree = ET.ElementTree(annotation)

    tree.write(f'{path}/{obj_name}_{render_number}.xml')

if __name__ == "__main__":
    dirname, _ = os.path.split(os.path.abspath(__file__))
    path = f'{dirname}/../../blender/setting.json'
    setting = json.load(open(f'{dirname}/../../blender/setting.json'))
    with open(f'{dirname}/../../../tmp/kc_pixel_lst.txt', 'r') as f:
        kc_pixel_lst = json.load(f)

    render_number = int(sys.argv[1])
    main(render_number, kc_pixel_lst)

