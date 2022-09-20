import xml.etree.ElementTree as ET
import os
import json

setting = json.load(open('../../blender/setting.json'))
kc_name = json.load(open("../../../data/key_component/data.json"))['kc_name']
pixel_width = int(setting['px_width'])
pixel_height = int(setting['px_height'])

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(filename):

    in_file = open(setting['yolo_xml_path'] + filename)
    out_file = open(setting['yolo_txt_path'] + os.path.splitext(filename)[0] + '.txt', 'w') 
    tree = ET.parse(in_file)
    root = tree.getroot()
 
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_id = kc_name.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

        if (b[0] > 0 and b[0] < pixel_width) and (b[1] > 0 and b[1] < pixel_width) and (b[2] > 0 and b[2] < pixel_height) and (b[3] > 0 and b[3] < pixel_height):
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

directory = os.fsencode(setting['yolo_xml_path'])
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".xml"):
        try:
            convert_annotation(filename)
        except Exception as e:
            print(e)
