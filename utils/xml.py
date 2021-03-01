import codecs
import os
from xml.etree import ElementTree

import cv2
import numpy as np
from lxml import etree

ENCODE_METHOD = "utf-8"
DECODE_METHOD = "utf-8"


def stringify_xml(elem):
    """
    Convert etree element to string
    """
    return ElementTree.tostring(elem, "utf8")


def reform_xml_from_string(xml_string):
    return etree.fromstring(xml_string)


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, "utf8")
    root = etree.fromstring(rough_string)
    return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace(
        "  ".encode(), "\t".encode()
    )


def save_prettified_xml(xml_fpath, elem):
    """
    Save a prettified version of an xml tree to a specified location.

    Parameters
    ----------
    xml_fpath - str
        Full file path where xml file will be saved.
    elem - xml.etree.Element
        Node in xml tree.
    """
    with codecs.open(xml_fpath, "w", encoding=ENCODE_METHOD) as outfile:
        prettifyResult = prettify(elem)
        outfile.write(prettifyResult.decode(DECODE_METHOD))


def create_xml_template(*args):
    """
    Create, populate and stringify all but the objects
    of a labelImg xml file. Objects are the elements that
    house the bounding box values.
    """

    (
        folder_text,
        filename_text,
        path_text,
        segmented_text,
        database_text,
        width_text,
        height_text,
        depth_text,
    ) = args

    # Elements
    annotation = etree.Element("annotation")

    folder = etree.SubElement(annotation, "folder")
    filename = etree.SubElement(annotation, "filename")
    path = etree.SubElement(annotation, "path")
    source = etree.SubElement(annotation, "source")
    size = etree.SubElement(annotation, "size")
    segmented = etree.SubElement(annotation, "segmented")
    source_database = etree.SubElement(source, "database")
    size_width = etree.SubElement(size, "width")
    size_height = etree.SubElement(size, "height")
    size_depth = etree.SubElement(size, "depth")

    # Assign values
    folder.text = folder_text
    filename.text = filename_text
    path.text = path_text

    segmented.text = segmented_text

    source_database.text = database_text

    size_width.text = width_text
    size_height.text = height_text
    size_depth.text = depth_text

    return stringify_xml(annotation)


def append_xml_object_to_annotation(robndbox, object_class, annotation):
    """
    Append an object (xml element housing bounding box information)

    Parameters
    ----------
    bndbox - np.array
        Coordinates describing bounding box proposed by an object detection model.
        Must be of shape (4,).
        Represents [left, top, bottom, right] pixel values of the bounding
        box in image.
    annotation - lxml.etree.Element
        Partially formed element used to save bounding boxes
        in LabelImg.
    object_class - str
        Class of defect found in this image.
    """

    object_ = etree.SubElement(annotation, "object")

    object_name = etree.SubElement(object_, "name")
    object_pose = etree.SubElement(object_, "pose")
    object_truncated = etree.SubElement(object_, "truncated")
    object_difficult = etree.SubElement(object_, "difficult")
    object_robndbox = etree.SubElement(object_, "robndbox")

    object_robndbox_cx = etree.SubElement(object_robndbox, "cx")
    object_robndbox_cy = etree.SubElement(object_robndbox, "cy")
    object_robndbox_w = etree.SubElement(object_robndbox, "w")
    object_robndbox_h = etree.SubElement(object_robndbox, "h")
    object_robndbox_angle = etree.SubElement(object_robndbox, "angle")

    object_name.text = object_class
    object_pose.text = "Unspecified"
    object_truncated.text = str(0)
    object_difficult.text = str(0)

    object_robndbox_cx.text = str(robndbox[0])
    object_robndbox_cy.text = str(robndbox[1])
    object_robndbox_w.text = str(robndbox[2])
    object_robndbox_h.text = str(robndbox[3])
    # print('In this weird fnc',  str(robndbox[4]))
    object_robndbox_angle.text = str(robndbox[4])
    
    return reform_xml_from_string(stringify_xml(annotation))


def xml_annotations_from_dict(input_dict, output_dir, image_fpath, image_shape, image_fname=None):
    """ Takes the detections of a single frame as for a 1 box example:
        {"rotated_boxes": [[cx, cy, w, h, angle]], "classes": ["defect"]}"""

    image_height, image_width, n_channels = image_shape

    boxes = input_dict["rotated_boxes"]
    # print("boxes passed to Jan's func", boxes)
    # boxes = np.array(boxes).astype(int)

    
    if not image_fname:
        target_image_fname = os.path.basename(image_fpath)
    else:
        target_image_fname = image_fname

    path_text = os.path.join(output_dir, target_image_fname)

    # Create body of labelImg format xml, and populate
    xml_str = create_xml_template(
        output_dir,
        target_image_fname,
        path_text,
        "0.1",
        "Unknown",
        str(image_width),
        str(image_height),
        str(n_channels),
    )

    annotation = reform_xml_from_string(xml_str)

    # Add bounding box objects to xml body
    for i in range(len(boxes)):
        label = input_dict["classes"][i]
        # print('Angle', boxes[4])
        annotation = append_xml_object_to_annotation(boxes[i], label, annotation)
        
    if not image_fname:
        xml_filepath = image_fpath.replace(".jpg", ".xml")
    else:
        xml_filepath = image_fname.replace(".jpg", ".xml")
    print('Saving XML to path', output_dir + xml_filepath)
    save_prettified_xml(output_dir + xml_filepath, annotation)


def get_lab_ret(xml_path):    
    ret = []
    # print(xml_path)
    with open(xml_path, 'r', encoding='UTF-8') as fp:
        ob = []
        flag = 0
        for p in fp:
            key = p.split('>')[0].split('<')[1]
            if key == 'cx':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'cy':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'w':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'h':
                ob.append(p.split('>')[1].split('<')[0])
            if key == 'angle':
                ob.append(p.split('>')[1].split('<')[0])
                flag = 1
            if flag == 1:
                x1 = float(ob[0])
                y1 = float(ob[1])
                w = float(ob[2])
                h = float(ob[3])
                angle = float(ob[4]) # reading in radians
                # print('Reading in angle from XML:', angle)

                bbox = [x1, y1, w, h, angle]  # COCO 对应格式[x,y,w,h]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret