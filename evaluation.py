# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:19:02 2020

@author: Lim
"""
import os, sys
import cv2
import math
import time
import torch
import evaluation
import numpy as np
from imutils import paths

sys.path.append(r'./backbone')

from dlanet_dcn import DlaNet
from resnet import ResNet  
from predict import pre_process, ctdet_decode, post_process, merge_outputs
from dataset import coco_box_to_bbox
from utils.boxe import Rectangle


def dump_bbox(rect):

    detection_ = []
    for point in rect.get_vertices_points(): 
        detection_.append([point.x, point.y])
    return [val for sublist in detection_ for val in sublist]

def dump_box_to_text(box, filename='tx.txt'):
    with open(filename, "a") as outfile:
        for val in box:
            outfile.write("%i " % val)

        outfile.write("defect 0\n" )

def process(images, return_time=False):
    with torch.no_grad():
      output = model(images)
      hm = output['hm'].sigmoid_()
      ang = output['ang'].relu_()
      wh = output['wh']
      reg = output['reg'] 
     
    #   torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, ang, reg=reg, K=100) # K 是最多保留几个目标
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = bbox1[0] - bbox1[2] / 2.0, bbox1[1] - bbox1[3] / 2.0
        xmax1, ymax1 = bbox1[0] + bbox1[2] / 2.0, bbox1[1] + bbox1[3] / 2.0
        xmin2, ymin2 = bbox2[0] - bbox2[2] / 2.0, bbox2[1] - bbox2[3] / 2.0
        xmax2, ymax2 = bbox2[0] + bbox2[2] / 2.0, bbox2[1] + bbox2[3] / 2.0

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1 ) 
    area2 = (xmax2 - xmin2 ) * (ymax2 - ymin2 )
 
    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def iou_rotate_calculate(boxes1, boxes2):
    area1 = boxes1[2] * boxes1[3]
    area2 = boxes2[2] * boxes2[3]
    #    print("####boxes2:", boxes1.shape)
    #    print("####boxes2:", boxes2.shape)
    r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
    r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious=0
    return ious

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
                # print('reading xml angle', ob[4])
                # angle = float(ob[4])*180/math.pi
                angle = float(ob[4])
                # angle = to_deg(float(ob[4]))


                # angle = angle if angle < 180 else angle-180
                # print('angle', angle)
                # print('converted angle', angle)
                # print('\n')

                bbox = [x1, y1, w, h, angle]  # COCO 对应格式[x,y,w,h]
                ret.append(bbox)
                ob = []
                flag = 0
    return ret

def get_pre_ret(img_path, device):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (960, 540))
    images, meta = pre_process(image)
    images = images.to(device)
    output, dets, forward_time = process(images, return_time=True)
    
    dets = post_process(dets, meta)
    ret = merge_outputs(dets)
    
    res = np.empty([1,7])
    for i, c in ret.items():
        tmp_s = ret[i][ret[i][:,5]>0.3]
        tmp_c = np.ones(len(tmp_s)) * (i+1)
        tmp = np.c_[tmp_c,tmp_s]
        res = np.append(res,tmp,axis=0)
    res = np.delete(res, 0, 0)
    res = res.tolist()
    return res, image

def pre_recall(root_path, device, iou=0.5):
    imgs = paths.list_images(root_path)
    num = 0
    all_pre_num = 0
    all_lab_num = 0
    miou = 0
    mang = 0
    for img in imgs:
        img = img.split('/')[-1]
        if img.split('.')[-1] == 'jpg':
            img_path = os.path.join(root_path, img)
            xml_path = os.path.join(root_path, img.split('.')[0] + '.xml')
            detections, image = get_pre_ret(img_path, device)

            labels = get_lab_ret(xml_path)
            all_pre_num += len(detections)
            all_lab_num += len(labels)

            for class_name, lx, ly, rx, ry, ang, prob in detections:
                ang = np.interp(ang, (0, 1), (-1, 1))
                # if ang < 0:
                #     ang = -1*ang
                detection = np.array([(rx+lx)/2, (ry+ly)/2, (rx-lx), (ry-ly), ang])
                detection_rect = Rectangle(*detection, colour=(0, 0, 255))
                detection_rect.draw(image)

                print('detection angle', ang)

                for point in detection_rect.get_vertices_points(): 
                    print(point.x, point.y)
                    cv2.circle(image, (point.x, point.y), 2, (255, 0, 0), -1)

            #         cv2.imshow('test', image)
            #         key = cv2.waitKey()
            #         if key == ord('q'):
            #             break

                # dump_box_to_text(dump_bbox(detection_rect))

            # label_path = os.path.join(root_path, img.split('.')[0] + '.txt')

            # for cx, cy, w, h, ang_l in labels:
            #     label = np.array([cx, cy, w, h, ang_l])     
            #     print('Ang label', ang_l)
            #     cv2.imshow('test', image)             
            #     label[0:label.shape[0]-1] = label[0:label.shape[0]-1]/2 # scale according to image decrease in dims
            #     label_rect = Rectangle(label[0], label[1], label[2], label[3], label[4])
            #     dump_box_to_text(dump_bbox(label_rect))

            #     for point in label_rect.get_vertices_points(): 
            #         # print(point.x, point.y)
            #         cv2.circle(image, (point.x, point.y), 2, (255, 0, 0), -1)
                    
            #         # cv2.imshow('test', image)
            #         # key = cv2.waitKey()
            #         # if key == ord('q'):
            #         #     break
                
                # label_rect.draw(image)
                # dump_box_to_text(dump_bbox(label_rect), label_path)

            cv2.imshow('test', image)
            key = cv2.waitKey()
            if key == ord('q'):
                break
                    
                # #     # box = coco_box_to_bbox(lab_one)
                # #     iou = iou_rotate_calculate(detection, label)
                # #     ang_err = abs(ang - ang_l)/180

                #     # if True:
                #     #     num += 1
                #     #     miou += iou
                #     #     mang += ang_err
                #         # ratio = 1
                #         # print('Angle', ang)

        # cv2.imshow('test', image)
        # key = cv2.waitKey()
        # if key == ord('q'):
        #     break
            # filepat = img_path.split('/')[-1]
            # print('frame_' + filepat)
            # number = int(filepat.split('.')[0])
            # cv2.imwrite('frame_' + str(number-1000).zfill(10) + '.jpg', image)
            # print(img_path)
    # return num/all_pre_num, num/all_lab_num, mang/num, miou/num


if __name__ == '__main__':
    # model = ResNet(34)
    model = DlaNet(34)
    device = torch.device('cuda')
    model.load_state_dict(torch.load('./last.pth'))
    model.eval()
    model.cuda()
    
    # p, r, mang, miou = pre_recall('../tests/t2/', device)
    pre_recall('../tests/t2/', device)
    # F1 = (2*p*r)/(p+r)
    # print(F1)