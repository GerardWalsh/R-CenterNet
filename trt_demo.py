from operator import mod
import os
import sys
import argparse
from time import perf_counter
import math

import cv2
import torch
import numpy as np
from torch import nn
import torch.onnx
import tensorrt as trt
import pycuda.autoinit
import pycuda
import pycuda.driver as cuda
from imutils import paths
import ipdb

sys.path.append(r"./backbone")
from resnet import ResNet

from predict import pre_process  # , ctdet_decode, post_process, merge_outputs

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="The root folder to the streams subfolders.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model for inference.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        help="Model for inference.",
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Whether to display detections"
    )
    parser.add_argument(
        "--post-process",
        type=bool,
        default=False,
        help="Whether to process network output",
    )
    return parser.parse_args()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(
    max_batch_size=1,
    onnx_file_path="",
    engine_file_path="",
    fp16_mode=False,
    int8_mode=False,
    overwrite=False,
):
    def build_engine(max_batch_size):

        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            explicit_batch
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False

            if int8_mode:
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found".format(onnx_file_path))

            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                res = parser.parse(model.read())
            if res:
                print("Completed parsing of ONNX file")
                print("# Layers = ", network.num_layers)
                print(
                    "Building an engine from file {}; this may take a while...".format(
                        onnx_file_path
                    )
                )
            else:
                print("Parse Failed, Layers = ", network.num_layers)
                exit()

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if engine_file_path:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path) and not overwrite:
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    elif not os.path.exists(onnx_file_path):
        print("Cannot find any ONNX file or TRT file")
        exit()
    else:
        return build_engine(max_batch_size)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def pre_process_image(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 224, 224
    c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(
        image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR
    )

    inp_image = (inp_image / 255.0).astype(np.float32)

    return inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width), {
        "c": c,
        "s": s,
        "out_height": inp_height // 4,
        "out_width": inp_width // 4,
    }


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    num_classes = 1
    dets = ctdet_post_process(
        dets.copy(),
        [meta["c"]],
        [meta["s"]],
        meta["out_height"],
        meta["out_width"],
        num_classes,
    )
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
        dets[0][j][:, :5] /= 1
    return dets[0]


def ctdet_decode(heat, wh, ang, reg=None, K=100):
    batch, _, _, _ = heat.size()
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat(
        [
            xs - wh[..., 0:1] / 2,
            ys - wh[..., 1:2] / 2,
            xs + wh[..., 0:1] / 2,
            ys + wh[..., 1:2] / 2,
            ang,
        ],
        dim=2,
    )
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:6].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        ret.append(top_preds)
    return ret


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
        kth = len(scores) - max_obj_per_img
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, 2 + 1):
            keep_inds = detections[j][:, 5] >= thresh
            detections[j] = detections[j][keep_inds]
    return detections


def get_vertices_points(rotated_bounding_box):
    x0, y0, width, height, _angle = rotated_bounding_box

    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    #
    pt0 = Point(int(x0 - a * height - b * width), int(y0 + b * height - a * width))
    pt1 = Point(int(x0 + a * height - b * width), int(y0 - b * height - a * width))
    pt2 = Point(int(2 * x0 - pt0.x), int(2 * y0 - pt0.y))
    pt3 = Point(int(2 * x0 - pt1.x), int(2 * y0 - pt1.y))
    pts = [pt0, pt1, pt2, pt3]
    return pts


def draw_polygon(image, pts, colour=(255, 0, 0), thickness=2):
    """
    Draws a rectangle on a given image.
    :param image: What to draw the rectangle on
    :param pts: Array of point objects
    :param colour: Colour of the rectangle edges
    :param thickness: Thickness of the rectangle edges
    :return: Image with a rectangle
    """

    for i in range(0, len(pts)):
        n = (i + 1) if (i + 1) < len(pts) else 0
        cv2.line(image, (pts[i].x, pts[i].y), (pts[n].x, pts[n].y), colour, thickness)

    return image


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    # ipdb.set_trace()
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def dump_boxes_to_text(bboxes, filename="output.txt", label="defect"):
    with open(filename, "w") as f:
        for box in bboxes:
            f.write(label + " ")
            for coord in box:
                f.write(str(coord) + " ")
            f.write("\n")


def get_image_data(root_path, input_size):
    image_paths = [x for x in paths.list_images(root_path)]
    image_paths.sort()
    images = [cv2.imread(path) for path in image_paths]
    preprocessed_images = [
        pre_process(image, image_size=input_size) for image in images
    ]
    return preprocessed_images, images, image_paths


def demo(root_path, input_size):
    # debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
    # model = ResNet(18)
    # model.load_state_dict(torch.load("./last_18_224_1e4.pth"))
    # x = torch.randn((1, 3, 224, 224)).float().cuda()
    # model.eval()
    # model.cuda()
    # out_shape = outs.shape

    trt_engine_path = str(args.model_path)

    engine = get_engine(1, "", trt_engine_path, int8_mode=True)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    preprocessed_image_data, images, image_paths = get_image_data(root_path, input_size)

    # torch_outs = model(images[0][0])
    # ipdb.set_trace()
    feature_map_size = int(input_size // 4)
    import ipdb

    print("Starting inference")
    t = perf_counter()
    for data in zip(preprocessed_image_data, images, image_paths):
        (preprocessed_input, meta), image, image_path = data  # transpose(2, 0, 1)
        # ipdb.set_trace()
        inputs[0].host = preprocessed_input[0].numpy().reshape(-1)
        output_data = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        output = [
            np.expand_dims(x.reshape((-1, feature_map_size, feature_map_size)), 0)
            for x in output_data
        ]
        # ipdb.set_trace()
        if args.post_process:
            hm = torch.Tensor(output[0]).sigmoid_()
            ang = torch.Tensor(output[2]).relu_()
            wh = torch.Tensor(output[1])
            reg = torch.Tensor(output[3])

            dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)
            dets = post_process(dets, meta)
            ret = merge_outputs(dets)

            detections = np.empty([1, 7])
            for i, c in ret.items():
                tmp_s = ret[i][ret[i][:, 5] > 0.3]
                tmp_c = np.ones(len(tmp_s)) * (i + 1)
                tmp = np.c_[tmp_c, tmp_s]
                detections = np.append(detections, tmp, axis=0)
                detections = np.delete(detections, 0, 0)
                detections = detections.tolist()

            detection_lol = []
            for detection in detections:
                class_name, lx, ly, rx, ry, ang, prob = detection
                detection_list = [
                    (rx + lx) / 2,
                    (ry + ly) / 2,
                    (rx - lx),
                    (ry - ly),
                    ang,
                ]
                detection_lol.append([prob] + detection_list)

            # ipdb.set_trace()
            detection_txt_file_path = image_path.split(".")[0] + ".txt"
            dump_boxes_to_text(detection_lol, detection_txt_file_path)
            if args.display:
                detection_lol = []
                for detection in detections:
                    class_name, lx, ly, rx, ry, ang, prob = detection
                    detection_list = [
                        (rx + lx) / 2,
                        (ry + ly) / 2,
                        (rx - lx),
                        (ry - ly),
                        ang,
                    ]
                    detection_lol.append(detection_list)
                for rotated_bounding_box in detection_lol:
                    cv2.circle(
                        image,
                        (int(rotated_bounding_box[0]), int(rotated_bounding_box[1])),
                        2,
                        (255, 0, 0),
                        -1,
                    )
                    pts = get_vertices_points(rotated_bounding_box)
                    draw_polygon(image, pts)
                # ipdb.set_trace()
                cv2.imshow("test", image)
                key = cv2.waitKey(1)

        # ipdb.set_trace()
    t1 = perf_counter()
    print("FPS: {}".format(len(images) / (t1 - t)))


if __name__ == "__main__":
    args = parse_args()
    demo(args.dir, args.input_size)
