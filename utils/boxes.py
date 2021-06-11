import pathlib

import cv2
import math
import numpy as np
import pyclipper


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Rectangle:
    def __init__(self, x, y, w, h, angle, image, flip=False, colour=(0, 255, 0)):

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = -1 * angle if flip else angle
        self.colour = colour
        self.imshape = image.shape[0:2]
        self.flip = flip
        # print(self.imshape)

    def dump_box(self, filename="mt.txt"):
        points = []
        for point in self.get_vertices_points():
            points.append([point.x, point.y])
            print(points)

        return [val for p in points for val in points]

    def draw(self, image, flip=False):
        pts = self.get_vertices_points(flip=flip)
        draw_polygon(image, pts, self.colour)

    def get_vertices_points(self, flip=False):
        x0, y0, width, height, _angle = self.x, self.y, self.w, self.h, self.angle
        if self.flip:
            x0 = self.imshape[1] - x0

        b = math.cos(_angle) * 0.5
        a = math.sin(_angle) * 0.5
        #
        pt0 = Point(int(x0 - a * height - b * width), int(y0 + b * height - a * width))
        pt1 = Point(int(x0 + a * height - b * width), int(y0 - b * height - a * width))
        pt2 = Point(int(2 * x0 - pt0.x), int(2 * y0 - pt0.y))
        pt3 = Point(int(2 * x0 - pt1.x), int(2 * y0 - pt1.y))
        pts = [pt0, pt1, pt2, pt3]
        return pts

    def find_intersection_shape_area(self, clip_rectangle, vertices=False):

        (
            point_0_rec1,
            point_1_rec1,
            point_2_rec1,
            point_3_rec1,
        ) = self.get_vertices_points()
        (
            point_0_rec2,
            point_1_rec2,
            point_2_rec2,
            point_3_rec2,
        ) = clip_rectangle.get_vertices_points()

        subj = (
            (point_0_rec1.x, point_0_rec1.y),
            (point_1_rec1.x, point_1_rec1.y),
            (point_2_rec1.x, point_2_rec1.y),
            (point_3_rec1.x, point_3_rec1.y),
        )

        if (
            len(set(subj)) != 4
            or int(self.w) == 0
            or int(self.h) == 0
            or int(clip_rectangle.w) == 0
            or int(clip_rectangle.h) == 0
        ):
            if vertices:
                return 0, []
            else:
                return 0

        clip = (
            (point_0_rec2.x, point_0_rec2.y),
            (point_1_rec2.x, point_1_rec2.y),
            (point_2_rec2.x, point_2_rec2.y),
            (point_3_rec2.x, point_3_rec2.y),
        )

        if len(set(clip)) != 4 or check_intersection_is_line(clip_rectangle):
            if vertices:
                return 0, []
            else:
                return 0

        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPath(subj, pyclipper.PT_SUBJECT, True)

        solutions = pc.Execute(
            pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO
        )

        if len(solutions) == 0:
            if vertices:
                return 0, []
            else:
                return 0

        solution = solutions[0]

        pts = list(map(lambda i: Point(i[0], i[1]), solution))

        if vertices:
            return self._area(solution), pts

        return self._area(solution)

    def find_union_shape_area(self, clip_rectangle, vertices=False):

        (
            point_0_rec1,
            point_1_rec1,
            point_2_rec1,
            point_3_rec1,
        ) = self.get_vertices_points()
        (
            point_0_rec2,
            point_1_rec2,
            point_2_rec2,
            point_3_rec2,
        ) = clip_rectangle.get_vertices_points()

        subj = (
            (point_0_rec1.x, point_0_rec1.y),
            (point_1_rec1.x, point_1_rec1.y),
            (point_2_rec1.x, point_2_rec1.y),
            (point_3_rec1.x, point_3_rec1.y),
        )

        if (
            len(set(subj)) != 4
            or int(self.w) == 0
            or int(self.h) == 0
            or int(clip_rectangle.w) == 0
            or int(clip_rectangle.h) == 0
        ):
            if vertices:
                return 0, []
            else:
                return 0

        clip = (
            (point_0_rec2.x, point_0_rec2.y),
            (point_1_rec2.x, point_1_rec2.y),
            (point_2_rec2.x, point_2_rec2.y),
            (point_3_rec2.x, point_3_rec2.y),
        )

        if len(set(clip)) != 4 or check_intersection_is_line(clip_rectangle):
            if vertices:
                return 0, []
            else:
                return 0

        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPath(subj, pyclipper.PT_SUBJECT, True)

        solutions = pc.Execute(
            pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO
        )
        solution = solutions[0]

        pts = list(map(lambda i: Point(i[0], i[1]), solution))

        if vertices:
            return self._area(solution), pts

        return self._area(solution)

    # Green's Theorem - Finds area of any simple polygon that only requires the coordinates of each vertex
    def _area(self, p):
        return 0.5 * abs(
            sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in self._segments(p))
        )

    def _segments(self, p):
        return zip(p, p[1:] + [p[0]])

    def __str__(self):
        return "Rectangle: x: {}, y: {}, w: {}, h: {}, angle: {}".format(
            self.x, self.y, self.w, self.h, self.angle
        )


def draw_polygon(image, pts, colour=(255, 255, 255), thickness=2):
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


def dump_validation_batch(epoch, save_directory, image_file_name, gt_boxes, test_boxes):
    data_types = ["gt", "detections"]
    box_data = [gt_boxes, test_boxes]
    for i, data_type in enumerate(data_types):
        dump_boxes(
            epoch=epoch,
            data_type=data_type,
            save_directory=save_directory,
            image_file_name=image_file_name,
            detections=box_data[i],
        )


def dump_box_to_text(box, filename="tx.txt", label="defect"):
    with open(filename, "a") as outfile:
        outfile.write("%s " % label)
        for val in box:
            # print(val)
            outfile.write("%.3f " % val)
        outfile.write("\n")


def dump_boxes(epoch, data_type, save_directory, image_file_name, detections):
    for box in detections:
        _, lx, ly, rx, ry, ang, prob = box
        detection_list = [
            (rx + lx) / 2,
            (ry + ly) / 2,
            (rx - lx),
            (ry - ly),
            ang,
        ]
        if data_type != "gt":
            detection_list = [prob] + detection_list

        save_path = (
            save_directory
            / pathlib.Path(f"epoch_{epoch}")
            / pathlib.Path(data_type)
            / pathlib.Path((image_file_name + ".txt"))
        )
        dump_box_to_text(
            detection_list,
            str(save_path),
        )
