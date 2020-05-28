import argparse
import os

import cv2
import torch
from PIL import Image

import cariface
from datagen import test_transform
import openmesh as om


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, default=None, help='source caricature image')
    args = parser.parse_args()

    model = cariface.CariFace()
    model.init_numbers(cpu=True)
    model.init_data()
    model.load_model()
    model.model1.eval()
    model.model2.eval()

    source_image = cv2.imread(args.image)
    source_image = source_image[:, :, ::-1]
    source_image_t = test_transform(Image.fromarray(source_image))
    source_image_work = model.to_device(source_image_t).float()
    # source_image_work = source_image_work.reshape([1, 3, 224, 224])
    source_image_work = source_image_work.unsqueeze(0)

    with torch.no_grad():
        # solve points
        points, euler_angle, scale, trans = model.solve_points(source_image_work)
        # solve landmarks
        lands_2d = cariface.CalculateLandmark2D(euler_angle, scale, trans, points, model.landmark_index,
                                                model.landmark_num)

    mesh = om.read_trimesh("toy_example/mean_face.obj")
    vertex = points.squeeze().numpy()
    for i in range(6144):
        mesh.points()[i] = vertex[:, i]
    om.write_mesh(os.path.splitext(os.path.basename(args.image))[0]+".obj", mesh)
