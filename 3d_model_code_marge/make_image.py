import bpy
import numpy as np
import os
import re
import shutil
import math
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

def object_shift(object, locate_x, locate_y, locate_z, rotate_x, rotate_y, rotate_z):
    obj = object
    obj.location.x = locate_x
    obj.location.y = locate_y
    obj.location.z = locate_z
    obj.rotation_euler[0] = rotate_x
    obj.rotation_euler[1] = rotate_y
    obj.rotation_euler[2] = rotate_z

def rendering(camera_num, size):
    scene = bpy.context.scene
    scene.render.resolution_x = size[0]
    scene.render.resolution_y = size[1]
    scene.render.resolution_percentage = 50  # 解像度の低下
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    bpy.context.scene.cycles.samples = 2  # サンプル数をさらに減らす
    bpy.context.scene.cycles.use_denoising = True  # デノイザーを使用
    bpy.context.scene.cycles.max_bounces = 1  # ライトパスの減少
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'  # Eeveeを使用
    bpy.context.scene.cycles.device = 'GPU'  # GPUを使用
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in prefs.devices:
        device.use = True

    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=os.path.dirname(os.path.abspath(__file__)) + '/image/' + str("{0:03d}".format(camera_num)) + '.png')

def rotate_all_objects(rotate_x, rotate_y, rotate_z):
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.rotation_euler[0] = np.pi - rotate_x + 0.5
            obj.rotation_euler[1] = np.pi - rotate_y + 1.5
            obj.rotation_euler[2] = np.pi - rotate_z + 4.5

def get_rotation_matrix(face_vector, mouth_vector):
    x_axis = mouth_vector / np.linalg.norm(mouth_vector)
    z_axis = face_vector / np.linalg.norm(face_vector)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.vstack([x_axis, y_axis, z_axis]).T

def calculate_euler_angles(rotation_matrix):
    pitch = np.arcsin(-rotation_matrix[2, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return roll, pitch, yaw

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    shirtFolderPath = "Image"
    listShirts = os.listdir(shirtFolderPath)
    fixedRatio = 450 / 190
    shirtRatioHeightWidth = 400 / 500
    imageNumber = 0
    imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
    imgButtonLeft = cv2.flip(imgButtonRight, 1)
    counterRight = 0
    counterLeft = 0
    selectionSpeed = 10

    if os.path.exists(os.path.dirname(os.path.abspath(__file__)) + '/image'):
        shutil.rmtree(os.path.dirname(os.path.abspath(__file__)) + '/image')

    for item in bpy.data.objects:
        if item.type == 'MESH' and item.name != "Camera":
            bpy.data.objects.remove(item)
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)

    path = os.path.dirname(os.path.abspath(__file__)) + '/' + 'blender' + '/'
    file_list = os.listdir(path)
    obj_list = [item for item in file_list if item[-3:] == 'obj']
    item = obj_list[0]
    full_path_to_file = os.path.join(path, item)

    bpy.ops.wm.obj_import(filepath=full_path_to_file)
    rendering_count = 1
    image_size = [64, 64]
    rendering_distance = 3
    camera_height = 0

    camera = bpy.data.objects["Camera"]


    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findPose(img, draw=False)
        h, w, c = img.shape
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            lm3 = lmList[3]
            lm6 = lmList[6]
            lm0 = lmList[0]
            lm9 = lmList[9]
            lm10 = lmList[10]

            eye_center = (np.array([lm6[0]/w, lm6[1]/h, lm6[2]/w]) + np.array([lm3[0]/w, lm3[1]/h, lm3[2]/w])) / 2  
            mouth_center = (np.array([lm9[0]/w, lm9[1]/h, lm9[2]/w]) + np.array([lm10[0]/w, lm10[1]/h, lm10[2]/w])) / 2

            face_vector = np.array([lm0[0]/w, lm0[1]/h, lm0[2]/w]) - eye_center
            mouth_vector = mouth_center - eye_center

            rotation_matrix = get_rotation_matrix(face_vector, mouth_vector)
            roll, pitch, yaw = calculate_euler_angles(rotation_matrix)

            print(f"Pitch: {pitch}, Yaw: {yaw}, Roll: {roll}")
            initial_x = rendering_distance
            initial_y = 0
            initial_z = camera_height
            initial_rotation_x = np.pi / 2
            initial_rotation_y = 0
            initial_rotation_z = np.pi / 2
            rotate_all_objects(pitch, yaw, roll)
            object_shift(camera, initial_x, initial_y, initial_z, initial_rotation_x, initial_rotation_y, initial_rotation_z)
            rendering(camera_num=0, size=image_size)

            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
            widthOfShirt = int((lm3[0] - lm6[0]) * fixedRatio)
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm3[0] - lm6[0]) / 190
            offset = int(200 * currentScale), int(300 * currentScale)

            try:
                #img = cvzone.overlayPNG(img, imgShirt, (lm6[0] - offset[0], lm6[1] - offset[1]))
                img = cvzone.overlayPNG(img, imgShirt, (lm0[0]- offset[0], lm0[1]- offset[1]))
            except:
                pass

            cv2.imshow("Image", img)
            cv2.waitKey(1)

    bpy.ops.wm.quit_blender()
