import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# STLファイルを読み込む
your_mesh = mesh.Mesh.from_file("./Glasses_model/files/Glasses-full-assembly.stl")

# 回転行列を定義する関数
def rotate_mesh(mesh, angle_x, angle_y, angle_z):
    # 回転行列を生成するためにラジアンに変換
    angle_x = np.deg2rad(angle_x)
    angle_y = np.deg2rad(angle_y)
    angle_z = np.deg2rad(angle_z)

    # X軸周りの回転行列
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])

    # Y軸周りの回転行列
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])

    # Z軸周りの回転行列
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])

    # 総合回転行列
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    # メッシュの頂点を回転させる
    mesh.vectors = np.dot(mesh.vectors.reshape(-1, 3), rotation_matrix).reshape(-1, 3, 3)

# 任意の角度でメッシュを回転させる
rotate_mesh(your_mesh, 45, 30, 60)  # 例として45度、30度、60度で回転

# 3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, facecolors='blue', edgecolors='black'))

# スケールを自動調整
scale = your_mesh.points.flatten('F')
ax.auto_scale_xyz(scale, scale, scale)

# 背景を透明に設定
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# 軸を非表示にする
ax.axis('off')

# カメラの角度を調整
ax.view_init(elev=30, azim=45)  # 任意の視点角度を設定

# 描画
canvas = FigureCanvas(fig)
canvas.draw()

# PNG画像として保存（透過設定）
img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
img = img.reshape(canvas.get_width_height()[::-1] + (4,))
img = np.roll(img, 3, axis=2)  # ARGBからRGBAに変換
plt.imsave('rotated_model_transparent.png', img, format='png')

plt.show()