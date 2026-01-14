from ultralytics import YOLO

# 加载预训练的 YOLOv11n 模型
model = YOLO('D:\PythonProject\\needle_test2_20260104\\ultralytics-8.3.200\\runs\pose\\needle_pose\weights\\best.pt')
source = 'D:\PythonProject\\needle_test2_20260104\\ultralytics-8.3.200\datasets\images\\test\\frame_196.png' #更改为自己的图片路径

# 运行推理，并附加参数
model.predict(source, save=True)

