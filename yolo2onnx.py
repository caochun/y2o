from ultralytics import YOLO

# 加载已训练好的模型
model = YOLO('best.pt')

model.export(
    format="onnx",
    dynamic=False,
    simplify=True,
    opset=12,
    nms=False,
    imgsz=[640, 640]  # 固定图像大小
)

# 输出的文件将保存在相同目录下，名为 best.onnx
