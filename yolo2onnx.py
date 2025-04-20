from ultralytics import YOLO

# 加载已训练好的模型
model = YOLO('best.pt')

# 导出为 ONNX 格式，保持精度
model.export(format='onnx', 
             dynamic=True,     # 支持动态输入尺寸
             simplify=False,    # 简化 ONNX 图结构
             opset=12)         # 默认是 12，可按需要修改

# 输出的文件将保存在相同目录下，名为 best.onnx
