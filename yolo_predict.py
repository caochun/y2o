from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('best.pt')
print(model.model.yaml)  # 输出模型类名

# 分类数量
num_classes = model.model.model[-1].nc  # 最后一层是 Detect，里面有 nc（num_classes）
print(f"分类数量: {num_classes}")

# 类别标签（如果模型保存了）
if hasattr(model.model, 'names'):
    print("类别标签:")
    for i, name in model.model.names.items():
        print(f"  {i}: {name}")

# 输入图片尺寸（训练时设置的默认尺寸）
if hasattr(model.model, 'args') and hasattr(model.model.args, 'imgsz'):
    input_size = model.model.args.imgsz
    print(f"模型训练输入尺寸: {input_size} x {input_size}")
else:
    print("未找到默认输入尺寸信息")
    
# 加载图片
image_path = 'images/000002.jpg'
results = model(image_path)

# 显示检测结果（带边框）
for result in results:
    result.show()  # 使用默认窗口展示图像
    # 如果你想保存图片，也可以这样做：
    # result.save(filename='output.jpg')
