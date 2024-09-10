from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
image_path = './1.jpg'
#加载图片
original_image = Image.open(image_path)
#定义调整图片的大小
resize_transform = transforms.Resize((32,32))
resized_image = resize_transform(original_image)

#翻转图像,p为1.0表示必翻转
flip_transform = transforms.RandomHorizontalFlip(p=1.0)
flipped_image = flip_transform(resized_image)

#将pil图像或numpy数组转换为Pytorch张量
tensor_transform = transforms.ToTensor()
tensor_image = tensor_transform(flipped_image)

#打印信息
#tensor.image是一个shape为[3,32,32]的张量
# 对于图片通常为通道数(channels),高度(height),宽度(width)

print(f"Tensor shape:{tensor_image.shape}")
#tensor_image.min()为获取张量中最小的元素，打印出来为tensor(0.500)
#.item()为转换为标准数据格式，即为float或int
print(f"Tensor values range:[{tensor_image.min().item()},{tensor_image.max().item()}]")

#Normalize((mean),(std))因为是三通道，每个通道对应一个mean和std，归一化结果为X_new = (X_old-mean)/std
# 应用到这里就是X_new = (X_old-0.5)/0.5
normalize_transform = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
normalized_image = normalize_transform(tensor_image)
if __name__ == "__main__":
    img = original_image
    plt.imshow(img)
    plt.title("img")
    plt.show()