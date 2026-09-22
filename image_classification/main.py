from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights

# https://docs.pytorch.org/vision/main/models?utm_source=chatgpt.com

'''
Превращение изображения в объект torch.Tensor, представляющий из себя
многомерную таблицу, разделенную по трём каналам RGB, в каждом из которых 
написано какая степень цвета этого канала в данном пикселе.
.shape() выводит общие данные о количестве каналов и размере изображения.
'''
img = decode_image("image_classification/samples/PchUb3A.jpg")

'''
Импортируем архитектуру resnet50 с готовыми весами (предобученная) и
переводим её в режим для использования (.eval()) а не обучения (.train())
'''
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

'''
Преобразование тензора изображения в подходящий формат для resnet50
'''
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

'''
Получаем batch изображений, убираем первое измерение (squeeze(0)) т.к. изображение 
только одно и softmax(0) превращаем все параметры объектов в классификации изображения в 
сумму равную единице.

.item() превращает из тензора в пайтон число
.argmax() выводит максимальное значение
'''
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
# превращение названия категории объекта из числа в название
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")