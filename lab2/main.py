import os
from PIL import Image
import numpy as np

# Изменение рабочей директории на ту, где находится скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_image(filename: str) -> Image.Image:
    """Загружает изображение и конвертирует его в RGB."""
    print("Пытаюсь загрузить файл:", os.path.abspath(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")
    image = Image.open(filename)
    return image.convert('RGB')

def save_image(image_array: np.array, filename: str) -> None:
    """Сохраняет массив numpy как изображение в формате BMP."""
    img = Image.fromarray(np.uint8(image_array))
    img.save(filename)

def rgb_to_grayscale(image: Image.Image) -> np.array:
    """Преобразует RGB изображение в полутоновое (grayscale)."""
    img_array = np.array(image)
    # Взвешенное усреднение каналов (стандартные веса для преобразования в grayscale)
    grayscale_array = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
    return grayscale_array

def adaptive_threshold_binarization(grayscale_array: np.array, block_size: int = 15, C: int = 5) -> np.array:
    """Адаптивная бинаризация методом Эйквила."""
    height, width = grayscale_array.shape
    binarized_array = np.zeros_like(grayscale_array)

    for y in range(height):
        for x in range(width):
            # Определение локальной области (блока)
            x1 = max(x - block_size // 2, 0)
            x2 = min(x + block_size // 2, width - 1)
            y1 = max(y - block_size // 2, 0)
            y2 = min(y + block_size // 2, height - 1)

            # Вычисление среднего значения в блоке
            local_mean = np.mean(grayscale_array[y1:y2+1, x1:x2+1])

            # Применение порога
            if grayscale_array[y, x] > (local_mean - C):
                binarized_array[y, x] = 255
            else:
                binarized_array[y, x] = 0

    return binarized_array

def generate_report(image_filenames: list) -> None:
    """Генерирует отчет в формате Markdown."""
    # Формирование текста отчета
    report_text = """
# Лабораторная работа №2: Обесцвечивание и бинаризация растровых изображений

## Задание 1: Приведение полноцветного изображения к полутоновому
Каждое изображение было преобразовано в полутоновое с использованием взвешенного усреднения каналов (R, G, B).

## Задание 2: Приведение полутонового изображения к монохромному
К каждому полутоновому изображению была применена адаптивная бинаризация методом Эйквила.

"""

    for filename in image_filenames:
        report_text += f"""
### Изображение: {filename}

#### Исходное изображение
![Исходное изображение]({filename})

#### Полутоновое изображение
![Полутоновое изображение](Output_Grayscale_{filename}.bmp)

#### Бинаризованное изображение
![Бинаризованное изображение](Output_Binarized_{filename}.bmp)
"""

    report_text += """
## Выводы
1. Все изображения успешно преобразованы в полутоновые с использованием взвешенного усреднения каналов (R, G, B).
2. К полутоновым изображениям применена адаптивная бинаризация методом Эйквила, что позволило получить монохромные изображения.
3. Результаты работы сохранены в соответствующих файлах.
"""

    # Сохранение отчета в файл
    with open('report_lab2.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Проверка текущей рабочей директории
    print("Текущая рабочая директория:", os.getcwd())

    # Список файлов для обработки
    image_filenames = ['fingerprint.png', 'map.png', 'photo.png', 'screenshot.png', 'text.png', 'x_ray.png']

    # Обработка каждого изображения
    for filename in image_filenames:
        print(f"Обработка файла: {filename}")
        image = load_image(filename)

        # 1. Приведение к полутоновому изображению
        grayscale_array = rgb_to_grayscale(image)
        save_image(grayscale_array, f'Output_Grayscale_{filename}.bmp')

        # 2. Адаптивная бинаризация
        binarized_array = adaptive_threshold_binarization(grayscale_array)
        save_image(binarized_array, f'Output_Binarized_{filename}.bmp')

    # Генерация отчета
    generate_report(image_filenames)

    print("Лабораторная работа выполнена. Полученные файлы с изображениями:")
    for filename in image_filenames:
        print(f"Output_Grayscale_{filename}.bmp, Output_Binarized_{filename}.bmp")
    print("Отчет сохранен в файле report_lab2.md")

if __name__ == "__main__":
    main()