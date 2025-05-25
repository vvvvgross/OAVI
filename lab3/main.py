import os
import numpy as np
from PIL import Image

# Изменение рабочей директории на ту, где находится скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_image(filename: str) -> np.array:
    """Загружает изображение и возвращает его как массив numpy."""
    print("Пытаюсь загрузить файл:", os.path.abspath(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")
    image = Image.open(filename)
    return np.array(image)

def save_image(image_array: np.array, filename: str) -> None:
    """Сохраняет массив numpy как изображение в формате BMP."""
    img = Image.fromarray(np.uint8(image_array))
    img.save(filename)

def rgb_to_grayscale(image: np.array) -> np.array:
    """Преобразует RGB изображение в полутоновое (grayscale)."""
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

def apply_median_filter(image: np.array, mask: np.array) -> np.array:
    """Применяет медианный фильтр с разреженной маской."""
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    # Проходим по каждому пикселю изображения
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Собираем значения пикселей по маске
            values = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if mask[dy + 1, dx + 1] == 1:
                        values.append(image[y + dy, x + dx])
            
            # Вычисляем медиану (ранг 3/5)
            if len(values) >= 3:
                filtered_image[y, x] = np.median(values[:3])
            else:
                filtered_image[y, x] = image[y, x]

    return filtered_image

def create_difference_image(original: np.array, filtered: np.array) -> np.array:
    """Создает разностное изображение как модуль разности."""
    return np.abs(original - filtered)

def generate_report(image_filenames: list) -> None:
    """Генерирует отчет в формате Markdown."""
    # Формирование текста отчета
    report_text = """
# Лабораторная работа №3: Фильтрация изображений и морфологические операции

## Задание 1: Медианный фильтр
К каждому изображению был применен медианный фильтр с разреженной маской (косой крест) и рангом 3/5.

## Задание 2: Разностное изображение
Для каждого изображения было вычислено разностное изображение (модуль разности между исходным и отфильтрованным изображением).

"""

    for filename in image_filenames:
        report_text += f"""
### Изображение: {filename}

#### Исходное изображение
![Исходное изображение]({filename})

#### Отфильтрованное изображение
![Отфильтрованное изображение](Filtered_{filename.replace('.png', '.bmp')})

#### Разностное изображение
![Разностное изображение](Difference_{filename.replace('.png', '.bmp')})
"""

    report_text += """
## Выводы
1. Медианный фильтр успешно применен ко всем изображениям. Он заменил значения пикселей на медиану в окрестности, заданной разреженной маской.
2. Разностное изображение показывает, какие пиксели были изменены фильтром.
3. Результаты работы сохранены в соответствующих файлах.
"""

    # Сохранение отчета в файл
    with open('report_lab3.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Проверка текущей рабочей директории
    print("Текущая рабочая директория:", os.getcwd())

    # Список файлов для обработки
    image_filenames = ['first_image.png', 'second_image.png', 'third_image.png', 'fourth_bin.png']

    # Разреженная маска (косой крест)
    mask = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    # Обработка каждого изображения
    for filename in image_filenames:
        print(f"Обработка файла: {filename}")
        image = load_image(filename)

        # Если изображение цветное (3 канала), преобразуем его в полутоновое
        if len(image.shape) == 3:
            print("Преобразование цветного изображения в полутоновое...")
            image = rgb_to_grayscale(image)

        # Применение медианного фильтра
        filtered_image = apply_median_filter(image, mask)
        save_image(filtered_image, f'Filtered_{filename.replace(".png", ".bmp")}')

        # Создание разностного изображения
        difference_image = create_difference_image(image, filtered_image)
        save_image(difference_image, f'Difference_{filename.replace(".png", ".bmp")}')

    # Генерация отчета
    generate_report(image_filenames)

    print("Лабораторная работа выполнена. Полученные файлы с изображениями:")
    for filename in image_filenames:
        print(f"Filtered_{filename.replace('.png', '.bmp')}, Difference_{filename.replace('.png', '.bmp')}")
    print("Отчет сохранен в файле report_lab3.md")

if __name__ == "__main__":
    main()