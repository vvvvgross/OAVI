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

def compute_gradients(grayscale_array: np.array) -> tuple:
    """Вычисляет градиенты Gx и Gy с использованием оператора Шарра."""
    height, width = grayscale_array.shape
    Gx = np.zeros_like(grayscale_array, dtype=np.float32)
    Gy = np.zeros_like(grayscale_array, dtype=np.float32)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Вычисление Gx
            Gx[y, x] = (3 * grayscale_array[y-1, x-1] + 10 * grayscale_array[y, x-1] + 3 * grayscale_array[y+1, x-1]) - \
                       (3 * grayscale_array[y-1, x+1] + 10 * grayscale_array[y, x+1] + 3 * grayscale_array[y+1, x+1])
            
            # Вычисление Gy
            Gy[y, x] = (3 * grayscale_array[y-1, x-1] + 10 * grayscale_array[y-1, x] + 3 * grayscale_array[y-1, x+1]) - \
                       (3 * grayscale_array[y+1, x-1] + 10 * grayscale_array[y+1, x] + 3 * grayscale_array[y+1, x+1])
    
    return Gx, Gy

def compute_gradient_magnitude(Gx: np.array, Gy: np.array) -> np.array:
    """Вычисляет итоговую градиентную матрицу G."""
    G = np.abs(Gx) + np.abs(Gy)
    return G

def normalize_gradient(G: np.array) -> np.array:
    """Нормализует градиентную матрицу G в диапазон от 0 до 255."""
    G_normalized = 255 * (G - np.min(G)) / (np.max(G) - np.min(G))
    return G_normalized.astype(np.uint8)

def binarize_gradient(G: np.array, threshold: int) -> np.array:
    """Бинаризует градиентную матрицу G с использованием порога."""
    G_binarized = np.where(G > threshold, 255, 0)
    return G_binarized.astype(np.uint8)

def generate_report(image_filenames: list) -> None:
    """Генерирует отчет в формате Markdown."""
    # Формирование текста отчета
    report_text = """
# Лабораторная работа №4: Выделение контуров на изображении

## Задание 1: Приведение полноцветного изображения к полутоновому
Каждое изображение было преобразовано в полутоновое с использованием взвешенного усреднения каналов (R, G, B).

## Задание 2: Вычисление градиентов с использованием оператора Шарра
К каждому полутоновому изображению были применены операторы Шарра для вычисления градиентов Gx и Gy.

## Задание 3: Нормализация и бинаризация градиентной матрицы
Градиентная матрица G была нормализована и бинаризована с использованием порога.

"""

    for filename in image_filenames:
        report_text += f"""
### Изображение: {filename}

#### Исходное изображение
![Исходное изображение]({filename})

#### Полутоновое изображение
![Полутоновое изображение](Grayscale_{filename.replace('.png', '.bmp')})

#### Градиентная матрица G
![Градиентная матрица G](Gradient_{filename.replace('.png', '.bmp')})

#### Бинаризованная градиентная матрица G
![Бинаризованная градиентная матрица G](Binarized_{filename.replace('.png', '.bmp')})
"""

    report_text += """
## Выводы
1. Все изображения успешно преобразованы в полутоновые с использованием взвешенного усреднения каналов (R, G, B).
2. К полутоновым изображениям применены операторы Шарра для вычисления градиентов Gx и Gy.
3. Градиентная матрица G была нормализована и бинаризована.
4. Результаты работы сохранены в соответствующих файлах.
"""

    # Сохранение отчета в файл
    with open('report_lab4.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    
    print("Текущая рабочая директория:", os.getcwd())

    # Список файлов для обработки
    image_filenames = ['fingerprint.png', 'map.png', 'photo.png', 'screenshot.png', 'text.png', 'x_ray.png']

    # Обработка каждого изображения
    for filename in image_filenames:
        print(f"Обработка файла: {filename}")
        try:
            # Загрузка изображения
            image = load_image(filename)

            # Если изображение цветное (3 канала), преобразуем его в полутоновое
            if len(image.shape) == 3:
                print("Преобразование цветного изображения в полутоновое...")
                image = rgb_to_grayscale(image)

            # Сохранение полутонового изображения
            save_image(image, f'Grayscale_{filename.replace(".png", ".bmp")}')

            # Вычисление градиентов
            Gx, Gy = compute_gradients(image)
            
            # Вычисление итоговой градиентной матрицы
            G = compute_gradient_magnitude(Gx, Gy)
            
            # Нормализация градиентной матрицы
            G_normalized = normalize_gradient(G)
            save_image(G_normalized, f'Gradient_{filename.replace(".png", ".bmp")}')
            
            # Бинаризация градиентной матрицы
            threshold = 50  
            G_binarized = binarize_gradient(G_normalized, threshold)
            save_image(G_binarized, f'Binarized_{filename.replace(".png", ".bmp")}')
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

    # Генерация отчета
    generate_report(image_filenames)

    print("Лабораторная работа выполнена. Полученные файлы с изображениями:")
    for filename in image_filenames:
        print(f"Grayscale_{filename.replace('.png', '.bmp')}, Gradient_{filename.replace('.png', '.bmp')}, Binarized_{filename.replace('.png', '.bmp')}")
    print("Отчет сохранен в файле report_lab4.md")

if __name__ == "__main__":
    main()