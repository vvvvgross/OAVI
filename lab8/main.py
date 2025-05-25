import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.color import rgb2hsv, hsv2rgb
from skimage.transform import resize

# Установка рабочей директории
os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_IMAGES = ['map.png', 'photo.png', 'screenshot.png', 'text.png']
CELL_SIZE = (8, 8)
BLOCK_SIZE = (2, 2)
ORIENTATIONS = 9

def load_image(filename: str) -> np.array:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")
    return np.array(Image.open(filename))

def save_image(image: np.array, filename: str) -> None:
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    Image.fromarray(image).save(filename)

def rgb_to_grayscale(image: np.array) -> np.array:
    if len(image.shape) == 3:
        return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return image

def rgb_to_hsl(image: np.array) -> np.array:
    hsv = rgb2hsv(image / 255.0)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hsl = np.zeros_like(image, dtype=np.float32)
    hsl[:, :, 0] = h * 255
    hsl[:, :, 1] = v * 255
    hsl[:, :, 2] = s * 255
    return hsl

def hsl_to_rgb(hsl: np.array) -> np.array:
    h = hsl[:, :, 0] / 255
    v = hsl[:, :, 1] / 255
    s = hsl[:, :, 2] / 255
    hsv = np.stack([h, s, v], axis=2)
    rgb = hsv2rgb(hsv) * 255
    return rgb.astype(np.uint8)

def logarithmic_transform(image: np.array) -> np.array:
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image)
    return log_transformed.astype(np.uint8)

def compute_hog_features(image: np.array) -> tuple:
    fd, hog_image = hog(
        image,
        orientations=ORIENTATIONS,
        pixels_per_cell=CELL_SIZE,
        cells_per_block=BLOCK_SIZE,
        visualize=True,
        channel_axis=None
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd, hog_image

def compute_gradients(image: np.array) -> tuple:
    image = image.astype(np.int32)
    height, width = image.shape
    Gx = np.zeros_like(image, dtype=np.int32)
    Gy = np.zeros_like(image, dtype=np.int32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            Gx[y, x] = (3*image[y-1, x-1] + 10*image[y, x-1] + 3*image[y+1, x-1]) - \
                       (3*image[y-1, x+1] + 10*image[y, x+1] + 3*image[y+1, x+1])
            Gy[y, x] = (3*image[y-1, x-1] + 10*image[y-1, x] + 3*image[y-1, x+1]) - \
                       (3*image[y+1, x-1] + 10*image[y+1, x] + 3*image[y+1, x+1])

    return Gx, Gy

def plot_histogram(image: np.array, title: str, filename: str) -> None:
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel('Значение яркости')
    plt.ylabel('Частота')
    plt.savefig(filename)
    plt.close()

def generate_report():
    report_text = """
# Лабораторная работа №8: Текстурный анализ и контрастирование (Вариант 6)

## Задание
1. Использование HOG с контурным оператором из ЛР4
2. Логарифмическое преобразование яркости
3. Сравнение текстурных признаков до и после преобразования

## Результаты для каждого изображения
"""

    for filename in INPUT_IMAGES:
        report_text += f"""
### Изображение: {filename}

#### Исходное изображение
![Исходное изображение](output/original_{filename})

#### Полутоновое изображение
![Полутоновое изображение](output/grayscale_{filename})

#### Контрастированное изображение
![Контрастированное изображение](output/log_transformed_{filename})

#### Гистограмма исходного изображения
![Гистограмма исходного](output/hist_original_{filename.replace('.png', '.png')})

#### Гистограмма контрастированного изображения
![Гистограмма контрастированного](output/hist_log_{filename.replace('.png', '.png')})

#### Гистограмма яркости HOG визуализации исходного изображения
![Гистограмма HOG исходного](output/histogram_hog_original_{filename})

#### Гистограмма яркости HOG визуализации контрастированного изображения
![Гистограмма HOG контрастированного](output/histogram_hog_log_{filename})

#### HOG-визуализация исходного изображения
![HOG исходного](output/hog_original_{filename})

#### HOG-визуализация контрастированного изображения
![HOG контрастированного](output/hog_log_{filename})

#### Градиенты (оператор Шарра)
![Градиенты](output/gradients_{filename})
"""

    report_text += """
## Выводы
1. Логарифмическое преобразование эффективно улучшает контраст изображений
2. HOG-признаки позволяют анализировать текстуру изображения
3. Контурный оператор Шарра хорошо выделяет границы объектов
4. После контрастирования гистограммы яркости становятся более равномерными
5. Текстурные признаки становятся более выраженными после преобразования
"""

    with open('report_lab8.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def prepare_image_for_hog(image: np.array, max_size=512) -> np.array:
    if len(image.shape) == 3:
        image = rgb_to_grayscale(image)
    h, w = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = resize(image, (int(h*scale), int(w*scale)), anti_aliasing=True)
    image_uint8 = np.uint8(np.clip(image, 0, 255))
    return image_uint8

def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    for filename in INPUT_IMAGES:
        try:
            image = load_image(filename)
            save_image(image, f"output/original_{filename}")

            grayscale = rgb_to_grayscale(image)
            save_image(grayscale, f"output/grayscale_{filename}")

            if len(image.shape) == 3:
                hsl = rgb_to_hsl(image)
                L = hsl[:, :, 1]
                L_log = logarithmic_transform(L)
                hsl[:, :, 1] = L_log
                log_transformed = hsl_to_rgb(hsl)
            else:
                L_log = logarithmic_transform(grayscale)
                log_transformed = L_log

            save_image(log_transformed, f"output/log_transformed_{filename}")

            plot_histogram(grayscale, f"Гистограмма исходного: {filename}",
                           f"output/hist_original_{filename.replace('.png', '.png')}")
            plot_histogram(L_log, f"Гистограмма после преобразования: {filename}",
                           f"output/hist_log_{filename.replace('.png', '.png')}")

            gray_for_hog = prepare_image_for_hog(grayscale)
            _, hog_original = compute_hog_features(gray_for_hog)
            save_image(hog_original, f"output/hog_original_{filename}")

            log_for_hog = prepare_image_for_hog(L_log)
            _, hog_log = compute_hog_features(log_for_hog)
            save_image(hog_log, f"output/hog_log_{filename}")

            # Построение гистограмм яркости для HOG-визуализаций:
            plot_histogram(hog_original, f"Гистограмма яркости HOG визуализации исходного: {filename}",
                           f"output/histogram_hog_original_{filename}")
            plot_histogram(hog_log, f"Гистограмма яркости HOG визуализации контрастированного: {filename}",
                           f"output/histogram_hog_log_{filename}")

            Gx, Gy = compute_gradients(gray_for_hog)
            G = np.abs(Gx) + np.abs(Gy)
            G_normalized = (255 * (G - np.min(G)) / (np.max(G) - np.min(G))).astype(np.uint8)
            save_image(G_normalized, f"output/gradients_{filename}")

        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

    generate_report()
    print("Лабораторная работа выполнена. Отчет сохранен в файле report_lab8.md")

if __name__ == "__main__":
    main()
