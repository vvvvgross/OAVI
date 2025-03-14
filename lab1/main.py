import os
from PIL import Image
import numpy as np
import math

# Изменение рабочей директории на ту, где находится скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_image(filename: str) -> Image.Image:
    """Загружает изображение и конвертирует его в RGB."""
    print("Пытаюсь загрузить файл:", os.path.abspath(filename))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")
    image = Image.open(filename)
    return image.convert('RGB')

def image_to_np_array(image: Image.Image) -> np.array:
    """Преобразует изображение в массив numpy."""
    return np.array(image)

def save_image(image_array: np.array, filename: str) -> None:
    """Сохраняет массив numpy как изображение."""
    img = Image.fromarray(np.uint8(image_array))
    img.save(filename)

def extract_rgb_components(image: Image.Image) -> None:
    """Извлекает и сохраняет R, G, B компоненты изображения."""
    img_array = image_to_np_array(image)
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    save_image(np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2), 'Output_R.png')
    save_image(np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2), 'Output_G.png')
    save_image(np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2), 'Output_B.png')

def rgb_to_hsi(image: Image.Image) -> tuple:
    """Преобразует изображение из RGB в HSI."""
    img_array = np.array(image).astype(np.float32) / 255.0
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-10))

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.zeros_like(I)
    H[B > G] = (2 * np.pi - theta[B > G])
    H[B <= G] = theta[B <= G]
    H = H / (2 * np.pi)

    return H, S, I

def stretch_image(image: Image.Image, M: int) -> Image.Image:
    """Растягивает изображение в M раз."""
    src_width, src_height = image.size
    new_width, new_height = src_width * M, src_height * M
    new_image = Image.new(image.mode, (new_width, new_height))

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x, orig_y = new_x // M, new_y // M
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

def compress_image(image: Image.Image, N: int) -> Image.Image:
    """Сжимает изображение в N раз."""
    src_width, src_height = image.size
    new_width, new_height = src_width // N, src_height // N
    new_image = Image.new(image.mode, (new_width, new_height))

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x, orig_y = new_x * N, new_y * N
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

def two_pass_resample(image: Image.Image, M: int, N: int) -> Image.Image:
    """Передискретизация в два прохода: растяжение + сжатие."""
    stretched = stretch_image(image, M)
    return compress_image(stretched, N)

def one_pass_resample(image: Image.Image, K: float) -> Image.Image:
    """Передискретизация в один проход."""
    src_width, src_height = image.size
    new_width, new_height = int(src_width / K), int(src_height / K)
    new_image = Image.new(image.mode, (new_width, new_height))

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x, orig_y = int(new_x * K), int(new_y * K)
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

def generate_report(
    image: Image.Image,
    M: int,
    N: int,
    K: float,
    stretched_img: Image.Image,
    compressed_img: Image.Image,
    two_pass_img: Image.Image,
    one_pass_img: Image.Image,
) -> None:
    """Генерирует отчет в формате Markdown."""
    # Основные данные для отчета
    report_data = {
        "image_size": image.size,
        "M": M,
        "N": N,
        "K": K,
        "stretched_img_size": stretched_img.size,
        "compressed_img_size": compressed_img.size,
        "two_pass_img_size": two_pass_img.size,
        "one_pass_img_size": one_pass_img.size,
    }

    # Формирование текста отчета
    report_text = f"""
# Отчет по лабораторной работе

## 1. Выделение компонент R, G, B
Были разделены каналы изображения. Ниже представлены результаты:
- **Канал R:** ![](Output_R.png)
- **Канал G:** ![](Output_G.png)
- **Канал B:** ![](Output_B.png)

## 2. Преобразование изображения в HSI
Ниже представлены компоненты:
- **Яркостная компонента (I):** ![](Output_Intensity.png)
- **Инвертированная яркостная компонента:** ![](Output_Inverted_Intensity.png)

## 3. Изменение размера изображения

### 3.1 Изначальные размеры изображения
- **Размер изображения:** {report_data["image_size"]}

### 3.2 Растяжение (интерполяция)
- **Коэффициент растяжения (M):** {report_data["M"]}
- **Результат растяжения:** ![](Output_Stretched.png)
- **Размер изображения после растяжения:** {report_data["stretched_img_size"]}

### 3.3 Сжатие (децимация)
- **Коэффициент сжатия (N):** {report_data["N"]}
- **Результат сжатия:** ![](Output_Compressed.png)
- **Размер изображения после сжатия:** {report_data["compressed_img_size"]}

### 3.4 Передискретизация в два прохода (растяжение + сжатие)
- **Коэффициент (K = M / N):** {report_data["K"]}
- **Результат передискретизации в два прохода:** ![](Output_TwoPass_Resampled.png)
- **Размер изображения после передискретизации:** {report_data["two_pass_img_size"]}

### 3.5 Передискретизация в один проход
- **Коэффициент (K):** {report_data["K"]}
- **Результат передискретизации в один проход:** ![](Output_OnePass_Resampled.png)
- **Размер изображения после передискретизации:** {report_data["one_pass_img_size"]}
"""

    # Сохранение отчета в файл
    with open('report_lab1.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Проверка текущей рабочей директории
    print("Текущая рабочая директория:", os.getcwd())

    input_filename = 'image.png'
    image = load_image(input_filename)

    # Извлечение RGB компонент
    extract_rgb_components(image)

    # Преобразование в HSI
    H, S, I = rgb_to_hsi(image)
    save_image((I * 255).astype(np.uint8), 'Output_Intensity.png')
    save_image(((1 - I) * 255).astype(np.uint8), 'Output_Inverted_Intensity.png')

    # Изменение размера изображения
    M, N = 3, 2
    K = M / N

    stretched_img = stretch_image(image, M)
    stretched_img.save('Output_Stretched.png')

    compressed_img = compress_image(image, N)
    compressed_img.save('Output_Compressed.png')

    two_pass_img = two_pass_resample(image, M, N)
    two_pass_img.save('Output_TwoPass_Resampled.png')

    one_pass_img = one_pass_resample(image, K)
    one_pass_img.save('Output_OnePass_Resampled.png')

    # Генерация отчета
    generate_report(image, M, N, K, stretched_img, compressed_img, two_pass_img, one_pass_img)

    print("Лабораторная работа выполнена. Полученные файлы с изображениями:")
    print("Output_R.png, Output_G.png, Output_B.png, Output_Intensity.png, Output_Inverted_Intensity.png")
    print("Output_Stretched.png, Output_Compressed.png, Output_TwoPass_Resampled.png, Output_OnePass_Resampled.png")
    print("Отчет сохранен в файле report_lab1.md")
    print(f'M = {M}, N = {N}, K = {K}')

if __name__ == "__main__":
    main()