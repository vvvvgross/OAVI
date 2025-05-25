import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import csv

# Изменение рабочей директории на ту, где находится скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Шрифт и размер символов (используем курсивный шрифт)
FONT_PATH = "timesi.ttf"  # Укажите путь к шрифту Times New Roman Italic
FONT_SIZE = 52
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Алфавит для варианта 26 (русские строчные курсивные буквы)
ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

def generate_character_images():
    """Генерирует изображения символов и сохраняет их в папку."""
    if not os.path.exists("characters"):
        os.makedirs("characters")
    
    for char in ALPHABET:
        # Создаем изображение с символом
        image = Image.new("L", (FONT_SIZE, FONT_SIZE), 255)  # Белый фон
        draw = ImageDraw.Draw(image)
        draw.text((5, 5), char, font=FONT, fill=0)  # Черный текст
        
        # Обрезаем белые поля
        bbox = image.getbbox()
        cropped_image = image.crop(bbox)
        
        # Сохраняем изображение
        cropped_image.save(f"characters/{char}.png")

def calculate_features(image: np.array) -> dict:
    """Вычисляет признаки для изображения символа."""
    height, width = image.shape
    total_pixels = height * width
    
    # Вес (масса чёрного) каждой четверти изображения
    quarter_height = height // 2
    quarter_width = width // 2
    
    quarters = [
        image[:quarter_height, :quarter_width],
        image[:quarter_height, quarter_width:],
        image[quarter_height:, :quarter_width],
        image[quarter_height:, quarter_width:]
    ]
    
    weights = [np.sum(quarter == 0) for quarter in quarters]
    normalized_weights = [weight / (quarter_height * quarter_width) for weight in weights]
    
    # Координаты центра тяжести
    y_indices, x_indices = np.where(image == 0)
    center_x = np.mean(x_indices) if len(x_indices) > 0 else 0
    center_y = np.mean(y_indices) if len(y_indices) > 0 else 0
    
    # Нормированные координаты центра тяжести
    normalized_center_x = center_x / width
    normalized_center_y = center_y / height
    
    # Осевые моменты инерции
    moment_x = np.sum((y_indices - center_y) ** 2) if len(y_indices) > 0 else 0
    moment_y = np.sum((x_indices - center_x) ** 2) if len(x_indices) > 0 else 0
    
    # Нормированные осевые моменты инерции
    normalized_moment_x = moment_x / (height * width)
    normalized_moment_y = moment_y / (height * width)
    
    # Профили X и Y
    profile_x = np.sum(image == 0, axis=0)
    profile_y = np.sum(image == 0, axis=1)
    
    return {
        "weights": weights,
        "normalized_weights": normalized_weights,
        "center_x": center_x,
        "center_y": center_y,
        "normalized_center_x": normalized_center_x,
        "normalized_center_y": normalized_center_y,
        "moment_x": moment_x,
        "moment_y": moment_y,
        "normalized_moment_x": normalized_moment_x,
        "normalized_moment_y": normalized_moment_y,
        "profile_x": profile_x,
        "profile_y": profile_y
    }

def save_features_to_csv(features: dict, filename: str):
    """Сохраняет признаки в CSV-файл."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Символ", "Вес 1", "Вес 2", "Вес 3", "Вес 4",
                         "Удельный вес 1", "Удельный вес 2", "Удельный вес 3", "Удельный вес 4",
                         "Центр X", "Центр Y", "Норм. центр X", "Норм. центр Y",
                         "Момент X", "Момент Y", "Норм. момент X", "Норм. момент Y"])
        
        for char, feature in features.items():
            writer.writerow([
                char,
                *feature["weights"],
                *feature["normalized_weights"],
                feature["center_x"], feature["center_y"],
                feature["normalized_center_x"], feature["normalized_center_y"],
                feature["moment_x"], feature["moment_y"],
                feature["normalized_moment_x"], feature["normalized_moment_y"]
            ])

def save_profiles(features: dict):
    """Сохраняет профили X и Y в виде столбчатых диаграмм."""
    if not os.path.exists("profiles"):
        os.makedirs("profiles")
    
    for char, feature in features.items():
        # Профиль X
        plt.bar(range(len(feature["profile_x"])), feature["profile_x"])
        plt.title(f"Профиль X для символа {char}")
        plt.xlabel("X")
        plt.ylabel("Количество черных пикселей")
        plt.savefig(f"profiles/{char}_profile_x.png")
        plt.close()
        
        # Профиль Y
        plt.bar(range(len(feature["profile_y"])), feature["profile_y"])
        plt.title(f"Профиль Y для символа {char}")
        plt.xlabel("Y")
        plt.ylabel("Количество черных пикселей")
        plt.savefig(f"profiles/{char}_profile_y.png")
        plt.close()

def generate_report(features: dict):
    """Генерирует отчет в формате Markdown."""
    report_text = """
# Лабораторная работа №5: Выделение признаков символов

## Задание 1: Генерация эталонных изображений символов
Были сгенерированы изображения символов русского алфавита (строчные курсивные буквы).

## Задание 2: Вычисление признаков
Для каждого символа были вычислены следующие признаки:
- Вес (масса чёрного) каждой четверти изображения.
- Удельный вес (вес, нормированный к четверти площади).
- Координаты центра тяжести.
- Нормированные координаты центра тяжести.
- Осевые моменты инерции по горизонтали и вертикали.
- Нормированные осевые моменты инерции.
- Профили X и Y.

## Результаты
"""

    for char, feature in features.items():
        report_text += f"""
### Символ: {char}

#### Изображение символа
![Изображение символа](characters/{char}.png)

#### Профиль X
![Профиль X](profiles/{char}_profile_x.png)

#### Профиль Y
![Профиль Y](profiles/{char}_profile_y.png)
"""

    report_text += """
## Выводы
1. Все символы были успешно сгенерированы и сохранены в папку `characters`.
2. Для каждого символа были вычислены признаки и сохранены в файл `features.csv`.
3. Профили X и Y были сохранены в виде столбчатых диаграмм в папку `profiles`.
"""

    # Сохранение отчета в файл
    with open('report_lab5.md', 'w', encoding='utf-8') as report_file:
        report_file.write(report_text)

def main():
    # Генерация изображений символов
    print("Генерация изображений символов...")
    generate_character_images()
    
    # Вычисление признаков для каждого символа
    features = {}
    for char in ALPHABET:
        image = np.array(Image.open(f"characters/{char}.png").convert("L"))
        features[char] = calculate_features(image)
    
    # Сохранение признаков в CSV-файл
    print("Сохранение признаков в CSV-файл...")
    save_features_to_csv(features, "features.csv")
    
    # Сохранение профилей X и Y
    print("Сохранение профилей X и Y...")
    save_profiles(features)
    
    # Генерация отчета
    print("Генерация отчета...")
    generate_report(features)
    
    print("Лабораторная работа выполнена. Отчет сохранен в файле report_lab5.md")

if __name__ == "__main__":
    main()