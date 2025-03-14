import os
import numpy as np
from PIL import Image

# Изменение рабочей директории на ту, где находится скрипт
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_image(filename: str) -> Image.Image:
    """Загружает изображение и конвертирует его в RGB."""
    print(f"Пытаюсь загрузить файл: {os.path.abspath(filename)}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")
    image = Image.open(filename)
    print(f"Изображение загружено. Размер: {image.size}, Формат: {image.format}")
    return image.convert('RGB')

def save_image(image_array: np.array, filename: str) -> None:
    """Сохраняет массив numpy как изображение в формате BMP."""
    img = Image.fromarray(np.uint8(image_array))
    img.save(filename)
    print(f"Изображение сохранено: {filename}")

def rgb_to_grayscale(image: Image.Image) -> np.array:
    """Преобразует RGB изображение в полутоновое (grayscale)."""
    img_array = np.array(image)
    # Взвешенное усреднение каналов (стандартные веса для преобразования в grayscale)
    grayscale_array = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
    return grayscale_array

def otsu_threshold(grayscale_array: np.array) -> float:
    """
    Вычисляет оптимальный порог для бинаризации методом Отсу.
    :param grayscale_array: Полутоновое изображение в виде массива numpy.
    :return: Оптимальный порог.
    """
    # Проверка на минимальное количество пикселей
    if grayscale_array.size < 2:
        print("Окно слишком маленькое, возвращаем порог 128")
        return 128  # Возвращаем средний порог, если окно слишком маленькое

    # Проверка типа данных
    if grayscale_array.dtype != np.uint8:
        grayscale_array = grayscale_array.astype(np.uint8)

    # Вычисление гистограммы
    hist, _ = np.histogram(grayscale_array, bins=256, range=(0, 256))
    
    # Проверка суммы гистограммы
    hist_sum = hist.sum()
    if hist_sum == 0:
        print("Гистограмма пустая, возвращаем порог 128")
        return 128  # Возвращаем средний порог, если гистограмма пустая

    # Нормализация гистограммы
    hist = hist.astype(np.float32) / hist_sum
    
    # Вычисление кумулятивных сумм и средних значений
    cum_sum = hist.cumsum()
    cum_mean = (np.arange(256) * hist).cumsum()
    
    # Вычисление глобального среднего
    global_mean = cum_mean[-1]
    
    # Вычисление межклассовой дисперсии для каждого порога
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(256):
        if cum_sum[t] == 0 or cum_sum[t] == 1:
            continue
        # Вероятности классов
        w0 = cum_sum[t]
        w1 = 1 - w0
        # Проверка на нулевое значение w0
        if w0 == 0:
            continue
        # Средние значения классов
        mean0 = cum_mean[t] / w0
        mean1 = (global_mean - cum_mean[t]) / w1
        # Межклассовая дисперсия
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t
    
    print(f"Оптимальный порог: {optimal_threshold}")
    return optimal_threshold

def adaptive_binarization_eikvil(grayscale_array: np.array, small_window_size: int = 3, large_window_size: int = 15, epsilon: int = 15) -> np.array:
    """
    Применяет адаптивную бинаризацию Эйквила к полутоновому изображению.
    :param grayscale_array: Полутоновое изображение в виде массива numpy.
    :param small_window_size: Размер маленького окна (например, 3x3).
    :param large_window_size: Размер большого окна (например, 15x15).
    :param epsilon: Порог для разницы математических ожиданий.
    :return: Бинаризованное изображение.
    """
    height, width = grayscale_array.shape
    binarized_array = np.zeros_like(grayscale_array)
    
    # Половины размеров окон
    small_half = small_window_size // 2
    large_half = large_window_size // 2
    
    print(f"Начало обработки изображения. Размер: {height}x{width}")
    
    for y in range(large_half, height - large_half):
        for x in range(large_half, width - large_half):
            # Определение области большого окна R
            y1 = max(y - large_half, 0)
            y2 = min(y + large_half + 1, height)
            x1 = max(x - large_half, 0)
            x2 = min(x + large_half + 1, width)
            large_window = grayscale_array[y1:y2, x1:x2]
            
            # Проверка на минимальный размер окна
            if large_window.size < 2:
                continue  # Пропускаем, если окно слишком маленькое
            
            # Вычисление оптимального порога T для большого окна
            T = otsu_threshold(large_window)
            
            # Разделение пикселей на два кластера
            cluster0 = large_window[large_window < T]
            cluster1 = large_window[large_window >= T]
            
            # Вычисление математических ожиданий
            M0 = cluster0.mean() if cluster0.size > 0 else 0
            M1 = cluster1.mean() if cluster1.size > 0 else 0
            
            # Проверка условия
            if abs(M0 - M1) >= epsilon:
                # Бинаризация пикселей в маленьком окне r
                y1_small = max(y - small_half, 0)
                y2_small = min(y + small_half + 1, height)
                x1_small = max(x - small_half, 0)
                x2_small = min(x + small_half + 1, width)
                small_window = grayscale_array[y1_small:y2_small, x1_small:x2_small]
                binarized_array[y1_small:y2_small, x1_small:x2_small] = (small_window > T) * 255
            else:
                # Замена яркости на яркость ближайшего класса
                y1_small = max(y - small_half, 0)
                y2_small = min(y + small_half + 1, height)
                x1_small = max(x - small_half, 0)
                x2_small = min(x + small_half + 1, width)
                small_window = grayscale_array[y1_small:y2_small, x1_small:x2_small]
                binarized_array[y1_small:y2_small, x1_small:x2_small] = np.where(
                    abs(small_window - M0) < abs(small_window - M1), M0, M1
                )
    
    print("Обработка изображения завершена.")
    return binarized_array

def generate_report(image_filenames: list) -> None:
    """Генерирует отчет в формате Markdown."""
    # Формирование текста отчета
    report_text = """
# Лабораторная работа №2: Обесцвечивание и бинаризация растровых изображений

## Задание 1: Приведение полноцветного изображения к полутоновому
Каждое изображение было преобразовано в полутоновое с использованием взвешенного усреднения каналов (R, G, B).

## Задание 2: Адаптивная бинаризация Эйквила
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
    print("Отчет сохранен в файле report_lab2.md")

def main():
    # Проверка текущей рабочей директории
    print("Текущая рабочая директория:", os.getcwd())

    # Список файлов для обработки
    image_filenames = ['fingerprint.png', 'map.png', 'photo.png', 'screenshot.png', 'text.png', 'x_ray.png']

    # Обработка каждого изображения
    for filename in image_filenames:
        print(f"\nОбработка файла: {filename}")
        try:
            image = load_image(filename)

            # 1. Приведение к полутоновому изображению
            grayscale_array = rgb_to_grayscale(image)
            save_image(grayscale_array, f'Output_Grayscale_{filename}.bmp')

            # 2. Адаптивная бинаризация Эйквила
            binarized_array = adaptive_binarization_eikvil(grayscale_array)
            save_image(binarized_array, f'Output_Binarized_{filename}.bmp')
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

    # Генерация отчета
    generate_report(image_filenames)

    print("\nЛабораторная работа выполнена. Полученные файлы с изображениями:")
    for filename in image_filenames:
        print(f"Output_Grayscale_{filename}.bmp, Output_Binarized_{filename}.bmp")
    print("Отчет сохранен в файле report_lab2.md")

if __name__ == "__main__":
    main()
