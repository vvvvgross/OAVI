import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import librosa
import soundfile as sf
from pydub import AudioSegment
import os

# Установка рабочей директории
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class AudioAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.audio = None
        self.sample_rate = None
        self.duration = None
        self.spectrogram = None
        self.times = None
        self.frequencies = None
        self.filtered_spectrogram = None
        
    def load_audio(self):
        """Загрузка аудиофайла и преобразование в моно"""
        try:
            # Для WAV файлов
            self.sample_rate, self.audio = wavfile.read(self.filename)
            if len(self.audio.shape) > 1:
                self.audio = self.audio.mean(axis=1)  # Преобразование стерео в моно
        except:
            # Для других форматов используем librosa
            self.audio, self.sample_rate = librosa.load(self.filename, sr=None, mono=True)
        
        # Нормализация аудио
        self.audio = self.audio / np.max(np.abs(self.audio))
        self.duration = len(self.audio) / self.sample_rate
        
    def plot_waveform(self):
        """Построение графика звуковой волны"""
        plt.figure(figsize=(12, 4))
        time = np.linspace(0, self.duration, num=len(self.audio))
        plt.plot(time, self.audio)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.savefig('waveform.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_spectrogram(self):
        """Вычисление спектрограммы с окном Ханна"""
        nperseg = 1024  # Длина сегмента для FFT
        noverlap = nperseg // 2  # Перекрытие сегментов
        
        self.frequencies, self.times, self.spectrogram = signal.spectrogram(
            self.audio,
            fs=self.sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        
    def plot_spectrogram(self, spectrogram, title, filename):
        """Построение спектрограммы с логарифмической шкалой частот"""
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(self.times, self.frequencies, 10 * np.log10(spectrogram + 1e-10),
                       shading='gouraud', cmap='viridis')
        plt.yscale('symlog', linthresh=100)  # Логарифмическая шкала с линейной частью для низких частот
        plt.ylim(20, self.sample_rate/2)  # Ограничение по частотам
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Intensity (dB)')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def estimate_noise(self):
        """Оценка уровня шума с использованием медианного фильтра"""
        # Оценка шума как медианного значения по частотам
        noise_profile = np.median(self.spectrogram, axis=1)
        return noise_profile
        
    def apply_wiener_filter(self):
        """Применение фильтра Винера для удаления шума"""
        noise_profile = self.estimate_noise()
        
        # Создаем маску для фильтрации
        spectral_mask = 1 - (noise_profile[:, np.newaxis] / (self.spectrogram + 1e-10))
        spectral_mask = np.clip(spectral_mask, 0, 1)
        
        # Применяем маску к спектрограмме
        self.filtered_spectrogram = self.spectrogram * spectral_mask
        
        # Восстановление сигнала
        _, reconstructed_audio = signal.istft(
            np.sqrt(self.filtered_spectrogram) * np.exp(1j * np.zeros_like(self.filtered_spectrogram)),
            fs=self.sample_rate,
            window='hann',
            nperseg=1024,
            noverlap=512
        )
        
        # Обрезка по длине оригинального сигнала и нормализация
        reconstructed_audio = reconstructed_audio[:len(self.audio)]
        reconstructed_audio = reconstructed_audio / np.max(np.abs(reconstructed_audio))
        return reconstructed_audio
        
    def find_peaks(self, delta_t=0.1, freq_range=(40, 50)):
        """Нахождение моментов с максимальной энергией"""
        # Определение индексов частотного диапазона
        freq_idx = (self.frequencies >= freq_range[0]) & (self.frequencies <= freq_range[1])
        
        # Вычисление энергии в заданном частотном диапазоне
        energy_in_band = np.sum(self.spectrogram[freq_idx, :], axis=0)
        
        # Нахождение пиков с заданным временным интервалом
        min_samples = int(delta_t * len(self.times) / self.duration)
        peaks, _ = signal.find_peaks(energy_in_band, distance=min_samples)
        
        peak_times = self.times[peaks]
        peak_energies = energy_in_band[peaks]
        
        return peak_times, peak_energies
        
    def save_filtered_audio(self, filtered_audio, output_filename):
        """Сохранение отфильтрованного аудио"""
        sf.write(output_filename, filtered_audio, self.sample_rate)
        
    def generate_report(self, peak_times, peak_energies):
        """Генерация отчета в формате Markdown"""
        report = f"""
# Лабораторная работа №9: Анализ шума

## Задание 1: Запись и загрузка аудио
Анализируемый файл: `{self.filename}`
- Частота дискретизации: {self.sample_rate} Гц
- Длительность: {self.duration:.2f} секунд

![Waveform](waveform.png)

## Задание 2: Спектрограмма
Построена спектрограмма с использованием оконного преобразования Фурье (окно Ханна).

![Original Spectrogram](spectrogram_original.png)

## Задание 3: Фильтрация шума
Применен фильтр Винера для удаления шума. Результаты сохранены в файл `filtered_audio.wav`.

Спектрограмма после фильтрации:

![Filtered Spectrogram](spectrogram_filtered.png)

## Задание 4: Анализ энергии
Найдены моменты времени с максимальной энергией в диапазоне 40-50 Гц:

| Время (с) | Энергия |
|-----------|---------|
"""
        for t, e in zip(peak_times, peak_energies):
            report += f"| {t:.2f} | {e:.2e} |\n"

        report += """
## Выводы
1. Успешно построены графики звуковой волны и спектрограммы до и после фильтрации.
2. Реализована оценка уровня шума и применен фильтр Винера.
3. Обнаружены моменты времени с максимальной энергией в заданном частотном диапазоне.
4. Фильтрация позволила уменьшить уровень шума, что видно на спектрограмме.
"""

        with open('report_lab9.md', 'w', encoding='utf-8') as f:
            f.write(report)

def main():
    # Инициализация анализатора
    analyzer = AudioAnalyzer('input_audio.wav')  # Замените на ваш файл
    
    # Выполнение анализа
    print("Загрузка аудиофайла...")
    analyzer.load_audio()
    
    print("Построение графика волны...")
    analyzer.plot_waveform()
    
    print("Вычисление спектрограммы...")
    analyzer.compute_spectrogram()
    
    print("Построение оригинальной спектрограммы...")
    analyzer.plot_spectrogram(analyzer.spectrogram, 'Original Spectrogram', 'spectrogram_original.png')
    
    print("Применение фильтра Винера...")
    filtered_audio = analyzer.apply_wiener_filter()
    analyzer.save_filtered_audio(filtered_audio, 'filtered_audio.wav')
    
    print("Построение фильтрованной спектрограммы...")
    analyzer.plot_spectrogram(analyzer.filtered_spectrogram, 'Filtered Spectrogram', 'spectrogram_filtered.png')
    
    print("Поиск пиков энергии...")
    peak_times, peak_energies = analyzer.find_peaks()
    
    print("Генерация отчета...")
    analyzer.generate_report(peak_times, peak_energies)
    
    print("Лабораторная работа выполнена. Отчет сохранен в report_lab9.md")

if __name__ == "__main__":
    main()