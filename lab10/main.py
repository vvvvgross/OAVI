import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
from scipy.signal.windows import hann
import os
from typing import Tuple, List


class VoiceAnalyzer:
    def __init__(self):
        os.makedirs("output", exist_ok=True)
        self.sample_files = {
            'a': 'a.wav',
            'i': 'i.wav',
            'bark': 'gav.wav'
        }

    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        rate, data = wavfile.read(filename)
        if data.ndim > 1:
            data = data[:, 0]
        data = data / np.max(np.abs(data))  # нормализация
        return data, rate

    def compute_spectrogram(self, data: np.ndarray, rate: int, filename: str, formants: List[float] = None) -> None:
        nperseg = 1024
        noverlap = nperseg // 2
        window = hann(nperseg)

        f, t, Sxx = spectrogram(data, rate, window=window,
                                nperseg=nperseg, noverlap=noverlap,
                                scaling='spectrum')

        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='inferno')
        plt.ylim(0, 5000)
        plt.ylabel('Частота [Гц]')
        plt.xlabel('Время [сек]')
        plt.title(f'Спектрограмма: {filename}')
        plt.colorbar(label='Интенсивность [дБ]')

        # Отметим форманты горизонтальными линиями
        if formants:
            for freq in formants:
                if freq > 0:
                    plt.axhline(freq, color='cyan', linestyle='--', linewidth=1.5, label=f'{freq:.1f} Гц')

            # Чтобы не дублировалась легенда
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.savefig(f'output/spectrogram_{filename}.png')
        plt.close()

        return f, t, Sxx

    def find_main_tone(self, f: np.ndarray, Sxx: np.ndarray) -> float:
        avg_spectrum = np.mean(Sxx, axis=1)
        peaks, _ = find_peaks(avg_spectrum, height=0.1 * np.max(avg_spectrum))

        if len(peaks) == 0:
            return 0.0

        main_tone = f[peaks[0]]
        max_harmonics = 0

        for peak in peaks:
            freq = f[peak]
            harmonics = 0
            for h in range(2, 6):
                harmonic_freq = freq * h
                idx = np.argmin(np.abs(f - harmonic_freq))
                if avg_spectrum[idx] > 0.1 * avg_spectrum[peak]:
                    harmonics += 1
            if harmonics > max_harmonics:
                max_harmonics = harmonics
                main_tone = freq

        return main_tone

    def find_formants(self, f: np.ndarray, Sxx: np.ndarray, n_formants=3) -> List[float]:
        """Находит n_formants формант из усредненного спектра"""
        avg_spectrum = np.mean(Sxx, axis=1)
        peaks, _ = find_peaks(avg_spectrum, height=0.1 * np.max(avg_spectrum), distance=5)
        sorted_peaks = sorted(peaks, key=lambda i: avg_spectrum[i], reverse=True)
        formants = [f[i] for i in sorted_peaks[:n_formants]]
        return formants + [0.0] * (n_formants - len(formants))

    def analyze_audio(self, filename: str) -> dict:
        data, rate = self.load_audio(filename)
        f, t, Sxx = self.compute_spectrogram(data, rate, filename)

        threshold = 0.01 * np.max(Sxx)
        nonzero_freqs = f[np.any(Sxx > threshold, axis=1)]
        min_freq = np.min(nonzero_freqs) if len(nonzero_freqs) > 0 else 0
        max_freq = np.max(nonzero_freqs) if len(nonzero_freqs) > 0 else 0

        main_tone = self.find_main_tone(f, Sxx)
        formants = self.find_formants(f, Sxx)

        # Перерисовать спектрограмму с формантами
        self.compute_spectrogram(data, rate, filename, formants)

        return {
            'filename': filename,
            'duration': len(data) / rate,
            'sample_rate': rate,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'main_tone': main_tone,
            'formants': formants,
            'spectrogram': f'output/spectrogram_{filename.split(".")[0]}.png'
        }

    def generate_report(self, results: dict):
        report = "# Лабораторная работа №10: Обработка голоса\n\n## Результаты анализа\n"
        for label, res in results.items():
            report += f"""
### {label.upper()} ({res['filename']})
- Длительность: {res['duration']:.2f} сек
- Частотный диапазон: {res['min_freq']:.1f} – {res['max_freq']:.1f} Гц
- Основной тон: {res['main_tone']:.1f} Гц
- Форманты: {', '.join(f"{x:.1f}" for x in res['formants'])} Гц
- ![Спектрограмма]({res['spectrogram']})
"""

        with open("output/report_lab10.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("Отчет сохранен в output/report_lab10.md")

    def run_analysis(self):
        results = {}
        for label, file in self.sample_files.items():
            try:
                print(f"Анализируем {file}...")
                results[label] = self.analyze_audio(file)
            except Exception as e:
                print(f"Ошибка при обработке {file}: {e}")

        self.generate_report(results)


if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    analyzer.run_analysis()
