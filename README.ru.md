# PulseVision

[English](README.md) | [Русский](README.ru.md)

PulseVision — открытая реализация на PyTorch архитектуры **EfficientPhys-C**, предложенной в научной статье [«EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement»](https://arxiv.org/abs/2110.04447) (Liu et al., 2021). Проект выполняет бесконтактное измерение частоты сердечных сокращений (rPPG — remote photoplethysmography) в реальном времени с обычной веб-камеры.

## Описание метода

Дистанционная фотоплетизмография фиксирует едва различимые изменения спектрального отражения кожи, вызванные пульсовой волной кровенаполнения сосудов. Алгоритм работы состоит из четырех основных этапов:

1. **Детекция лица и стабилизация области интереса (ROI):**  
   В течение стартовой калибровки на 300 кадрах (~10 секунд) с помощью MediaPipe FaceLandmarker вычисляются общие усредненные границы лица. Фиксация единой неподвижной рамки исключает дрожание пикселей и артефакты детектора, которые вносят шум в дифференциальный сигнал.

2. **Пространственно-временная обработка (EfficientPhys-C):**  
   Проект воспроизводит модель EfficientPhys-C из оригинальной статьи. Область лица масштабируется до $72 \times 72$ пикселей и передается в модель скользящими окнами по 100 кадров. Архитектура использует стандартные двумерные свертки, дополненные блоками сдвига каналов во времени (Temporal Shift Module, TSM) и пространственным вниманием, что позволяет моделировать временную динамику без тяжелых 3D-сверток.

3. **Обучение на производную сигнала и интегрирование:**  
   Модель обучена предсказывать первую временную производную пульсовой волны ($\Delta y_t$). Восстановление формы сигнала через операцию кумулятивной суммы (`cumsum`) работает как фильтр низких частот ($1/\omega$), существенно подавляя высокочастотный сенсорный шум.

4. **Цифровая обработка сигналов (DSP):**  
   Восстановленная волна центрируется, фильтруется с помощью полосового фильтра Баттерворта 2-го порядка ($0.75 - 2.5$ Гц, что соответствует $45 - 150$ уд/мин) и анализируется быстрым преобразованием Фурье (FFT) для извлечения доминирующей частоты пульса.

## Быстрый старт

### Требования
* Python 3.10+
* Стандартная USB или встроенная веб-камера

### Установка
Склонируйте репозиторий и установите зависимости:

```bash
git clone https://github.com/mmmishaa/pulse-vision.git
cd pulse-vision
pip install -r requirements.txt
```

### Настройка переменных окружения (Опционально)
Для обычного запуска с веб-камеры настройка `.env` не требуется.

Если вы хотите запустить подготовку датасета или публикацию данных на Kaggle, создайте файл `.env` из шаблона:

```bash
cp .env.example .env
```

* `HF_TOKEN`: Токен доступа Hugging Face для загрузки сырого датасета (`kyegorov/mcd_rppg`) через `src/data/prepare_mcd.py`.
* `KAGGLE_USERNAME` / `KAGGLE_KEY` (или `KAGGLE_API_TOKEN`): Учетные данные для публикации датасета на Kaggle через Kaggle API.

### Запуск приложения
Запустите основной скрипт:

```bash
python main.py
```

Предобученные веса моделей скачиваются автоматически при первом запуске.

### Управление
* Держите голову относительно неподвижно во время стартовой калибровки (~10 секунд).
* Для выхода нажмите клавишу `q` или `Esc`.

## Структура репозитория

```text
pulse-vision/
├── docs/                     # Научные статьи и справочные материалы
├── notebooks/                # Блокноты для обучения и валидации
│   ├── efficientphys-c-train.ipynb
│   └── efficientphys-c-evaluation.ipynb
├── scripts/                  # Вспомогательные скрипты
│   └── download_weights.py   # Скрипт автоматической загрузки весов
├── src/
│   ├── capture/              # Модули захвата видеопотока
│   ├── data/                 # Предобработка и загрузчики датасета
│   ├── dsp/                  # Цифровая обработка сигналов и FFT
│   ├── models/               # Архитектура EfficientPhys-C, слои и лосс-функция
│   ├── train/                # Пайплайн обучения и шедулеры
│   ├── utils/                # Конфигурация путей проекта
│   └── vision/               # Детекция ключевых точек лица MediaPipe
├── main.py                   # Точка входа для инференса с веб-камеры
└── requirements.txt          # Зависимости проекта
```

## Датасеты и блокноты

* **Оригинальная статья:** Liu et al. (2021). *EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement*. [arXiv:2110.04447](https://arxiv.org/abs/2110.04447).
* **Оригинальный датасет:** [MCD-rPPG на Hugging Face](https://huggingface.co/datasets/kyegorov/mcd_rppg)
* **Предобработанный датасет:** [MCD-rPPG Dataset на Kaggle](https://www.kaggle.com/datasets/mishasavinov/mcd-rppg-dataset)
* **Ноутбук обучения:** [efficientphys-c-train.ipynb](notebooks/efficientphys-c-train.ipynb) (Kaggle: [mishasavinov/efficientphys-c-train](https://www.kaggle.com/code/mishasavinov/efficientphys-c-train))
* **Ноутбук валидации:** [efficientphys-c-evaluation.ipynb](notebooks/efficientphys-c-evaluation.ipynb) (Kaggle: [mishasavinov/efficientphys-c-evaluation](https://www.kaggle.com/code/mishasavinov/efficientphys-c-evaluation))
* **Предобученные веса модели:** [GitHub Release v0.1.0](https://github.com/mmmishaa/pulse-vision/releases/tag/v0.1.0)
