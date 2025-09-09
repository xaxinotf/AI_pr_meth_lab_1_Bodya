




---

# AI\_pr\_meth\_lab\_1\_Bodya

**Предмет:** *Штучний інтелект: принципи та методи*
**Викладач:** *Марченко Олександр Олександрович* — професор, гарант ОНП “Інтелектуальні системи”

> Датасет: **Electricity Load Diagrams 2011–2014 (LD2011)**
> Kaggle-міра: [https://www.kaggle.com/datasets/michaelrlooney/electricity-load-diagrams-2011-2014](https://www.kaggle.com/datasets/michaelrlooney/electricity-load-diagrams-2011-2014)

---

## 1) Постанова задачі

Потрібно побудувати систему прогнозування електроспоживання.
Ми прогнозуємо значення навантаження (load) наперед на певний горизонт часу, маючи історію вимірів.

* **Вхід:** історичні вимірювання споживання електроенергії (часовий ряд), опціонально — лагові ознаки та календарні індикатори.
* **Вихід:** прогноз на горизонти N кроків (у нашому проєкті типово 1–24 години).
* **Метрики якості:** MAE, RMSE, MAPE.

Щоб порівняти підходи різної складності, робимо три моделі:

1. **Проста (Baseline):** лінійна регресія з ElasticNet-регуляризацією на лагових ознаках.
2. **Середня (LSTM):** невелика рекурентна нейромережа (LSTM) з ковзними вікнами.
3. **Складна (Transformer):** Temporal Fusion Transformer (TFT) із пакета `pytorch-forecasting`.

Усі три моделі зберігають прогнози та метрики у `reports/tables/`, а **Dashboard (Dash/Plotly)** дозволяє подивитися графіки: накладання та side-by-side, а також таблицю метрик.

---

## 2) Опис датасету (LD2011)

* Період: **2011–2014**.
* Частота: **15 хвилин** (у проєкті також використовуємо агрегацію **1 година**).
* Об’єкти: **≈370 споживачів** (ідентифікатори колонок: `MT_001`, `MT_002`, …).
* Формат: wide-таблиця з датами у рядках та колонками — клієнти (кожна комірка — споживання в kWh за інтервал).

У проєкті:

* вихідний файл зберігається як `data/raw/LD2011_2014.txt`;
* перетворені версії лежать у `data/interim/`:

  * `ld2011_15min.parquet` — 15-хв. дані;
  * `ld2011_hourly.parquet` — ті самі дані, агреговані по годинах (mean).

---

## 3) Структура репозиторію

```
.
├─ dash_app/
│  ├─ assets/                 # стилі Dash
│  └─ app.py                  # Dash-додаток
├─ data/
│  ├─ raw/                    # сирі дані (LD2011_2014.txt)
│  ├─ interim/                # підготовлені parquet (15min, hourly)
│  └─ processed/              # (резерв під подальшу обробку)
├─ models/
│  ├─ baseline/               # ваги простої моделі (*.pt)
│  ├─ lstm/                   # ваги LSTM (*.pt)
│  └─ transformer/            # логи/чекпоінти TFT
├─ reports/
│  ├─ figures/                # збережені графіки (опційно)
│  └─ tables/                 # CSV з прогнозами та метриками
├─ src/
│  ├─ data.py                 # завантаження сирого LD2011 + збереження parquet
│  ├─ features.py             # генерація ознак (лагів) для baseline/LSTM
│  ├─ eval.py                 # MAE/RMSE/MAPE
│  ├─ train_baseline.py       # навчання лінійної моделі
│  ├─ train_lstm.py           # навчання LSTM
│  └─ train_transformer.py    # навчання TFT (pytorch-forecasting)
└─ README.md
```

---

## 4) Як це реалізовано (короткий огляд модулів)
<img width="1851" height="1100" alt="image" src="https://github.com/user-attachments/assets/97c7b46e-2f7a-4535-aff6-5568c00f3a84" />
### `src/data.py`

* `load_ld2011()` — читає `data/raw/LD2011_2014.txt` у wide-форматі з `DatetimeIndex`.
* `save_parquet_versions(df)` — зберігає дві версії: 15 хв (`interim/ld2011_15min.parquet`) і hourly (`interim/ld2011_hourly.parquet`) та повертає шляхи.

### `src/features.py`

* `build_feature_table(df_wide, series_id)` — формує “довгий” датафрейм для однієї серії з лагами та базовими календарними фічами (використовується baseline/LSTM).

### `src/eval.py`

* Реалізації метрик `mae`, `rmse`, `mape` над векторами `y_true`, `y_pred`.

### `src/train_baseline.py`

* **Модель:** проста лінійна регресія (`torch.nn.Linear`) + ElasticNet-регуляризація у loss.
* **Вхід:** обрана серія (`--series_id`), лаги (`--lags`), горизонт (`--horizon`), частота (`--freq` = `hourly` / `15min`).
* **Вихід:**

  * `reports/tables/baseline_<series>_forecast.csv` з прогнозами (OOF)
  * `reports/tables/baseline_<series>_metrics.csv` з підсумковими MAE/RMSE/MAPE
  * `models/baseline/<series>.pt` — збережені ваги.

### `src/train_lstm.py`

* **Модель:** маленька LSTM (`TinyLSTM`) поверх ковзних вікон; вхід — ознаки з `features.py`.
* **Аргументи:** `--series_id`, `--T` (довжина історичного вікна), `--horizon`, `--batch`, `--epochs`, `--lr`.
* **Вихід:**

  * `reports/tables/lstm_<series>_forecast.csv` / `lstm_<series>_metrics.csv`
  * `models/lstm/<series>.pt`.

### `src/train_transformer.py`

* **Модель:** Temporal Fusion Transformer (TFT) із `pytorch-forecasting`.
* Працює **на GPU**. Має швидкий старт, тихі логи, прапорець `--skip` (щоб лише підготувати дані та не тренувати).
* Підтримує різні версії `pytorch-forecasting` (робастний парсер виходу `model.predict(..., return_x=True, return_index=True)`).
* Зберігає **весь горизонт** і, якщо доступно, `series_id` у прогнозі.
* **Вихід:**

  * `reports/tables/transformer_MT_all_forecast.csv`
  * `reports/tables/transformer_MT_all_metrics.csv`
  * чекпоінти у `models/transformer/tft_logs/...`.

### `dash_app/app.py`

* Візуалізація трьох моделей: *overlay*, *side-by-side* і *metrics-table*.
* Читає CSV з `reports/tables/`.
* Якщо у трансформера в прогнозі є `series_id`, графік можна легко фільтрувати по серії (підтримка закладена — див. блок із читанням `transformer_MT_all_forecast.csv`).

---

## 5) Встановлення

> **Передумови:** NVIDIA GPU з CUDA (у нас було `torch 2.5.1+cu121`, `lightning 2.5.5`, `pytorch_forecasting 1.0.0`).
> Windows 10/11, Python 3.10–3.12.

```powershell
# 1) Клон
git clone https://github.com/xaxinotf/AI_pr_meth_lab_1_Bodya.git
cd AI_pr_meth_lab_1_Bodya

# 2) Віртуальне середовище
python -m venv .venv
.venv\Scripts\activate

# 3) Встановлення залежностей (мінімальний набір)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.5.5 pytorch-forecasting==1.0.0
pip install pandas numpy scikit-learn
pip install dash plotly
```

> **Дані:** поклади файл `LD2011_2014.txt` у `data/raw/`. Перший запуск будь-якого скрипта сам створить parquet-версії у `data/interim/`.

---

## 6) Команди запуску

### 6.1 Baseline (лінійна регресія + ElasticNet)

```powershell
# Приклад: одна серія, горизонт 96 кроків, 672 лаги, частота 15 хв
python -m src.train_baseline --series_id MT_001 --horizon 96 --lags 672 --freq 15min
```

Після завершення дивись:

* `reports/tables/baseline_MT_001_forecast.csv`
* `reports/tables/baseline_MT_001_metrics.csv`

---

### 6.2 LSTM

```powershell
# Приклад: вікно 168, горизонт 24, 3 епохи
python -m src.train_lstm --series_id MT_001 --T 168 --horizon 24 --batch 256 --epochs 3 --lr 1e-3
```

Результати:

* `reports/tables/lstm_MT_001_forecast.csv`
* `reports/tables/lstm_MT_001_metrics.csv`

---

### 6.3 Transformer (TFT)

**Швидкий прогін (мало серій, коротка історія):**

```powershell
python -m src.train_transformer --max_series 5 --enc 48 --pred 12 --epochs 1 --workers 0
```

**Середній прогін (трошки більше даних):**

```powershell
python -m src.train_transformer --max_series 20 --enc 96 --pred 24 --epochs 2 --workers 0
```

**Лише підготувати дані (без тренування/інференсу):**

```powershell
python -m src.train_transformer --skip
```

Після завершення дивись:

* `reports/tables/transformer_MT_all_forecast.csv`
* `reports/tables/transformer_MT_all_metrics.csv`

---

### 6.4 Dash-додаток

```powershell
python -m dash_app.app
```

Відкриється на [http://127.0.0.1:8050](http://127.0.0.1:8050) і покаже:

* вкладку **Compare (overlay)** — накладені Actual та прогнози;
* **Side-by-side** — окремі панелі для кожної моделі;
* **Metrics** — таблицю MAE/RMSE/MAPE.

---

## 7) Як я це виконав (коротко про рішення)

1. **Підготовка даних**

   * Зчитав сирий LD2011 (`data/raw/LD2011_2014.txt`), зберіг parquet-версії (15хв та hourly).
   * Для Transformer перевів wide→long, додав `time_idx` (години від мінімального часу).

2. **Baseline**

   * Побудував лагові ознаки й просту лінійну модель з L1/L2.
   * Валідовував TimeSeriesSplit, збирав OOF-прогнози, рахував метрики.

3. **LSTM**

   * Згенерував ковзні вікна `(T → y[t+horizon])`.
   * Невелика LSTM + лінійна голова, тренування на GPU.
   * Прогноз для всіх вікон без shuffle; збереження у CSV.

4. **Transformer (TFT)**

   * Використав `pytorch-forecasting` (TemporalFusionTransformer) з QuantileLoss, але **зберігаю поінтовий прогноз** (для сумісності версій).
   * Зробив код робастним до різних форматів виходу `model.predict()`: `(preds, x)` або `(preds, x, index)`.
   * Зберігаю **весь горизонт** та, якщо можливо, `series_id` і конвертую `time_idx → timestamp`.

5. **Візуалізація**

   * Dash/Plotly: три вкладки; дані тягнуться з `reports/tables/*.csv`.
   * Якщо у трансформера є `series_id`, можна фільтрувати конкретну серію (патч уже передбачений у коді завантаження).

---

## 8) Типові помилки / поради

* **Tkinter: `RuntimeError: main thread is not in main loop`**
  Ми насильно встановили бекенд Matplotlib у `Agg` і відключили будь-яку взаємодію з Tk (`os.environ["MPLBACKEND"]="Agg"` у `train_transformer.py`).

* **`predict_dataloader` warnings (num\_workers)**
  На Windows залишай `--workers 0` для стабільності.

* **Різні версії `pytorch-forecasting`**
  У нас: `pytorch_forecasting==1.0.0`. Повернення з `model.predict(return_x=True)` може відрізнятися за довжиною tuple — код це обробляє.

* **Очистити старі логи/виходи**
  Якщо змінив параметри і хочеш “чистий” запуск, прибери:

  ```
  models/transformer/tft_logs/
  reports/tables/transformer_MT_all_*.csv
  ```

  (Baseline/LSTM мають власні файли у `reports/tables/`.)

---

## 9) Швидкі шпаргалки команд (PowerShell, в один рядок)

* **TFT — невеликий прогін**

  ```powershell
  python -m src.train_transformer --max_series 5 --enc 48 --pred 12 --epochs 1 --workers 0
  ```

* **TFT — середній прогін**

  ```powershell
  python -m src.train_transformer --max_series 20 --enc 96 --pred 24 --epochs 2 --workers 0
  ```

* **Baseline**

  ```powershell
  python -m src.train_baseline --series_id MT_001 --horizon 96 --lags 672 --freq 15min
  ```

* **LSTM**

  ```powershell
  python -m src.train_lstm --series_id MT_001 --T 168 --horizon 24 --batch 256 --epochs 3 --lr 1e-3
  ```

* **Dash**

  ```powershell
  python -m dash_app.app
  ```


**Якщо щось “ломається” — пиши параметри запуску та лог, виправимо на місці.**


