# AI-HR-Selection

Проект по созданию ИИ-системы для автоматизации отбора кандидатов на основе анализа резюме, тестов и цифрового следа.  
Основная модель: ансамбль **NLP + Random Forest**.

---

## Цель проекта
- Сократить время и ресурсы HR-подразделений.
- Снизить субъективность при отборе кандидатов.
- Повысить качество найма за счёт интеллектуальной оценки.

---

## Структура репозитория
- `docs/` — документация:
  - [case_description.md](docs/case_description.md) — описание кейса.
  - [data_pipeline.md](docs/data_pipeline.md) — система хранения и обработки данных.
  - [model_choice.md](docs/model_choice.md) — обоснование выбора NLP + Random Forest.
  - [integration.md](docs/integration.md) — интеграция в HR-процессы.
  - [benefits_limits.md](docs/benefits_limits.md) — выгоды и ограничения.
- `src/` — исходный код:
  - `preprocessing/` — NLP-пайплайн.
  - `models/` — Random Forest и ансамбль.
  - `evaluation/` — метрики качества.
- `notebooks/` — Jupyter/Colab эксперименты.
- `data/` — примеры данных (анонимизированные).
- `tests/` — тесты для проверки моделей.

---

## Технологии
- **NLP**: BERT / RoBERTa для анализа текста резюме.
- **Random Forest**: классификация на основе признаков.
- **Хранилище**: PostgreSQL + MongoDB + облако (AWS/GCP/Azure).
- **Метрики**: Precision, Recall, F1-score, ROC-AUC.

---

## Интеграция в HR
- Панель для рекрутеров с рейтингом кандидатов.
- API для подключения к HRM-системам (Workday, SAP SuccessFactors).
- Explainable AI для прозрачности решений.

---

## Выгоды и ограничения
**Выгоды**: экономия времени, снижение субъективности, масштабируемость.  
**Ограничения**: этические риски, юридические требования (GDPR, ФЗ-152), необходимость обновления модели.

---

## Как начать

```bash
# 1. Клонировать репозиторий
git clone https://github.com/username/AI-HR-Selection.git

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить ноутбук
jupyter notebook notebooks/Ensemble_pipeline.ipynb

