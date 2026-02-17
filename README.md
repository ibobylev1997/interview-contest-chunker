# interview-contest-chunker

Решение тестового задания: функция, которая режет `pandas.DataFrame` на чанки по колонке дат/таймстемпов так, чтобы одинаковые значения даты не попадали в разные чанки.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Запуск тестов

```bash
pytest -q
```
