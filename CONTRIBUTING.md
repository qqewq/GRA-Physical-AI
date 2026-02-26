# Contributing to GRA Physical AI / Как внести вклад в GRA Physical AI

First off, thank you for considering contributing to GRA Physical AI! Your help is essential for making this project a success.  
Прежде всего, спасибо, что решили внести вклад в GRA Physical AI! Ваша помощь необходима для успеха проекта.

Please take a moment to review this document in order to make the contribution process easy and effective for everyone involved.  
Пожалуйста, уделите время ознакомлению с этим документом, чтобы процесс внесения вклада был лёгким и эффективным для всех участников.

## 📋 Table of Contents / Содержание
- [Code of Conduct / Кодекс поведения](#code-of-conduct--кодекс-поведения)
- [How Can I Contribute? / Как я могу внести вклад?](#how-can-i-contribute--как-я-могу-внести-вклад)
  - [Reporting Bugs / Сообщение об ошибках](#reporting-bugs--сообщение-об-ошибках)
  - [Suggesting Enhancements / Предложение улучшений](#suggesting-enhancements--предложение-улучшений)
  - [Pull Requests](#pull-requests)
- [Development Guidelines / Рекомендации по разработке](#development-guidelines--рекомендации-по-разработке)
  - [Code Style / Стиль кода](#code-style--стиль-кода)
  - [Testing / Тестирование](#testing--тестирование)
  - [Documentation / Документация](#documentation--документация)
  - [GRA‑Specific Considerations / Особенности GRA](#gra-specific-considerations--особенности-gra)
- [Getting Help / Получение помощи](#getting-help--получение-помощи)

---

## Code of Conduct / Кодекс поведения

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [maintainer's email].  
Этот проект и все его участники регулируются [Кодексом поведения](CODE_OF_CONDUCT.md). Участвуя, вы соглашаетесь соблюдать этот кодекс. Пожалуйста, сообщайте о неприемлемом поведении по [email мейнтейнера].

---

## How Can I Contribute? / Как я могу внести вклад?

### Reporting Bugs / Сообщение об ошибках

**Before submitting a bug report** / **Перед отправкой отчёта об ошибке**:
- Check the [Issues](https://github.com/qqewq/gra-physical-ai/issues) to see if the problem has already been reported.
- Проверьте [Issues](https://github.com/qqewq/gra-physical-ai/issues), возможно, проблема уже сообщалась.

**How to submit a good bug report** / **Как составить хороший отчёт**:
- Use a clear and descriptive title.
- Describe the exact steps to reproduce the problem.
- Provide specific examples (commands, code snippets).
- Describe the behavior you observed and what you expected.
- Include details about your environment (OS, Python version, simulator versions, etc.).
- Используйте понятный и описательный заголовок.
- Опишите точные шаги для воспроизведения проблемы.
- Приведите конкретные примеры (команды, фрагменты кода).
- Опишите наблюдаемое поведение и ожидаемое.
- Укажите детали окружения (ОС, версия Python, версии симуляторов и т.д.).

### Suggesting Enhancements / Предложение улучшений

**Before submitting an enhancement suggestion** / **Перед отправкой предложения**:
- Check the [Issues](https://github.com/qqewq/gra-physical-ai/issues) to see if it's already been suggested.
- Проверьте [Issues](https://github.com/qqewq/gra-physical-ai/issues), возможно, это уже предлагалось.

**How to submit a good enhancement suggestion** / **Как составить хорошее предложение**:
- Use a clear and descriptive title.
- Provide a step-by-step description of the suggested enhancement.
- Explain why this enhancement would be useful to most users.
- Include examples of how it would work.
- Используйте понятный и описательный заголовок.
- Опишите по шагам предлагаемое улучшение.
- Объясните, почему это улучшение будет полезно большинству пользователей.
- Приведите примеры работы.

### Pull Requests

- Fill in the [Pull Request template](.github/PULL_REQUEST_TEMPLATE.md) — it helps us understand your changes.
- Follow the [Development Guidelines](#development-guidelines--рекомендации-по-разработке).
- Keep pull requests focused on a single topic — avoid mixing unrelated changes.
- If you add new code, include tests and documentation.
- Make sure all tests pass.
- Заполните [шаблон пул-реквеста](.github/PULL_REQUEST_TEMPLATE.md) — это помогает нам понять ваши изменения.
- Следуйте [Рекомендациям по разработке](#development-guidelines--рекомендации-по-разработке).
- Держите пул-реквесты сфокусированными на одной теме — избегайте смешивания несвязанных изменений.
- Если вы добавляете новый код, включите тесты и документацию.
- Убедитесь, что все тесты проходят.

---

## Development Guidelines / Рекомендации по разработке

### Code Style / Стиль кода

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- Use type hints where possible.
- Format code with [Black](https://github.com/psf/black) (line length 88).
- Use [isort](https://github.com/PyCQA/isort) to sort imports.
- Следуйте [PEP 8](https://www.python.org/dev/peps/pep-0008/) для кода Python.
- Используйте подсказки типов (type hints) где возможно.
- Форматируйте код с помощью [Black](https://github.com/psf/black) (длина строки 88).
- Используйте [isort](https://github.com/PyCQA/isort) для сортировки импортов.

### Testing / Тестирование

- Write tests for any new functionality.
- Place tests in the `tests/` directory mirroring the structure of `src/`.
- Run existing tests with `pytest` before submitting.
- If your changes affect simulation or hardware, include instructions for manual testing.
- Пишите тесты для любой новой функциональности.
- Размещайте тесты в директории `tests/`, зеркально отражая структуру `src/`.
- Запускайте существующие тесты с помощью `pytest` перед отправкой.
- Если ваши изменения затрагивают симуляцию или оборудование, включите инструкции для ручного тестирования.

### Documentation / Документация

- Update the `docs/` folder if you change functionality.
- Use clear, concise language. For bilingual docs, keep English and Russian versions synchronized.
- Include docstrings for all public modules, classes, and functions (using Google or NumPy style).
- If you add a new feature, provide an example in `docs/examples/`.
- Обновляйте папку `docs/`, если вы меняете функциональность.
- Используйте ясный и краткий язык. Для двуязычной документации синхронизируйте английскую и русскую версии.
- Включайте docstrings для всех публичных модулей, классов и функций (в стиле Google или NumPy).
- Если вы добавляете новую функцию, предоставьте пример в `docs/examples/`.

### GRA‑Specific Considerations / Особенности GRA

- When modifying GRA layers (`src/layers/`), clearly document which level (G0, G1, ...) is affected and how.
- If you change the foam calculation or nullification logic, update the mathematical description in `docs/theory/`.
- Ensure backward compatibility or clearly mark breaking changes in the pull request.
- При изменении GRA-слоёв (`src/layers/`) чётко документируйте, какой уровень (G0, G1, ...) затрагивается и как.
- Если вы меняете расчёт пены или логику обнуления, обновите математическое описание в `docs/theory/`.
- Обеспечьте обратную совместимость или чётко помечайте разрушающие изменения в пул-реквесте.

---

## Getting Help / Получение помощи

If you need help with anything, feel free to:
- Open an [Issue](https://github.com/qqewq/gra-physical-ai/issues) with the "question" label.
- Reach out to the maintainers via [email or chat, if applicable].

Если вам нужна помощь, не стесняйтесь:
- Открыть [Issue](https://github.com/qqewq/gra-physical-ai/issues) с меткой "question".
- Связаться с мейнтейнерами по [email или чат, если применимо].

---

Thank you for contributing to GRA Physical AI! 🚀  
Спасибо за вклад в GRA Physical AI! 🚀