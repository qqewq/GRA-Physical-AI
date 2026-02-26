https://doi.org/10.5281/zenodo.18790129
# GRA Physical AI

**GRA Physical AI** — это открытая платформа для создания согласованных, безопасных и этичных физических ИИ‑агентов (роботов) на основе **многоуровневой архитектуры GRA Мета‑обнулёнки**.  
Проект интегрируется со стеком NVIDIA (Isaac Lab, GR00T, Cosmos) и добавляет уровни целей (\(G_1, G_2, \dots, G_K\)), которые автоматически выявляют и устраняют конфликты между выполнением задачи, безопасностью и человеческими предпочтениями.

**GRA Physical AI** is an open platform for building aligned, safe, and ethical embodied AI agents (robots) based on the **multilevel GRA Meta‑Nullification architecture**.  
The project integrates with the NVIDIA stack (Isaac Lab, GR00T, Cosmos) and adds goal layers (\(G_1, G_2, \dots, G_K\)) that automatically detect and resolve conflicts between task performance, safety, and human preferences.

---

## 🌟 Ключевые особенности / Key Features

- **Многоуровневое согласование** – цели на нескольких уровнях (задача, безопасность, этика) работают совместно, минимизируя внутреннюю «пену».  
- **Интеграция со стеком NVIDIA** – готовая обёртка для Isaac Lab, GR00T N1.6, Cosmos Reason 2.  
- **Механизм обнуления** – при высокой пене автоматически запускается пересборка политики (дообучение, корректирующий фильтр).  
- **Открытый код** – полностью модульная архитектура, легко расширяемая под новые симуляторы и модели.  
- **Готовность к сертификации** – встроенные метрики безопасности и этики упрощают прохождение регуляторных требований.

- **Multilevel alignment** – goals at several levels (task, safety, ethics) work together, minimizing internal "foam".  
- **NVIDIA stack integration** – ready‑to‑use wrappers for Isaac Lab, GR00T N1.6, Cosmos Reason 2.  
- **Nullification mechanism** – when foam is high, automatic policy rebuilding (fine‑tuning, corrective filter) is triggered.  
- **Open source** – fully modular architecture, easily extensible to new simulators and models.  
- **Certification ready** – built‑in safety and ethics metrics simplify passing regulatory requirements.

---

## 🏛️ Архитектура / Architecture

Проект реализует иерархию уровней GRA поверх базовой политики (\(G_0\)):

- \(G_0\) – базовая политика (например, **GR00T N1.6**).  
- \(G_1\) – уровень задачи (достижение цели, эффективность).  
- \(G_2\) – уровень безопасности (отсутствие коллизий, соблюдение запретных зон).  
- \(G_3\) – уровень этики (человеческая обратная связь, «код аланов»).  
- \(G_K\) – мета‑уровень (принципы, не подлежащие обнулению).

Каждый уровень вычисляет свою **пену** \(\Phi^{(l)}\), а общая пена инициирует **обнуление** – пересмотр политики или целей.

The project implements a hierarchy of GRA layers on top of a base policy (\(G_0\)):

- \(G_0\) – base policy (e.g., **GR00T N1.6**).  
- \(G_1\) – task level (goal achievement, efficiency).  
- \(G_2\) – safety level (no collisions, restricted zones).  
- \(G_3\) – ethics level (human feedback, "Alan code").  
- \(G_K\) – meta‑level (principles that cannot be nullified).

Each level computes its own **foam** \(\Phi^{(l)}\), and the total foam triggers **nullification** – revision of the policy or goals.

Подробнее в [docs/architecture/layers.md](docs/architecture/layers.md).

---

## 🚀 Быстрый старт / Quick Start

### Требования / Prerequisites
- Python 3.9+
- NVIDIA Isaac Lab (установка по [инструкции](https://isaac-sim.github.io/IsaacLab))
- API ключи для Hugging Face (GR00T, Cosmos) – опционально

### Установка / Installation
```bash
git clone https://github.com/qqewq/gra-physical-ai.git
cd gra-physical-ai

pip install -r requirements.txt
