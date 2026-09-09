# 🎯 ART — Agent Reinforcement Trainer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

**Agentic Reinforcement Training** — GRPO (Group Relative Policy Optimization) mit LoRA für Qwen2.5 und Llama 3.1.

## 📋 Beschreibung

ART ist ein spezialisiertes Trainings-Framework, das Group Relative Policy Optimization (GRPO) mit Low-Rank Adaptation (LoRA) kombiniert. Es ermöglicht effizientes Fine-Tuning großer Sprachmodelle (Qwen2.5 7B, Llama 3.1 8B) für agentische Aufgaben — mit 4-Bit-Quantisierung, Reward-Modellierung und einer interaktiven Streamlit-Konfigurationsoberfläche.

- **GRPO-Training** — Group Relative Policy Optimization mit konfigurierbaren Generations, Beta und Temperatur
- **LoRA-Adapter** — Effizientes Fine-Tuning mit PEFT/LoRA auf Qwen und Llama
- **Reward-Modell** — Konfigurierbare Reward-Funktionen für agentische Aufgaben
- **4-Bit Quantisierung** — BitsAndBytes für speichereffizientes Training

## ✨ Features

- 🎯 **GRPO + LoRA** — State-of-the-Art RL-Training für Sprachmodelle
- 🤖 **Multi-Modell-Support** — Qwen2.5 (7B/1.5B) und Llama 3.1 (8B)
- ⚡ **4-Bit Training** — BitsAndBytes-Quantisierung für Consumer-GPUs
- 🏆 **Reward-Modell** — Flexible Reward-Funktionen mit konfigurierbaren Gewichten
- 🖥️ **Streamlit-App** — Interaktive Konfiguration von GRPO, LoRA und Reward-Parametern
- 📊 **W&B Tracking** — Vollständiges Experiment-Tracking
- 🧪 **Umfangreiche Tests** — Unit-Tests für alle Kernkomponenten
- 🔧 **vLLM Runtime** — Dedizierte Server-Integration für schnelle Inferenz

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/ART.git
cd ART

# Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# After: Serverless RL with instant feedback
from art.serverless.backend import ServerlessBackend

model = art.TrainableModel(
  project="voice-agent",
  name="agent-001",
  run_name="agent-001",
  base_model="Qwen/Qwen3.6-27B"
)

backend = ServerlessBackend(
    api_key="your_wandb_api_key"
)
model.register(backend)
# Edit and iterate in minutes, not hours!
```

## 🎮 Nutzung

### GRPO-Training starten

```bash
# Standard-Training mit Qwen 2.5 7B
python train_agent.py

# Llama 3.1 8B
python train_agent.py --model llama

# Schneller Test-Modus (Qwen 1.5B)
python train_agent.py --model qwen --test-mode

# Mit benutzerdefinierter Konfiguration
python train_agent.py --config my_config.py
```

### Streamlit-App

```bash
streamlit run app.py
```

Die App bietet drei Modi:
1. **GRPO-Training konfigurieren** — Modell, GRPO-Parameter, LoRA-Rank, Training-Setup
2. **Reward-Modell testen** — Reward-Funktionen mit Beispiel-Prompts testen
3. **LoRA-Parameter einstellen** — Rank, Alpha, Dropout, Target-Module

### Tests

```bash
pytest tests/ -v
```

## 🏗️ Tech-Stack

| Komponente | Technologie |
|---|---|
| **Sprache** | Python 3.10+ |
| **Framework** | PyTorch 2.0+, TRL (GRPOTrainer) |
| **Modelle** | Qwen2.5, Llama 3.1 |
| **Fine-Tuning** | PEFT (LoRA), BitsAndBytes (4-bit) |
| **Inferenz** | vLLM Runtime |
| **UI** | Streamlit |
| **Tracking** | Weights & Biases |
| **Testing** | pytest |

## 📁 Projektstruktur

```
ART/
├── train_agent.py          # GRPO-Training-Pipeline
├── app.py                  # Streamlit-Konfigurations-App
├── config.py               # Modell- und Trainingskonfigurationen
├── reward_model.py         # Reward-Modell und Reward-Funktionen
├── vllm_runtime/           # vLLM-Integration
│   └── src/art_vllm_runtime/
│       ├── dedicated_server.py
│       ├── lora_delta.py
│       └── patches.py
└── tests/
    ├── unit/               # Umfangreiche Unit-Tests
    │   ├── test_reward_model.py
    │   ├── test_grpo_config.py
    │   ├── test_sft.py
    │   └── ...
    └── support/
```

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Für Fragen oder Beiträge: Issue erstellen oder Pull Request öffnen.*
