# Mentalytics

Effective medical rehabilitation depends on good patient-therapist collaboration. But, cognitive impairments and psychological barriers affect nearly 30\% of rehabilitation patients, hindering clear communication and accurate patient assessments. Artificial Mental Models (AMMs) offer a solution by capturing patients' implicit expectations about therapy, aiding clearer communication and better treatment decisions. We demonstrate MENTALYTICS, a tool for knee rehabilitation, employing AMMs developed from fine-tuned large language models (LLaMA-2, LLaMA-3, Mistral, Phi-3). Trained via systematic data collection and an empirical user study (n=116), the proposed AMM predicts patients' expected pain and effort during exercises. The optimized LLaMA-3 (8B) model outperformed larger models, highlighting issues of overfitting and generalization. Results show that LLMs can serve as effective baseline models for AMMs, though challenges remain in domain-specific fine-tuning.

---

## 🧠 System Overview

Mentalytics is composed of three key modules:

### 1. **Data Acquisition**
- The data acquisition module is responsible for collecting and constructing patient profiles to support AMM training. Each patient profile is structured around key dimensions, including demographics, psychosocial factors, medical history, personality traits, and perceived difficulty levels in performing specific exercises.

### 2. **AMM Trainer**
- Uses the HuggingFace `meta-llama/Llama-3.1-8B-Instruct` model.
- Embeddings are computed using SentenceTransformers and stored in a **Redis vector database**.
- Injects relevant context dynamically into the prompt for accurate AI responses.
- Auxiliary model designed to mitigate biases and incorrect responses inherent in fine-tuned models.

### 4. **AMM Predictor**
- `Mentalytics Overview`: Project intro and architecture
- `Patient Profiles`: Patient Database
- `Physio Corner`: Visualization of anticipated pain & personality insights
- `AI Assistant`: Real-time therapy Q&A with context-aware AI agent

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure the following:

- Python 3.8+
- Access to HuggingFace with permission to use `meta-llama/Llama-3.1-8B-Instruct`
- Redis server (with vector DB support) running and accessible

---

### 📦 Installation

1. Clone this repository or download the ZIP and extract it.


3. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

4. Ensure Redis server with vector search is running and properly connected.



### Running the Web Application

To run the web application:

```bash
streamlit run fedwell_2_v2.py
```

1. Open a web browser and go to the address provided by Streamlit to access the web application.
