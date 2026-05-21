# **Confundo: Learning to Generate Robust Poison for Practical RAG Systems**

![](assets/overview.png)
**Abstract**: Retrieval-augmented generation (RAG) is increasingly deployed in real-world applications, where its reference grounded design makes outputs appear trustworthy. This trust has spurred research on poisoning attacks that craft malicious content, inject it into knowledge sources, and manipulate RAG responses. However, when evaluated in practical RAG systems, existing attacks suffer from severely degraded effectiveness. This gap stems from two overlooked realities: (i) content is often processed before use, which can fragment the poison and weaken its effect, and (ii) users often do not issue the exact queries anticipated during attack design. These factors can lead practitioners to underestimate risks and develop a false sense of security. To better characterize the threat to practical systems, we present Confundo, a learning-to-poison framework that fine-tunes a large language model as a poison generator to achieve high effectiveness, robustness, and stealthiness. Confundo provides a unified framework supporting multiple attack objectives, demonstrated by manipulating factual correctness, inducing biased opinions, and triggering hallucinations. By addressing these overlooked challenges, Confundo consistently outperforms a wide range of purpose built attacks across datasets and RAG configurations by large margins, even in the presence of defenses. Beyond exposing vulnerabilities, we also present a defensive use case that protects web content from unauthorized incorporation into RAG systems via scraping, with no impact on user experience.

---
For more technical details and experimental results, we invite you to check out our paper [[here]](https://arxiv.org/abs/2602.06616):  
**Haoyang Hu, Zhejun Jiang, Yueming Lyu, Junyuan Zhang, Yi Liu and Ka-Ho Chow,** *"Confundo: Learning to Generate Robust Poison for Practical RAG Systems,"*  [TODO]

```
@inproceedings{hu2026confundo,
  title={Confundo: Learning to Generate Robust Poison for Practical RAG Systems},
  author={Hu, Haoyang and Jiang, Zhejun and Lyu, Yueming and Zhang, Junyuan and Liu, Yi and Chow, Ka-Ho},
  booktitle={35th USENIX Security Symposium (USENIX Security 26)},
  year={2026}
}
```
---
## 🚀 Quick Start

### 1. Installation & Environment Setup

Clone this repository and install the required dependencies:
```
git clone [TODO]
cd [TODO]
pip install -r [TODO]
```

### 2. Running Core Attacks
Confundo provides a unified framework capable of supporting multiple adversarial objectives. We provide interactive notebooks for rapid deployment, along with pre-configured attack demonstrations across three core axes:

* **Factual Correctness Manipulation Attack**: Deceive the RAG to provide incorrect answers.

  👉 Run the interactive demo: [`./factual_correctness_manipulation.ipynb`](./factual_correctness_manipulation.ipynb)

* **Hallucination Induction**: Inducing RAG to produce hallucinations.

  👉 Run the interactive demo: [`./hallucination_induction.ipynb`](./hallucination_induction.ipynb)

* **Opinion Manipulation**: Manipulate RAGs to arrive at answers with subjective opinions.
  
  👉 Run the interactive demo: [`./opinion_manipulation.ipynb`](./opinion_manipulation.ipynb)

#### 🤗 Pre-trained Models
You can instantly access and download our open-sourced, fine-tuned poison generators directly from HuggingFace:
* **Factual Correctness:** [confundo-correctness](https://huggingface.co/AlienHu/confundo-correctness)
* **Opinion Manipulation:** [confundo-opinion](https://huggingface.co/AlienHu/confundo-opinion)
* **Hallucination Induction:** [confundo-hallucination](https://huggingface.co/AlienHu/confundo-hallucination)

---

## 📊 Cross-Pipeline Transferability & Robustness Analysis

### 🔄 Framework Transferability

Evaluate the effectiveness of Confundo across popular, unknown RAG pipelines:

- **LlamaIndex Pipeline Evaluation:**
  ```
  python ./transferability/pipeline/llamaindex.py
  ```

- **Haystack Pipeline Evaluation:**
  ```
  python ./transferability/pipeline/haystack.py
  ```

### 🛡️ Survivability Under Defensive Configurations

Test the resilience of our generated poison against various industry-standard defensive mechanics.

#### A. Heuristic Filtering

Assess how well the adversarial content bypasses automated document preprocessing and statistical guards:

- **Perplexity Filtering:**
  ```
  python ./defenses/filter/filtering_perplexity.py
  ```

- **Duplicate-Text Filtering:**
  ```
  python ./defenses/filter/filtering_duplicate.py
  ```

- **Keyword Density Filtering:**
  ```
  python ./defenses/filter/filtering_keyword.py
  ```

#### B. LLM-Based Intent Detection

Evaluate survival rates when commercial LLM judges act as intent monitors to detect adversarial content injection::
```
python ./defenses/intent.py
```

#### C. Paraphrasing

Measure the robustness when queries or retrieved documents undergo rewriting across three levels:

- **Level I — Paraphrase Question:** Rewrite user queries.
  ```
  python ./defenses/paraphase/paraphase-1.py
  ```

- **Level II — Paraphrase Retrieved Entry:** Rewrite retrieved documents.
  ```
  python ./defenses/paraphase/paraphase-2.py
  ```

- **Level III — Paraphrase Both Question and Entry:** Rewrite both queries and retrieved documents.
  ```
  python ./defenses/paraphase/paraphase-3.py
  ```

#### D. Reranking

Analyze performance degradation when a rerank model alters the position and priority of the poison entries:
```
python ./defenses/rerank.py
```

## 🏋️ Training the Poison Generator

### 1. Data Preparation

To fine-tune the generator, you must first prepare target answers and semantic variants. You can generate these customized assets using our notebook or bootstrap directly from our pre-packaged examples:

- **Target Answer Generation:** Generate incorrect target responses using the interactive tool `./tools/gen.ipynb`, or leverage our ready-made setup at:

  `./data/input_harry_potter_confundo.json`

- **Paraphrased Variants Generation:** Synthesize diverse query/entry variations via `./tools/gen.ipynb`, or load our optimization dataset directly from:

  `./data/RL_train_harry_potter.json`

### 2. Fine-Tuning Execution

Launch the reinforcement learning alignment loop to fine-tune the poison generator:
- Perplexity Filtering: ```python ./defenses/filter/filtering_keyword.py```
- Duplicate-Text Filtering: ```python ./defenses/filter/filtering_keyword.py```
- Keyword Density Filtering: ```python ./defenses/filter/filtering_keyword.py```

> 💡 **Note on Computation Time & Hardware:** A full training sequence typically requires a GPU with at least 48GB VRAM and takes 10+ hours, depending on your specific hardware configurations. Please note that training parameters or configurations may need adjustment based on your experimental environment (e.g., hardware or package versions). For a rapid preview, we provide a simplified training pipeline script at `[TODO]`. Alternatively, you can bypass the training phase entirely by downloading our fine-tuned models directly from HuggingFace:
* Factual Correctness: [confundo-correctness](https://huggingface.co/AlienHu/confundo-correctness)
* Opinion Manipulation: [confundo-opinion](https://huggingface.co/AlienHu/confundo-opinion)
* Hallucination Induction: [confundo-hallucination](https://huggingface.co/AlienHu/confundo-hallucination)

---

📬 If you encounter any issues, feel free to contact: haoyanghu@connect.hku.hk
