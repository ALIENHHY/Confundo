# **Confundo: Learning to Generate Robust Poison for Practical RAG Systems**

![](assets/intro-git.png)
**Abstract**: Retrieval-augmented generation (RAG) is increasingly deployed in real-world applications, where its reference grounded design makes outputs appear trustworthy. This trust has spurred research on poisoning attacks that craft malicious content, inject it into knowledge sources, and manipulate RAG responses. However, when evaluated in practical RAG systems, existing attacks suffer from severely degraded effectiveness. This gap stems from two overlooked realities: (i) content is often processed before use, which can fragment the poison and weaken its effect, and (ii) users often do not issue the exact queries anticipated during attack design. These factors can lead practitioners to underestimate risks and develop a false sense of security. To better characterize the threat to practical systems, we present Confundo, a learning-to-poison framework that fine-tunes a large language model as a poison generator to achieve high effectiveness, robustness, and stealthiness. Confundo provides a unified framework supporting multiple attack objectives, demonstrated by manipulating factual correctness, inducing biased opinions, and triggering hallucinations. By addressing these overlooked challenges, Confundo consistently outperforms a wide range of purpose built attacks across datasets and RAG configurations by large margins, even in the presence of defenses. Beyond exposing vulnerabilities, we also present a defensive use case that protects web content from unauthorized incorporation into RAG systems via scraping, with no impact on user experience.

For more technical details and experimental results, we invite you to check out our paper [[here]](https://arxiv.org/abs/2602.06616):  
**Haoyang Hu, Zhejun Jiang, Yueming Lyu, Junyuan Zhang, Yi Liu and Ka-Ho Chow,** *"Confundo: Learning to Generate Robust Poison for Practical RAG Systems,"*  [TODO]

```
@article{hu2026confundo,
  title={Confundo: Learning to Generate Robust Poison for Practical RAG Systems},
  author={Hu, Haoyang and Jiang, Zhejun and Lyu, Yueming and Zhang, Junyuan and Liu, Yi and Chow, Ka-Ho},
  journal={arXiv preprint arXiv:2602.06616},
  year={2026}
}
```
---
## 🚀 Quick Start

### 1. Installation & Environment Setup

Clone this repository and install the required dependencies:

Bash

```
git clone [TODO]
cd [TODO]
pip install -r [TODO]
```

### 2. Running Core Attacks

Confundo provides a unified framework capable of supporting multiple adversarial objectives. You can initiate specific poisoning implementations across three core axes:

- **Factual Correctness Manipulation Attack**: Forces the RAG system to generate completely incorrect or misleading factual assertions.
- **Hallucination Induction**: Intentionally triggers ungrounded hallucinations within the target LLM's responses.
- **Opinion Manipulation**: Systematically skews the generation towards biased perspectives or target viewpoints.

## 📊 Cross-Pipeline Transferability & Robustness Analysis

### 🔄 Framework Transferability

Evaluate how effectively Confundo's generated poison transfers across popular, unknown production-level RAG orchestration frameworks:

- **LlamaIndex Pipeline Evaluation:**

  Bash

  ```
  python ./transferability/pipeline/llamaindex.py
  ```

- **Haystack Pipeline Evaluation:**

  Bash

  ```
  python ./transferability/pipeline/haystack.py
  ```

### 🛡️ Survivability Under Defensive Configurations

Test the resilience of our generated poison against various industry-standard defensive mechanics.

#### A. Heuristic Filtering

Assess how well the adversarial content bypasses automated document preprocessing and statistical guards:

- **Perplexity Filtering:**

  Bash

  ```
  python ./defenses/filter/filtering_perplexity.py
  ```

- **Duplicate-Text Filtering:**

  Bash

  ```
  python ./defenses/filter/filtering_duplicate.py
  ```

- **Keyword Density Filtering:**

  Bash

  ```
  python ./defenses/filter/filtering_keyword.py
  ```

#### B. LLM-Based Intent Detection

Evaluate survival rates when advanced LLM judges act as context monitors to detect adversarial content injection:

Bash

```
python ./defenses/intent.py
```

#### C. Semantic Paraphrasing Defenses

Measure the robustness of the attack when queries or retrieved contexts undergo rewriting across three distinct operational levels:

- **Level I — Paraphrase Question:** Rewrites user queries to break exact-match triggers.

  Bash

  ```
  python ./defenses/paraphase/paraphase-1.py
  ```

- **Level II — Paraphrase Retrieved Entry:** Rewrites the retrieved text fragments to break localized adversarial structures.

  Bash

  ```
  python ./defenses/paraphase/paraphase-2.py
  ```

- **Level III — Paraphrase Both Question and Entry:** Full-spectrum rewriting defense.

  Bash

  ```
  python ./defenses/paraphase/paraphase-3.py
  ```

#### D. Neural Reranking

Analyze performance degradation when a neural rerank model alters the position and priority of the poisoned entries within the prompt context:

Bash

```
python ./defenses/rerank.py
```

## 🏋️ Training the Poison Generator

### 1. Data Preparation

To optimize the generator, you must first prepare target answers and semantic variants. You can generate these customized assets using our notebook or bootstrap directly from our pre-packaged examples:

- **Target Answer Generation:** Generate incorrect target responses using the interactive tool `./tools/gen.ipynb`, or leverage our ready-made setup at:

  `./data/input_harry_potter_confundo.json`

- **Paraphrased Variants Generation:** Synthesize diverse query/entry variations via `./tools/gen.ipynb`, or load our optimization dataset directly from:

  `./data/RL_train_harry_potter.json`

### 2. Fine-Tuning Execution

Launch the reinforcement learning alignment loop to fine-tune the LLM poison generator:

Bash

```
python ./rl_train.py
```

> 💡 **Note on Computation Time:** A full training sequence typically requires **10+ hours** depending on your hardware specifications. For a rapid preview, we provide a **simplified training pipeline** script at `[TODO]`. Alternatively, you can bypass the training phase entirely by downloading our pre-trained, fully-optimized checkpoints directly from HuggingFace at `[TODO]`.

## 📬 Contact & Support

If you encounter any issues, bugs, or have inquiries regarding the methodology and code, feel free to open an issue or contact:

- **Email:** haoyanghu@connect.hku.hk
