# Challenges for AI Content Detection

Large Language Models (LLMs) have become highly proficient at generating useful content with minimal human input or effort. These models are increasingly accessible to users, either through established product interfaces like ChatGPT or open-source implementations such as the Llama family. The growing capability and availability of LLMs have led to both overt and concealed use. Concealed use, or ghostwriting, has driven the development of methods to detect AI-generated content. However, detectors face significant challenges, including the broad range of topics submitted for analysis, the diversity of LLMs available to users, and deliberate efforts by users to manipulate generated content to evade detection.

# Our Approach

We propose **APOLLO**, a method for detecting AI-generated content that combines zero-shot and supervised detection techniques. The zero-shot component is a perplexity-based detector, which uses an LLM to estimate the likelihood of text being AI-generated. AI-generated content typically exhibits higher likelihood (lower perplexity) compared to human-generated text. The supervised component is trained on a carefully curated dataset of human and AI-generated text. APOLLO integrates these two approaches into an ensemble to enhance generalization and accuracy.

# Dataset

## Training Data

The dataset is arguably the most critical component for training and evaluating AI detectors. A comprehensive dataset should include diverse LLMs, domains, decoding strategies, and adversarial attacks. The most extensive open-source dataset we are aware of is **RAID** [[1]](#reference-1). It provides paired human-written and AI-generated content, helping prevent detectors from relying on domain or subject matter biases. We used the version of RAID that spans eight domains, eleven LLM generators, and eleven adversarial attacks.

Despite its comprehensiveness, RAID is not exhaustive. For instance, it does not include the most recent LLMs, such as GPT4o, and its prompts for generating AI text lack variety. To address these gaps, we incorporated several additional open-source datasets, culminating in data from approximately 14 sources detailed in [README.md](data/README.md). Furthermore, we generated supplemental data using carefully designed prompts to enrich the dataset.

## Evaluation Data
A key aspect of formulating a robust evaluation is creating an out-of-sample dataset to assess the performance of models on unseen data. We consider three types of evaluation data:

1. **Hold-out Data:** A portion of the training data deliberately set aside for evaluation purposes. We carefully construct out-of-domain test data to ensure robust evaluation of model performance on previously unseen distributions.

2. **External Open-Source Data:** Several open-source datasets excluded from the training set, specifically:
   - [GPT-4o 200k Dataset](https://huggingface.co/datasets/PawanKrd/gpt-4o-200k)
   - [Anthropic Persuasion Dataset](https://huggingface.co/datasets/Anthropic/persuasion)
   - [Alpaca-GPT4 Dataset](https://huggingface.co/datasets/vicgalle/alpaca-gpt4/tree/main)

3. **Manually Generated Data:** We manually compiled a dataset comprising approximately 100 human-generated and 100 AI-generated samples. This dataset was constructed by copying and pasting text encountered during web browsing to ensure it is representative of typical online content. The AI-generated samples were sourced from GPT-4o, Claude, Google Search AI, and other large language models.

# Open-Source Models

In this work, we evaluate various detection methods, focusing on zero-shot and supervised approaches. A state-of-the-art model in open-source AI detection is **Binoculars** [[2]](#reference-2), a perplexity-based method that leverages two large language models (LLMs). We also assessed **Zippy** [[5]](#reference-5), an open-source, compression-based zero-shot method with conceptual similarities to perplexity-based techniques. Additionally, we explored several data-driven or supervised detection methods, including the use of embeddings from **UAR** [[4]](#reference-4) to construct a nearest neighbor model with open-source data. These methods were selected for their commercial-friendly licenses and computational efficiency. Consequently, certain popular methods, such as **RADAR** [[7]](#reference-7), **Ghostbuster** [[8]](#reference-8), and **DetectGPT** [[9]](#reference-9), were excluded from the comparison.

Furthermore, We fine-tuned a **RoBERTa** [[3]](#reference-3) classifier, which serves as a component of the APOLLO ensemble. The model has been made publicly available on [Hugging Face](https://huggingface.co/fakespotailabs/roberta-base-ai-text-detection-v1) under the Apache-2.0 license.

# Performance

We begin by evaluating the detectors under **in-domain testing**, highlighting the limitations of such setups and underscoring the importance of **out-of-domain testing** for assessing generalization. Next, we demonstrate the robust performance of APOLLO in a strict out-of-domain setting, both with and without attacks. Finally, we evaluate APOLLO on a more realistic dataset to showcase its effectiveness in practical scenarios.

## Detectors on IID Data
AI detection products often report perfect accuracy on their evaluation datasets; however, when these models are evaluated externally [[1]](#reference-1), they frequently exhibit significant errors. This phenomenon can largely be attributed to **in-domain testing**. Models are typically evaluated by splitting a dataset randomly into training and test sets. The model is then trained on the training set and evaluated on the test set, which results in data that is independent and identically distributed (IID). In the context of AI detection, supervised models tend to perform exceptionally well on IID datasets.

Table 1 below illustrates the performance of various models on the RAID dataset [[1]](#reference-1) when randomly split into training and test sets. Supervised models like RoBERTa perform almost perfectly under these conditions, outperforming the zero-shot model Binoculars [[2]](#reference-2). At first glance, RoBERTa might seem to be the superior detector. 

However, in the following sections, we demonstrate why perplexity-based methods are still essential for generalization and how **APOLLO**, an ensemble approach, achieves better performance across diverse datasets compared to any single method.

```math
\begin{array}{c|c} 
\textbf{Model} & \textbf{AUROC (RAID)}\\ 
\hline 
\text{UAR} & 0.69\\ 
\text{Zippy} & 0.66\\ 
\text{Binoculars} & 0.82\\ 
\hline 
\text{RoBERTa} & 0.99^*\\ 
\text{APOLLO} & 0.99^*\\ 
\end{array}
```

**Table 1: IID Performance:** AUROC scores on the RAID dataset using random train-test splits. The top group of models is trained on external datasets, while the bottom group uses portions of RAID for training. * indicates the highest performance.

## Generalization Without Attacks

In this evaluation, we analyze a slice of the RAID dataset containing documents generated without adversarial attacks, repetition penalties, or heavily manipulated sampling settings. This scenario reflects a typical positive case for AI detectors, where users directly copy and paste text generated by models without modifications. While supervised models handle IID test samples well, our goal here is to assess model performance on text from a new domain (subject or topic) or generated by an unseen LLM.

Tables 2 and 3 present the results for out-of-domain (OOD) and out-of-LLM (OOLLM) evaluations, respectively. The first two columns in each table show the mean and worst-case performance across all tested categories. For OOD evaluation (Table 2), RoBERTa shows a notable drop in performance compared to its IID results from Table 1, highlighting its limited generalization to new domains. On the other hand, Binoculars, a zero-shot method, demonstrates relative robustness across domains. Similarly, APOLLO—an ensemble of supervised and zero-shot methods—achieves top-tier performance by combining the strengths of both approaches.

In OOLLM evaluation (Table 3), supervised models, such as RoBERTa, perform better but can still exhibit variation in performance depending on the held-out generator. Binoculars again shows consistent robustness, particularly in handling unseen LLMs. APOLLO outperforms individual methods in both evaluations, leveraging the in-domain strength of supervised models and the domain and generator robustness of zero-shot techniques.

```math
\begin{array}{c|cc|cccccccc} 
& \textbf{MEAN} & \textbf{WORST} & \textbf{Abstracts} & \textbf{Books} & \textbf{News} & \textbf{Poetry} & \textbf{Recipes} & \textbf{Reddit} & \textbf{Reviews} & \textbf{Wiki}\\ 
\hline 
\text{UAR} & 0.73 & 0.54 & 0.96 & 0.58 & 0.84 & 0.55 & 0.80 & 0.78 & 0.81 & 0.54\\ 
\text{Zippy} & 0.77 & 0.72 & 0.84 & 0.86 & 0.78 & 0.72 & 0.74 & 0.72 & 0.78 & 0.75\\ 
\text{Binoculars} & 0.98^* & 0.96 & 0.97^* & 0.99^* & 0.98 & 0.98 & 0.97 & 0.96 & 0.98 & 0.97\\ 
\hline 
\text{RoBERTa} & 0.86 & 0.31 & 0.71 & 0.31 & 0.97 & 0.95 & 0.99 & 0.98 & 0.99^* & 0.98^*\\ 
\text{APOLLO} & 0.98^* & 0.95 & 0.95 & 0.98 & 0.99^* & 0.96 & 1.00^* & 0.99^* & 0.99^* & 0.98^*\\ 
\end{array}
```

**Table 2: OOD Generalization, No Attack:** AUROC scores when each domain is excluded from training. The top group uses non-RAID datasets for training, while the bottom group trains on a portion of RAID. * denotes the best score.

```math
\begin{array}{c|cc|ccccccccccc} 
& \textbf{MEAN} & \textbf{WORST} & \textbf{ch.gpt} & \textbf{cohere} & \textbf{cohere.ch} & \textbf{gpt2} & \textbf{gpt3} & \textbf{gpt4} & \textbf{llama.ch} & \textbf{mistral} & \textbf{mistral.ch} & \textbf{mpt} & \textbf{mpt.ch}\\ 
\hline 
\text{UAR} & 0.73 & 0.53 & 0.88 & 0.61 & 0.78 & 0.62 & 0.68 & 0.78 & 0.84 & 0.60 & 0.83 & 0.53 & 0.87\\ 
\text{Zippy} & 0.71 & 0.62 & 0.77 & 0.62 & 0.74 & 0.71 & 0.75 & 0.64 & 0.76 & 0.62 & 0.80 & 0.64 & 0.78\\ 
\text{Binoculars} & 0.97 & 0.93 & 1.00^* & 0.97 & 0.98 & 0.94 & 1.00^* & 0.97 & 1.00^* & 0.93 & 1.00^* & 0.95 & 1.00^*\\ 
\hline 
\text{RoBERTa} & 0.99^* & 0.94 & 1.00^* & 0.94 & 0.99^* & 1.00^* & 1.00^* & 1.00^* & 1.00^* & 0.99^* & 1.00^* & 0.95 & 1.00^*\\ 
\text{APOLLO} & 0.99^* & 0.98^* & 1.00^* & 0.98^* & 0.99^* & 0.99 & 1.00^* & 1.00^* & 1.00^* & 0.99^* & 1.00^* & 0.98^* & 1.00^*\\ 
\end{array}
```

**Table 3: OOLLM Generalization, No Attack:** AUROC scores when each LLM is excluded from training. The top group uses non-RAID datasets for training, while the bottom group trains on a portion of RAID. * denotes the best score.

## Generalization With Attacks

As AI detection methods have become more prevalent, attackers have developed strategies to deceive them, often aiming to produce false negatives by making AI-generated documents appear human-written. In this evaluation, we sampled 500,000 documents from the RAID dataset across all settings and conducted out-of-domain (OOD), out-of-LLM (OOLLM), and out-of-attack (OOA) tests. These evaluations measure how well methods generalize to data excluded from their training datasets.

Under attack settings, zero-shot methods, such as Binoculars, show weaker generalization compared to their performance in no-attack scenarios. In OOD and OOLLM tests (Tables 4 and 5), supervised models perform better because they have access to training data that includes all attack strategies, allowing them to adapt effectively. However, zero-shot methods retain an advantage in the worst-case scenarios, as seen in the "WORST" column.

The OOA evaluation (Table 6) measures how well models generalize to unseen attacks after being trained on all others. While supervised methods exhibit higher expected performance (MEAN column), their robustness varies significantly across attack types (WORST column). Binoculars performs better in worst-case scenarios, but APOLLO surpasses both zero-shot and supervised methods by combining their strengths.

Across all evaluations, APOLLO demonstrates superior generalization by leveraging the robustness of zero-shot methods for worst-case scenarios and the adaptability of supervised methods for expected performance. This makes APOLLO the most generalizable and reliable method for detecting AI-generated content under diverse attack settings.


```math
\begin{array}{c|cc|cccccccc} 
& \textbf{MEAN} & \textbf{WORST} & \textbf{Abstracts} & \textbf{Books} & \textbf{News} & \textbf{Poetry} & \textbf{Recipes} & \textbf{Reddit} & \textbf{Reviews} & \textbf{Wiki}\\ 
\hline 
\text{UAR} & 0.70 & 0.58 & 0.86 & 0.61 & 0.75 & 0.58 & 0.66 & 0.75 & 0.75 & 0.61\\ 
\text{Zippy} & 0.70 & 0.64 & 0.73 & 0.76 & 0.71 & 0.75 & 0.64 & 0.68 & 0.69 & 0.65\\ 
\text{Binoculars} & 0.82 & 0.79 & 0.83 & 0.84 & 0.80 & 0.84 & 0.79 & 0.83 & 0.82 & 0.80\\ 
\hline 
\text{RoBERTa} & 0.86 & 0.45 & 0.94 & 0.87 & 0.92^* & 0.90 & 0.97^* & 0.96^* & 0.45 & 0.90^*\\ 
\text{APOLLO} & 0.91^* & 0.84^* & 0.95^* & 0.90 & 0.90 & 0.90 & 0.96 & 0.96^* & 0.84 & 0.88\\ 
\end{array}
```
**Table 4: OOD Generalization with Attacks**: AUROC scores when each domain is excluded from training. The top group trains on non-RAID datasets, while the bottom group trains on portions of RAID. * denotes the best score.

```math
\begin{array}{c|cc|ccccccccccc} 
& \textbf{MEAN} & \textbf{WORST} & \textbf{ch.gpt} & \textbf{cohere} & \textbf{cohere.ch} & \textbf{gpt2} & \textbf{gpt3} & \textbf{gpt4} & \textbf{llama.ch} & \textbf{mistral} & \textbf{mistral.ch} & \textbf{mpt} & \textbf{mpt.ch}\\ 
\hline 
\text{UAR} & 0.69 & 0.59 & 0.77 & 0.59 & 0.70 & 0.64 & 0.61 & 0.72 & 0.73 & 0.67 & 0.73 & 0.71 & 0.75\\ 
\text{Zippy} & 0.67 & 0.58 & 0.74 & 0.63 & 0.72 & 0.69 & 0.75 & 0.67 & 0.67 & 0.60 & 0.71 & 0.58 & 0.64\\ 
\text{Binoculars} & 0.85 & 0.57 & 0.95 & 0.91 & 0.93 & 0.77 & 0.97 & 0.90 & 0.95 & 0.70 & 0.91 & 0.57 & 0.75\\ 
\hline 
\text{RoBERTa} & 0.97^* & 0.94^* & 0.99^* & 0.94^* & 0.97^* & 0.97^* & 0.98 & 0.98 & 0.99^* & 0.96^* & 0.99^* & 0.95 & 0.97^*\\ 
\text{APOLLO} & 0.97^* & 0.94^* & 0.99^* & 0.94^* & 0.97^* & 0.97^* & 0.99^* & 0.99^* & 0.99^* & 0.95 & 0.99^* & 0.96^* & 0.97^*\\ 
\end{array}
```
**Table 5: OOLLM Generalization with Attacks**: AUROC scores when each LLM is excluded from training. The top group trains on non-RAID datasets, while the bottom group trains on portions of RAID. * denotes the best score.


```math
\begin{array}{c|cc|cccccccccccc} 
& \textbf{MEAN} & \textbf{WORST} & \textbf{Alternative Spelling} & \textbf{Article Deletion} & \textbf{Homoglyph} & \textbf{Insert Paragraphs} & \textbf{None} & \textbf{Number} & \textbf{Paraphrase} & \textbf{Perplexity Misspelling} & \textbf{Synonym} & \textbf{Upper-Lower} & \textbf{Whitespace} & \textbf{Zero-Width Space}\\ 
\hline 
\text{UAR} & 0.69 & 0.42 & 0.77 & 0.78 & 0.43 & 0.78 & 0.79 & 0.77 & 0.55 & 0.77 & 0.77 & 0.73 & 0.70 & 0.42\\ 
\text{Zippy} & 0.68 & 0.61 & 0.69 & 0.69 & 0.61 & 0.69 & 0.69 & 0.68 & 0.66 & 0.69 & 0.69 & 0.69 & 0.69 & 0.70^*\\ 
\text{Binoculars} & 0.81 & 0.65 & 0.85 & 0.86 & 0.67 & 0.83 & 0.85 & 0.85 & 0.79 & 0.85 & 0.80 & 0.85 & 0.84 & 0.65\\ 
\hline 
\text{RoBERTa} & 0.92 & 0.49 & 1.00^* & 1.00^* & 0.93^* & 0.85 & 1.00^* & 1.00^* & 0.81 & 1.00^* & 1.00^* & 1.00^* & 1.00^* & 0.49\\ 
\text{APOLLO} & 0.94^* & 0.66^* & 1.00^* & 0.99 & 0.89 & 0.96^* & 1.00^* & 1.00^* & 0.87^* & 1.00^* & 0.99 & 1.00^* & 0.99 & 0.66\\ 
\end{array}
```
**Table 6: OOA Generalization with Attacks**: AUROC scores when each attack type is excluded from training. The top group trains on non-RAID datasets, while the bottom group trains on portions of RAID. * denotes the best score.

## Evaluation on Non-RAID Data

The RAID dataset is an excellent benchmark for studying the out-of-distribution strengths of detectors. However, in this evaluation, we move beyond RAID to assess detectors on external datasets, including open-source and manually created datasets, as discussed earlier. These datasets incorporate the most recent LLMs, new domains, and diverse contexts, making them more representative of real-world scenarios users are likely to encounter.

To ensure consistent evaluation, we tuned the detectors using a sample from the common-crawl dataset "RedPajamas" [[6]](#reference-6), assuming it to be 100% human-generated. Detectors were adjusted to achieve a 5% false positive rate (FPR) on this dataset. 

Table 8 summarizes the accuracy results across various evaluation datasets. Among the tested models, APOLLO demonstrated robust performance across datasets, leveraging its ensemble design to outperform individual detectors in most cases. Binoculars maintained strong results in specific scenarios, particularly with VicGalle's Alpaca-GPT4 dataset, but APOLLO provided a more balanced and generalizable approach.

```math
\begin{array}{c|ccccc} 
& \textbf{Manual} & \textbf{PawanKRD\_GPT4o} & \textbf{Anthropic\_Persuasive} & \textbf{Red\_Pajamas} & \textbf{VicGalle\_Alpaca.GPT4}\\ 
\hline 
\text{Binoculars} & 0.85 & 0.86 & 0.56 & 0.95^* & 0.99^*\\ 
\text{RoBERTa} & 0.87^* & 0.91 & 0.92^* & 0.95^* & 0.93\\ 
\text{APOLLO} & 0.87^* & 0.94^* & 0.90 & 0.95^* & 0.97\\ 
\end{array}
```

**Table 8: Accuracy on Evaluation Sources.** Models are tuned to a 5% FPR rate on the RedPajamas dataset. * denotes the best score in each column.



# References

<a id="reference-1"></a>
[1] Dugan, Liam, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. “RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors.” arXiv, June 10, 2024. http://arxiv.org/abs/2405.07940.

<a id="reference-2"></a>
[2] Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. “Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text.” arXiv, January 22, 2024. http://arxiv.org/abs/2401.12070.

<a id="reference-3"></a>
[3] Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” arXiv, July 26, 2019. http://arxiv.org/abs/1907.11692.

<a id="reference-4"></a>
[4] Soto, Rafael Rivera, Kailin Koch, Aleem Khan, Barry Chen, Marcus Bishop, and Nicholas Andrews. “Few-Shot Detection of Machine-Generated Text Using Style Representations.” arXiv, March 27, 2024. http://arxiv.org/abs/2401.06712.

<a id="reference-5"></a>
[5] https://github.com/thinkst/zippy

<a id="reference-6"></a>
[6] Together Computer. "RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset." April, 2023. https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T

<a id="reference-7"></a>
[7] Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho. "RADAR: Robust AI-Text Detection via Adversarial Learning." arXiv, Oct 24, 2023. https://arxiv.org/abs/2307.03838

<a id="reference-8"></a>
[8] Vivek Verma, Eve Fleisig, Nicholas Tomlin, Dan Klein. "Ghostbuster: Detecting Text Ghostwritten by Large Language Models." arXiv, Apr 5, 2024. https://arxiv.org/abs/2305.15047

<a id="reference-9"></a>
[9] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn. "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature." arXiv, Jul 2023. https://arxiv.org/abs/2301.11305
