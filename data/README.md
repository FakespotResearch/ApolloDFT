# Training Data

## Breakdown

In order to have a wide coverage of prompts, domains and LLMs, we have aggregated and sampled our training data from the following sources:

Open source datasets:
  - https://github.com/botianzhe/CHEAT
  - https://github.com/haok1402/GPT-Sentinel-public
  - https://github.com/xinleihe/MGTBench
  - https://huggingface.co/datasets/turingbench/TuringBench
  - https://amazon-reviews-2023.github.io/
  - https://github.com/vivek3141/ghostbuster-data
  - https://huggingface.co/datasets/Hello-SimpleAI/HC3
  - https://github.com/tizfa/tweepfake_deepfake_text_detection
  - https://raid-bench.xyz/
  - https://huggingface.co/datasets/browndw/human-ai-parallel-corpus
  - https://huggingface.co/datasets/dmitva/human_ai_generated_text
  - https://huggingface.co/datasets/innova-ai/Human-Style-Answers
  - https://huggingface.co/datasets/nothingiisreal/Claude-3-Opus-Instruct-15K
  - https://huggingface.co/datasets/tiiuae/falcon-refinedweb

In addition to above data sources, to fill in some coverage gaps, we have also manually generated data from more recent LLMs by varying prompts and generation configurations:
  - AI generated e-commerce product reviews with GPT-4o
  - AI generated texts from LLMs like GPT-4o, GPT-4o-mini, Gemma 2, Phi 3 & 3.5, Mixtral MoE, Llama 3 & 3.1 & 3.2, Qwen 2.5

After aggregation, we carefully sampled data from different sources to balance prompts, domains and LLM variations as well as the AI:Human label ratio. The down sampling is conducted by selecting resonable amount of data from each data source by at least including all the human written texts, and finally balancing the AI:Human label ratio by sampling human texts from `falcon-refinedweb` data.

The amount of data required depends on the type and the size of the model to train.

## Statistics

The following distribution plots are created from aggregated, unsampled data.

### LLM Distribution

<p align="center">
  <img src=figures/ai_text_model_breakdown_bar.png width="80%">
</p>

- Data from https://huggingface.co/datasets/dmitva/human_ai_generated_text is excluded from the above plot due to missing LLM/model information for AI generated text.

### Word Count Distribution

<p align="center">
  <img src=figures/word_count_breakdown_bar.png width="80%">
</p>

- Included 200K sampled `falcon-refinedweb` data.
