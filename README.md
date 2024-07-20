# SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Measurement in Academic Papers  


## Overview
<b>SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Measurement in Academic Papers".</b>
The main contributions of this paper are reflected in the following three aspects.

-Firstly, we used the novelty scores from peer reviews as a benchmark to fine-tune current popular pre-trained language models (PLMs) designed for long texts to predict novelty scores. We also validated the effectiveness of using different section structures as input.

-Secondly, we conducted a small-scale test on novelty score prediction using prompt-based methods on LLMs. Furthermore, we analyzed the performance of the LLMs in this task under different chapter combinations, as well as the consistency between generated novelty scores and grounded scores.

-Thirdly, the results indicate that fine-tuned PLMs outperform LLMs in predicting novelty scores, though their performance is not yet satisfactory. Furthermore, our findings suggest that the introduction, results, and discussion sections are more beneficial for automatic novelty score prediction tasks.
## Research overview
The following is the research method of this paper. This figure provides an overview of the methodology used in this study,  which includes Section Structure Identification, Fine tuning PLMs for novelty score prediction, and Generate novelty score prediction using LLM.
![image](https://github.com/user-attachments/assets/40f269d0-09f6-4b22-979d-d32de7e15228)

## Directory structure

<pre>
SC4ANM                                            Root directory
├── llm_predict                                   LLM for novelty score prediction
│   ├── gpt_test.py                               Prompt GPT-3.5 and GPT-4
├── novelty_score_predict                         Train PLM for predict novelty score
│   ├── create_data.py                            Create train, valid and test dataset
│   ├── dataset.py                                Load and process data
│   ├── main.py                                   Train model
│   ├── model.py                                  Model architecture
│   ├── params.py                                 Model Parameters
│   ├── test.py                                   Test trained model on testset
│   ├── test.sh                                   Batch execution testing
│   ├── train.sh                                  Batch execution training
│   ├── util.py                                   Tools for training and testing
├── section_classfication                         Train PLM for section structure identification
│   ├── fintune_PLM                               Fine tuning SciBERT
│   │   ├── dataset.py                            Load and process data
│   │   ├── train.py                              Train model
│   │   ├── params.py                             Model Parameters
│   │   ├── section_extras.py                     Create dataset
│   │   ├── test.py                               Test trained model on testset
│   │   ├── model.py                              Model architecture
│   │   ├── util.py                               Tools for training and testing
│   ├── sc_llama.py                               Method for prompting llama3, identifing main text and using PLM
│   ├── section_classfication.py                  Section classfication on our data
│   ├── novel_dataset.json                        Original dataset
└── README.md
</pre>
## Citation
Please cite the following paper if you use these codes and datasets in your work.

> 
