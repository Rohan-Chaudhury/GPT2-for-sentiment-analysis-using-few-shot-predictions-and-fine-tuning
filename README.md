# GPT2-for-sentiment-analysis-using-few-shot-predictions-and-fine-tuning
Code to use GPT2 text generations capabilities for sentiment analysis (on IMDB dataset) via few shot learning and fine-tuning. 

Following the paper "Language models are unsupervised multitask learners." (link: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) I used GPT-2 for sentiment analysis using both few shot learning and fine-tuning method on IMDB Dataset and achieved:

1. in debugging mode: Fewshot accuracy: 0.90 and in non-debugging mode: Fewshot accuracy: 0.61 for few-shot predictions
2. in debugging mode: Fewshot accuracy: 0.70 and in non-debugging mode: Fewshot accuracy: 0.51 for fine-tuning

Repository also contains code on how to do text generations, calculate perplexity, and visualize the attention outputs. Report contains in detail on how to utilize the code.
