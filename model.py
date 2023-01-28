from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import numpy as np
import torch
from tqdm import tqdm
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling

def get_transformer_model():

	# Feel free to change models if having memory issue
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	model = GPT2LMHeadModel.from_pretrained("gpt2")

	# 'pt' for PyTorch, 'tf' for TensorFlow
	framework = 'pt'

	return TransformerModel(model, tokenizer, framework)


class TransformerModel(object):

	def __init__(self, model, tokenizer, framework='pt'):

		self.model = model
		self.tokenizer = tokenizer
		self.framework = framework

		##### Feel free to add more attributes here if needed #####


	def generate_text(self, prompt, max_new_tokens=10, num_return_sequences=1):
		"""
		The method generates the complementary text for a given starting
		text, i.e., the prompt.

		Args:
			prompt: the starting text as a string
			max_length [optional]: the max length of the generated text

		Return:
			results: the generated text as a string.
		"""

		##### Your code here #####
		# if max_new_tokens == 2:
		# 	# Sentiment analysis
		# 	input_ids = self.tokenizer.encode(prompt, return_tensors=self.framework)
		# 	outputs = self.model.generate(input_ids, max_length=100, num_return_sequences=num_return_sequences)
		# 	results = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

		# else:
		# Text generation and finetuning
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		input_ids = self.tokenizer.encode(prompt, return_tensors=self.framework).to(device)
		# print(input_ids)
		test=False
		if (test):
			if num_return_sequences==1:
				outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)
				results = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]
			else:
				outputs = self.model.generate(input_ids, max_new_tokens = max_new_tokens, num_beams=num_return_sequences,
													num_return_sequences=num_return_sequences)
				results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
		else:
			# if (max_new_tokens==2):
			# 	outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)
			# 	results = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]
			# else:
				if num_return_sequences==1:
					outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences,
												#   do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
												no_repeat_ngram_size = 2, repetition_penalty = 1.5, top_k=100, top_p=0.95, temperature=0.8)
					results = [self.tokenizer.decode(outputs[0], skip_special_tokens=True)]
				else:
					outputs = self.model.generate(input_ids, max_new_tokens = max_new_tokens, num_beams=num_return_sequences,
													num_return_sequences=num_return_sequences,
													no_repeat_ngram_size = 2, repetition_penalty = 1.5, num_beam_groups=num_return_sequences,
													diversity_penalty=1.5, top_k=100, top_p=0.95, temperature=0.8)
					results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


		##### Code done #####
		results = "\n".join(results)

		return results


	def evaluate_ppl(self, dataset):
		"""
		The method for evaluating the perplexity score on given datasets,
		e.g., WikiText-2.

		Args:
			dataset: a `datasets.Dataset' instance from Huggingface

		Return:
			score: A float number. The perplexity score.
		"""

		##### Your code here #####
		torch.cuda.empty_cache()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		score = 0.0
		# print(len(dataset))
		encodings = self.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

		max_length = self.model.config.n_positions

		stride = 128
		seq_len = encodings.input_ids.size(1)

		nlls = []
		prev_end_loc = 0
		for begin_loc in tqdm(range(0, seq_len, stride)):
			end_loc = min(begin_loc + max_length, seq_len)
			trg_len = end_loc - prev_end_loc  
			input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
			target_ids = input_ids.clone()

			target_ids[:, :-trg_len] = -100
			target_ids=target_ids.to(device)
			with torch.no_grad():
				outputs = self.model(input_ids, labels=target_ids)

				neg_log_likelihood = outputs.loss * trg_len

			nlls.append(neg_log_likelihood)

			prev_end_loc = end_loc
			if end_loc == seq_len:
				break

		score = torch.exp(torch.stack(nlls).sum() / end_loc)
		return score


	def get_template(self, doc, lbl):
		##### Write your own template below #####
		template = 'The movie review: \"%s\"\nThe observed sentiment: %s' %(doc, lbl)
		##### Template done #####

		return template


	def fewshot_sentiment(self, trainSet, test_doc):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
		Return:
			prediction: String. The predicted sentiment, 'positive' or 
						'negative'.
		"""

		# prompt = ''
		# for (doc, lbl) in trainSet:
		# 	prompt += self.get_template(doc, lbl)
		# 	prompt += '\n###\n'

		# prompt += self.get_template(test_doc, "")

		# # 'positive'/'negative' plus an EoS token
		# prediction = self.generate_text(prompt, max_new_tokens=2)

		# return prediction.split('\n###\n')[-1]

		temp_pos=[]
		temp_neg=[]
		prompt = ''
		for (doc, lbl) in trainSet:
			temp_prompt = self.get_template(doc, lbl)
			if lbl=='positive':
				temp_pos.append(temp_prompt)
			else:
				temp_neg.append(temp_prompt)
		
		if len(temp_pos)>len(temp_neg):
			for i,j in zip(temp_pos[:len(temp_neg)],temp_neg):
				prompt += i + '\n###\n' + j + '\n###\n'
			for i in temp_pos[len(temp_neg):]:
				prompt += i + '\n###\n'
		else:
			for i,j in zip(temp_pos,temp_neg[:len(temp_pos)]):
				prompt += i + '\n###\n' + j + '\n###\n'
			for i in temp_neg[len(temp_pos):]:
				prompt += i + '\n###\n'
		
		# print(len(test_doc.split('\n')))
		test_doc = "\n".join(test_doc.split('\n')[:int(.3*len(test_doc.split('\n')))])
		# test_doc += "\n" + "\n".join(test_doc.split('\n')[-int(.1*len(test_doc.split('\n'))):])
		prompt += self.get_template(test_doc, "")
		#(neg= 11,24 pos= 3,15)
		# 'positive'/'negative' plus an EoS token
		prediction = self.generate_text(prompt, max_new_tokens=2)

		return prediction.split('\n###\n')[-1]


	def visualize_attention(self, trainSet, test_doc, layer=-1):
		"""
		(Bonus) Visualize how attention works in the fewshot sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
			test_doc: String. The test document.
			layer: Integer. To speficify which attention layer to be visualized.
		Return:
			template: The template input to the language model.
			weights: 1D-Array. The attention score of each token in the template.
					 Values should be in [0,1], normalize if needed.
		"""

		prompt = ''
		for (doc, lbl) in trainSet:
			prompt += self.get_template(doc, lbl)
			prompt += '\n###\n'

		prompt += self.get_template(test_doc, "")

		##### Your code here #####

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		encodings = self.tokenizer(prompt, return_tensors="pt")

		# print(encodings.input_ids[0])
		input_ids = encodings.input_ids.to(device)
		attention_mask = encodings.attention_mask.to(device)

		outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
		attentions = outputs.attentions
		weights = attentions[layer][0].detach().cpu().numpy()

		weights = np.mean(weights, axis=2)
		weights = np.sum(weights, axis=0)
		weights = (weights - weights.min()) / (weights.max() - weights.min())
		template=[]
		for i in encodings.input_ids[0]:
			template.append(self.tokenizer.decode(i, skip_special_tokens=True))
		##### Code done #####
		assert len(template)==len(weights)

		return template, weights


	def finetune(self, trainSet):
		"""
		Taking advantage of the language model to perform sentiment analysis.

		Args:
			trainSet: List of tuples. Each tuple is a pair of (document, label),
					  where `document` is a string of the entire document and 
					  label is either 'positive' or 'negative'
		"""
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		templates = [{"text": self.get_template(doc, lbl)} for doc, lbl in trainSet]
		dataset = Dataset.from_list(templates)
		# Use "left" truncation so that the sentiment is not truncated.
		special_tokens_dict = {'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'}
		num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
		self.model.resize_token_embeddings(len(self.tokenizer))
		map_tokenize = lambda x: self.tokenizer(x['text'], truncation_side='left', padding='max_length', max_length=256, return_tensors='pt')
		dataset = dataset.map(map_tokenize, batched=True)
		# print(dataset[1])
		dataset = dataset.map(map_tokenize, batched=True, remove_columns=["text"])
		dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
		block_size = 128
		padding_token_id = self.tokenizer.pad_token_id
		ending_token_id = self.tokenizer.eos_token_id
		# print(ending_token_id)
		# print(padding_token_id)
		def group_texts(examples):
			# Concatenate all texts.
			# print("Examples:", examples)
			concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
			# print("Concatenated:", concatenated_examples)
			total_length = len(concatenated_examples[list(examples.keys())[0]])
			k=0
			for i in range(len(concatenated_examples['input_ids'])):
				if concatenated_examples['input_ids'][i] == padding_token_id:
					concatenated_examples['input_ids'][i] = ending_token_id
					k+=1
					if k==2:
						break
			# print("Total length:", total_length)
			total_length = total_length
			# print("Total length:", total_length)
			result = {
				k: [t[i : i + block_size] for i in range(0, total_length-block_size)]
				for k, t in concatenated_examples.items()
			}
			# print("Result:", result)
			result["labels"] = result["input_ids"].copy()
			return result

		

		self.model.to(device)
		lm_datasets = dataset.map(
			group_texts,
			batched=True,
			batch_size=4,
			num_proc=4,
		)
		# print(lm_datasets['train'][0])
		# templates = [{"text": self.get_template(doc, lbl)} for doc, lbl in trainSet]
		# dataset = Dataset.from_list(templates)
		# # Use "left" truncation so that the sentiment is not truncated.
		# special_tokens_dict = {'pad_token': '<PAD>'}
		# num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
		# self.model.resize_token_embeddings(len(self.tokenizer))
		# map_tokenize = lambda x: self.tokenizer(x['text'], truncation_side='left', padding='max_length', max_length=1024)
		# dataset = dataset.map(map_tokenize, batched=True)
		# dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
		

		# ##### Your code here #####



		training_args = TrainingArguments(
			output_dir='./results',        
			num_train_epochs=14,              
			per_device_train_batch_size=4,  
			per_device_eval_batch_size=4,   
			warmup_steps=500,                
			weight_decay=0.01,            
			learning_rate=1e-5,
			evaluation_strategy = "epoch",
			seed=42,
		)

		trainer = Trainer(
			model=self.model,                     
			args=training_args,
			train_dataset=lm_datasets['train'],
			eval_dataset=lm_datasets['test'],
		)
		# # Use accuracy as the evaluation metric.
		# def compute_metrics(eval_pred):
		# 	predictions, labels = eval_pred
		# 	predicted_classes = np.argmax(predictions, axis=1)
		# 	return {'accuracy': (predicted_classes == labels).astype(np.float32).mean().item()}
		
		# trainer = Trainer(
		# 	model=self.model,                        
		# 	args=training_args,  
		# 	train_dataset=dataset['train'],         
		# 	eval_dataset=dataset['test'], 
		# 	tokenizer=self.tokenizer,          
		# 	compute_metrics=compute_metrics        
		# )

		trainer.train()

		# ##### Code done #####










