import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, setup_chat_format

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import re

seed = 42

data = pd.read_csv('input.tsv',  delimiter='\t', encoding='utf-8')
data.reset_index(drop=True, inplace=True)

system_message = """
"""

dataset_tr, dataset_test = train_test_split(data, test_size=0.2, stratify=data['output'], random_state=seed)

def convert_dataset(data):
    prompt= [{"role": "system", "content": system_message}, 
             {"role": "user", "content": str(data["input"])},
             {"role": "assistant", "content": str(data["output"])}]
    return {'messages': prompt}

dataset_tr = Dataset.from_pandas(dataset_tr)
dataset_t = Dataset.from_pandas(dataset_test)

dataset_tr = dataset_tr.map(convert_dataset, remove_columns=dataset_tr.features)
dataset_t = dataset_t.map(convert_dataset, remove_columns=dataset_t.features)
print(dataset_tr[1]["messages"])
 
#dataset_tr.to_json("train_dataset.json", orient="records")
#dataset_t.to_json("test_dataset.json", orient="records")

# ___________##### TRAIN #####___________________

PATH = ''
FT_PATH = ''

llama_tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
llama_tokenizer.padding_side = "left"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

base_model = AutoModelForCausalLM.from_pretrained(
    PATH,
    quantization_config=quant_config,
    device_map={"": 0}
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model, tokenizer = setup_chat_format(base_model, llama_tokenizer)

peft_parameters = LoraConfig(
    lora_alpha=256,
    lora_dropout=0.1,
    r=256,
    bias="none",
    task_type="CAUSAL_LM"

)

train_params = TrainingArguments(
    output_dir='',
    num_train_epochs=6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1, #default 1
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    #max_grad_norm=0.3, # default to 1.0
    max_grad_norm=1.0, # default to 1.0
    warmup_ratio=0.03, # default to 0.0
    group_by_length=True,
    lr_scheduler_type="linear",
    seed=seed
)

ft = SFTTrainer(
    model=model,
    train_dataset=dataset_tr,
    peft_config=peft_parameters,
    tokenizer=tokenizer,
    args=train_params,
    max_seq_length=512
)

ft.train()
ft.model.save_pretrained(FT_PATH)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# ___________##### TEST #####___________________

merged_model_path = ''

base_model = AutoModelForCausalLM.from_pretrained(PATH, low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.float16, device_map={"": 0})
model = PeftModel.from_pretrained(base_model, FT_PATH)
model = model.merge_and_unload()
model.save_pretrained(merged_model_path)
print('The merged model saved')

fine_tuned_model = AutoModelForCausalLM.from_pretrained(merged_model_path)
print('The merged model uploaded')

tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
print('The tokenizer uploaded')

text_generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
print('The pipeline initialized')

test_sample = dataset_t[0]

generated_texts = []

for idx, data in enumerate(dataset_t):
    prompt = tokenizer.apply_chat_template(data["messages"][:2], tokenize=False, add_generation_prompt=True)

    outputs = text_generator(
        prompt,
        max_new_tokens=1,
        do_sample=True,
        temperature=0.1,
        top_k=10,
        top_p=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f'The prompt for index {idx} generated')
    generated_texts.append(outputs[0]['generated_text'])
print('The test texts generated')

true_labels = dataset_test['output'].tolist()
predicted_labels = [int(text.split()[-1]) if text.split()[-1].isdigit() else -1 for text in generated_texts]

output= {'comment' : dataset_test['input'],
         'label' : dataset_test['output'], 
         'pred' : predicted_labels,
         'full_pred' : generated_texts
                     }
    
output = pd.DataFrame(output)

#Search for the pattern 
pattern = "([^/]+$)"
match = re.search(pattern, PATH)
output.to_csv(f'predictions_{match.group(1)}.tsv')

# ___________##### EVALUATION #####___________________

accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

f_score = f1_score(true_labels, predicted_labels, average='weighted')
print("F1-score:", f_score)

class_labels = [0, 1, 2, 3]
cm = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_labels)
disp.plot()
plt.savefig(f'predictions_{match.group(1)}', dpi=300)
