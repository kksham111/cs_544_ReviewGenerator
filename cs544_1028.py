import gzip
import json
import pandas as pd
from datetime import datetime
import accelerate
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from transformers import TrainingArguments, Trainer
import os
from datasets import load_from_disk
from transformers import pipeline
import numpy as np

# date and time
time1 = datetime.now()
print("Current date and time:", time1)

# 读取商品数据集
with gzip.open('meta_All_Beauty.jsonl.gz', 'rt', encoding='utf-8') as f:
    product_data = [json.loads(line) for line in f]

df_product = pd.DataFrame(product_data)

# 读取评论数据集
with gzip.open('All_Beauty.jsonl.gz', 'rt', encoding='utf-8') as f:
    review_data = [json.loads(line) for line in f]

df_review = pd.DataFrame(review_data)


# 基于 'parent_asin' 进行合并
df_merged = pd.merge(df_review, df_product, on='parent_asin', how='inner')

# 创建一个临时 DataFrame，只保留适合去重的列
columns = ['title_x', 'text', 'rating', 'parent_asin', 'user_id', 'timestamp']

# Convert Unhashable Types to String
df_tmp3 = df_merged
columns_to_check = ['images_x', 'features', 'description', 'images_y', 'videos', 'categories', 'details', 'bought_together']

for column in columns_to_check:
    # Convert a list or dictionary to a string representation
    df_tmp3[column] = df_tmp3[column].apply(lambda x: str(x) if isinstance(x, list) or isinstance(x, dict) else x)

# 去重处理
df_cleaned = df_tmp3.drop_duplicates(subset=columns, keep='first')

# 构建 Prompt：结合商品标题、评分和特性
df_cleaned['prompt'] = (
    "Product: " + df_cleaned['title_y'] +
    ". Rating: " + df_cleaned['rating'].astype(str) +
    ". Features: " + df_cleaned['features'].apply(lambda x: ' '.join(x) if isinstance(x, list) else "") +
    ". Description: " + df_cleaned['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else "") +
    ". Please generate a review: "
)

# 将 Prompt 与实际评论组合成训练数据
df_cleaned['train_text'] = df_cleaned['prompt'] + df_cleaned['text'] + "<|endoftext|>"

print()
print(df_cleaned['train_text'][0])
print()


# split the data based on 'parent_asin
# Get unique parent_asin values
unique_parent_asin = df_cleaned['parent_asin'].unique()

# Shuffle the array in place
np.random.seed(42)  # Ensures reproducibility
np.random.shuffle(unique_parent_asin)

# Select 70% unique parent_asin for training, 15% for validation and 15% for test
# num_train = 100
num_train = int(len(unique_parent_asin) * 0.7)
num_val = int(len(unique_parent_asin) * 0.15)

train_asins = unique_parent_asin[:num_train]
val_asins = unique_parent_asin[num_train:num_train + num_val]
test_asins = unique_parent_asin[num_train + num_val:]

# Filter the DataFrame based on asins
train_df = df_cleaned[df_cleaned['parent_asin'].isin(train_asins)]
val_df = df_cleaned[df_cleaned['parent_asin'].isin(val_asins)]
test_df = df_cleaned[df_cleaned['parent_asin'].isin(test_asins)]

# 保存训练数据为文本文件
# 逐条写入数据，避免内存溢出
with open('train_data.txt', 'w', encoding='utf-8') as f:
    for line in train_df['train_text'].values:
        f.write(line + '\n')

with open('validation_data.txt', 'w', encoding='utf-8') as f:
    for line in val_df['train_text'].values:
        f.write(line + '\n')

with open('test_data.txt', 'w', encoding='utf-8') as f:
    for line in test_df['train_text'].values:
        f.write(line + '\n')

time2 = datetime.now()
print("Current date and time: ", time2)
print("Data Prepocessing time usage: ", time2 - time1)

print(accelerate.__version__)  # 检查版本是否 >= 0.26.0
dataset = load_dataset('text', data_files={'train': 'train_data.txt',
                                                'validation': 'validation_data.txt',
                                                'test': 'test_data.txt'})

# 检查是否检测到 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

try:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading model: {e}")

print("a")
# Move the model to GPU if available
model.to(device)
print("b")


def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    # 将填充部分的标签设置为 -100，避免计算损失时干扰
    tokenized_output["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in label]
        for label in tokenized_output["input_ids"]
    ]
    return tokenized_output


# 对数据集进行编码
encoded_dataset = dataset.map(tokenize_function, batched=True)

# Save the transformed dataset
encoded_dataset.save_to_disk("processed_dataset.txt")

loaded_dataset = load_from_disk("processed_dataset.txt")
run_name = f"my_experiment_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

training_args = TrainingArguments(
    output_dir="./gpt2-amazon-review",  # 输出模型目录
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10000,
    max_steps=50000,
    save_total_limit=2,
    logging_dir="./logs",  # 日志目录
    logging_steps=500,
    eval_strategy="steps",  # 或者使用 "steps" 进行评估
    evaluation_strategy="steps",  # Ensure evaluations occur during training
    eval_steps=10000,  # Evaluate every 10000 steps
    load_best_model_at_end=True,  # Optional: load the best model at the end of training
    metric_for_best_model='loss',  # Optional: specify your metric here
    run_name=run_name
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=loaded_dataset['train'],
    eval_dataset=loaded_dataset['validation'],  # Include the validation set here
)

trainer.train()


trainer.save_model("./gpt2-amazon-review")
tokenizer.save_pretrained("./gpt2-amazon-review")
print("d")


# 加载模型
model_path = "./gpt2-amazon-review"
generator = pipeline("text-generation", model=model_path, tokenizer=model_path)

# 生成评论
prompt = "Product: Geesta 12-Pack Premium Activated Charcoal Water Filter Disk. Rating: 5.0. Features:..."
review = generator(prompt, max_length=100, num_return_sequences=1)

print("promt: ")
print("    Product: Geesta 12-Pack Premium Activated Charcoal Water Filter Disk. Rating: 5.0. Features:...")
print(review[0]['generated_text'])


# Evaluate on Test Set
test_results = trainer.evaluate(loaded_dataset['test'])
print(test_results)


# Print time stamp
time3 = datetime.now()
print("Current date and time: ", time3)
print("Model trainning and evaluating time usage: ", time3 - time2)
