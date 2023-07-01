import pandas as pd
import time
df = pd.read_csv('SNM_dataset_format_converted.csv', header= None, sep=',,', encoding='utf-8', names=["source", "target"], engine='python')
# df = df.sample(n=100, random_state=42)
# save path
output_predicat_csv = 'predict/多次实验临时储存/Multi_SLG.csv'
# output_save_model = 'outputs/save_model/paperdata/fine-tune_pretrained/shinra_pretrained_best_finetuned'
# PLM load
# plm_load = 'shinra_pretrained_best_finetuned'
plm_load = 'shinra_pretrained_best_finetuned'





model_params={
    "MODEL":"T5_japanese_base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":8,          # training batch size
    "VALID_BATCH_SIZE":8,          # validation batch size
    "TRAIN_EPOCHS":5,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":256,   # max length of target text
    "SEED": 42                     # set seed for reproducibility 
}




df['source'] = df['source'].apply(lambda x: x[1:])
df['target'] = df['target'].apply(lambda x: x[:-1])
#文本统计数据集计算
df_source = df['source'].str.len()
df_target = df['target'].str.len()
df_source.describe()
df_target.describe()

df[:1]

# df["text"] = "summarize: "+df["text"]

df.head()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Importing libraries

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import T5Tokenizer
# from modeling_bart.models.t5.modeling_t5 import T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console=Console(record=True)

def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  # console.print(table)

training_logger = Table(Column("Epoch", justify="center" ), 
                        Column("Steps", justify="center"),
                        Column("Loss", justify="center"), 
                        title="Training Status",pad_edge=False, box=box.ASCII)


# Setting up the device for GPU usage
from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'

class YourDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and 
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.summ_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.summ_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long), 
        'source_mask': source_mask.to(dtype=torch.long), 
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }


loss_funcation = NLLLoss
def train(epoch, tokenizer, model, device, loader, optimizer):

  """
  Function to be called for training with the parameters passed from main function

  """

  model.train()
  start_time=time.time()
  for _,data in enumerate(loader, 0):
    y = data['target_ids'].to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)

    outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = outputs[0]
    loss = loss.mean()
    if _%500==0:
      elapsed_time = (time.time() - start_time)/60 
      training_logger.add_row(str(epoch), str(_), str(loss), str(elapsed_time))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()


def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              # do_sample=True, 
              max_length=400,
              # min_length=3,
              # top_k=50,
              # top_p=0.95
              # num_beams=4,
              # repetition_penalty=2.5,
              # length_penalty=1.0, 
              # early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%100==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def T5Trainer(dataframe, source_text, target_text, model_params ):
  
  """
  T5 trainer

  """

  # Set random seeds and deterministic pytorch for reproducibility


  torch.manual_seed(model_params["SEED"]) # pytorch random seed
  np.random.seed(model_params["SEED"]) # numpy random seed



  torch.backends.cudnn.deterministic = True

  # logging
  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  # tokenzier for encoding the text
  # tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
  tokenizer = T5Tokenizer.from_pretrained(plm_load)
  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
  # Further this model is sent to device (GPU/TPU) for using the hardware.


  device = torch.device('cuda:0') 

  model = T5ForConditionalGeneration.from_pretrained(plm_load)
  # model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
  # model = torch.load(plm_load)
  # device_ids=[0,1]
  # model = torch.nn.DataParallel(model, device_ids=device_ids)

  # model = model.cuda(device=device_ids[0])

  
  model = model.cuda(device=device)
  
  # logging
  console.log(f"[Data]: Reading data...\n")

  # Importing the raw dataset
  dataframe = dataframe[[source_text,target_text]]
  display_df(dataframe.head(2))

  
  # Creation of Dataset and Dataloader
  # Defining the train size. So 80% of the data will be used for training and the rest for validation. 
  train_size = 0.9
  train_dataset=dataframe.sample(frac=train_size,random_state = 123123)
  val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
  train_dataset = train_dataset.reset_index(drop=True)

  console.print(f"FULL Dataset: {dataframe.shape}")
  console.print(f"TRAIN Dataset: {train_dataset.shape}")
  console.print(f"TEST Dataset: {val_dataset.shape}\n")


  # Creating the Training and Validation dataset for further creation of Dataloader
  training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
  val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


  # Defining the parameters for creation of dataloaders
  train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


  val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  training_loader = DataLoader(training_set, **train_params)
  val_loader = DataLoader(val_set, **val_params)


   # Defining the optimizer that will be used to tune the weights of the network in the training session. 
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


  # Training loop
  console.log(f'[Initiating Fine Tuning]...\n')

  for epoch in range(model_params["TRAIN_EPOCHS"]):
      train(epoch, tokenizer, model, device, training_loader, optimizer)
      
  console.log(f"[Saving Model]...\n")
  #Saving the model after training
  # path = os.path.join(output_dir)
  # 单GPU保存模型
  # model.save_pretrained(path)
  # 多GPU保存模型
  # model.module.save_pretrained(path)
  # tokenizer.save_pretrained(path)
  # torch.save(model, os.path.join(output_save_model,"T5-base.pth"))



  # CM model save
  # model.save_pretrained(output_save_model)
  # tokenizer.save_pretrained(output_save_model)

  # evaluating test dataset
  console.log(f"[Initiating Validation]...\n")
  for epoch in range(model_params["VAL_EPOCHS"]):
    predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
    final_df.to_csv(os.path.join(output_predicat_csv))
  
#   console.save_text(os.path.join(output_dir,'logs.txt'))
  
  console.log(f"[Validation Completed.]\n")
#   console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
#   console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'best_max60min10_50_095.csv')}\n""")
#   console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")



# 普通一次实验
# T5Trainer(dataframe=df, source_text="source", target_text="target", model_params=model_params)


# import pandas as pd
# import numpy as np

# # 读取CSV文件
# data = pd.read_csv(output_predicat_csv)

# # 计算每个句子的差异
# differences = np.where(data['Generated Text'] == data['Actual Text'], 0, 1)

# # 计算准确率
# accuracy = 1 - np.mean(differences)

# print(f"Accuracy: {accuracy}")







# for experiment in range(3):
#     print(f"Running experiment {experiment + 1}")

#     T5Trainer(dataframe=df, source_text="source", target_text="target", model_params=model_params)
    
#     # 读取CSV文件
#     import pandas as pd
#     import numpy as np
#     data = pd.read_csv(output_predicat_csv)

#     # 计算每个句子的差异
#     differences = np.where(data['Generated Text'] == data['Actual Text'], 0, 1)

#     # 计算准确率
#     accuracy = 1 - np.mean(differences)


#     print(f"Accuracy for experiment : {accuracy}")



#     # 调用函数计算accuracy
#     print(f"Accuracy for SC or NER task experiment {experiment + 1}: {accuracy}")





# 3次实验
for experiment in range(1):
    print(f"Running experiment {experiment + 1}")

    T5Trainer(dataframe=df, source_text="source", target_text="target", model_params=model_params)
    
    # 读取CSV文件
    import pandas as pd
    import numpy as np
    data = pd.read_csv(output_predicat_csv)

    # 计算每个句子的差异
    differences = np.where(data['Generated Text'] == data['Actual Text'], 0, 1)

    # 计算准确率
    accuracy = 1 - np.mean(differences)

    def compare_text(df):
        prefix_correct = 0
        text_correct = 0
        new_accuracy_correct = 0
        total_rows = len(df)

        for index, row in df.iterrows():
            generated_text = row['Generated Text']
            actual_text = row['Actual Text']
            generated_text = str(generated_text)
            actual_text = str(actual_text)

            if generated_text[:4] == actual_text[:4]:
                prefix_correct += 1

            if generated_text[4:] == actual_text[4:]:
                text_correct += 1

            if generated_text[0] == '<' and generated_text[3] == '>' and generated_text[4:11] == actual_text[4:11]:
                new_accuracy_correct += 1

        prefix_accuracy = prefix_correct / total_rows
        text_accuracy = text_correct / total_rows
        new_accuracy = new_accuracy_correct / total_rows

        return prefix_accuracy, text_accuracy, new_accuracy

    # 调用函数计算accuracy
    prefix_accuracy, text_accuracy, new_accuracy = compare_text(data)
    print(f"SNM Accuracy for SNM task experiment {experiment + 1}: {accuracy}")
    print(f'SC accuracy: {prefix_accuracy:.2%}')
    print(f'NER accuracy: {text_accuracy:.2%}')
    print(f'Format accuracy: {new_accuracy:.2%}')


