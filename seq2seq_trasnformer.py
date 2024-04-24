# T5 Seq2Seq
import wandb
import transformers
from datasets import load_dataset
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader,Dataset,RandomSampler,SequentialSampler
from transformers import T5ForConditionalGeneration,T5Tokenizer,T5PreTrainedModel # SentencePiece library is required to download pretrained t5tokenizer
# Let's try T5TokenizerFast
from transformers.models.t5 import T5TokenizerFast



class CustomDataset(Dataset):
  def __init__(self,dataset,tokenizer,source_len,summ_len):
    self.dataset = dataset 
    self.tokenizer = tokenizer
    self.text_len = source_len
    self.summ_len = summ_len
    self.text = self.dataset['article']
    self.summary = self.dataset['highlights']

  def __len__(self):
    return len(self.text)

  def __getitem__(self,i):
    summary = str(self.summary[i])
    summary = ' '.join(summary.split())
    text = str(self.text[i])
    text = ' '.join(text.split())
    source = self.tokenizer.batch_encode_plus([text],max_length=self.text_len,return_tensors='pt',pad_to_max_length=True) # Each source sequence is encoded and padded to max length in batches
    target = self.tokenizer.batch_encode_plus([summary],max_length=self.summ_len,return_tensors='pt',pad_to_max_length=True) # Each target sequence is encoded and padded to max lenght in batches

    source_ids = source['input_ids'].squeeze()
    source_masks = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_masks = target['attention_mask'].squeeze()

    return {
        'source_ids':source_ids.to(torch.long),
        'source_masks':source_masks.to(torch.long),
        'target_ids':target_ids.to(torch.long),
        'target_masks':target_masks.to(torch.long)
    }

def train(epoch,model,tokenizer,loader,optimizer,device):
  model.train()
  print(loader)
  for step,data in enumerate(loader,0):
    y = data['target_ids'].to(device)
    y_ids = y[:,:-1].contiguous()
    lm_labels = y[:,1:].clone().detach() # requires_grad = False
    lm_labels[y[:,1:]==tokenizer.pad_token_id] = -100
    source_ids = data['source_ids'].to(device)
    masks = data['source_masks'].to(device)
    outputs = model(input_ids = source_ids,attention_mask = masks,decoder_input_ids=y_ids,labels=lm_labels)
    loss  = outputs[0]
    if step%10==0:
      print('Epoch:{} | Loss:{}'.format(epoch,loss))
      wandb.log({'training_loss':loss})
    optimizer.zero_grad()
    loss.backward() # optimize weights through loss backward
    optimizer.step()
    
    
def validation(epoch,tokenizer,model,device,loader):
  model.eval()
  predictions = []
  actual = []
  with torch.no_grad():
    for step,data in enumerate(loader,0):
      ids = data['source_ids'].to(device)
      mask = data['source_masks'].to(device)
      y_id = data['target_ids'].to(device)
      prediction = model.generate(input_ids=ids,attention_mask = mask,num_beams=2,max_length=170,repetition_penalty=2.5,early_stopping=True,length_penalty=1.0)

      # Decode y_id and prediction #
      preds = [tokenizer.decode(p,skip_special_tokens=True,clean_up_tokenization_spaces=False) for p in prediction]
      target = [tokenizer.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False) for t in y_id]

      if step%100==0:
        print('Completed')
      print('predictions',preds)
      print('actual',target)
      predictions.extend(preds)
      actual.extend(target)
  return predictions,actual

def main():
  wandb.init(project='huggingface')
  epochs = 20
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = T5TokenizerFast.from_pretrained('t5-base')

  ## Prepare Dataset ##
  ##  We will use cnn_dailymail summarization dataset for abstractive summarization #
  dataset = load_dataset('cnn_dailymail','3.0.0')

  # As we can observe, dataset is too large so for now we will consider just 8k rows for training
  #  and 4k rows for validation
  train_dataset = dataset['train'][:8000]
  val_dataset = dataset['validation'][:4000]

  #!pip install regex
  import re
  def preprocess(dataset):
    dataset['article'] = [re.sub('\\(.*?\\)','',t) for t in dataset['article']]
    dataset['article'] = [t.replace('--','') for t in dataset['article']]
    return dataset

  train_dataset = preprocess(train_dataset)
  val_dataset = preprocess(val_dataset)
  

  train_dataset = CustomDataset(train_dataset,tokenizer,270,160)
  val_dataset = CustomDataset(val_dataset,tokenizer,270,160)
  
  train_loader = DataLoader(dataset=train_dataset,batch_size=4,shuffle=True,num_workers=0)
  val_loader = DataLoader(dataset = val_dataset,batch_size=2,num_workers=0)

  # Define model
  model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)
  optimizer = Adam(model.parameters(),lr=3e-4,amsgrad=True)
  wandb.watch(model,log='all')
  # Call train function
  for epoch in range(epochs):
    train(epoch,model,tokenizer,train_loader,optimizer,device)

  # Call validation function
  for epoch in range(epochs):
    pred,target = validation(epoch,tokenizer,model,device,val_loader)
    print(pred,target)

main()  