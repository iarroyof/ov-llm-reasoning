# Se importan las librerias
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Se declara el nombre del modelo a usar
model_name = "google-t5/t5-small"

# Declaraciones
# Se carga el tokenizador 
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Se crea un batch de ejemplos usand DataCollector que es mas eficiente
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
# Metrica de evaluacion
rouge = evaluate.load("rouge")
# Se carga el modelo que se va ajustar
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Declaracion de rutas de datos
file_path_train = '/app/data/triplets_CC0_part1_and_part2_sin_vector.csv',
file_path_test = '/app/data/triplets_CC0_part3_with_header_sin_vector.csv',



class IterableJSONLDataset(IterableDataset):

    # Funcion para inicializar parametros
    def __init__(self, file_path, chunk_size, tokenizer):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.current_index = 1

    def __len__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            if self.file_path.endswith('.csv'):
                return total_lines - 1
            return total_lines
    
    def safe_str(self, value):
            if pd.isna(value):
                return ""
            return str(value)
    
    def devuelve_tripletas(self, reader):
        # Create a set of stop words 
        stop_words = set(stopwords.words('english')) 
        filtered_source = []

        for chunk in reader:
            for _,row in chunk.iterrows():
                row_source = self.safe_str(row.iloc[-3]) + ' ' + self.safe_str(row.iloc[-2])
                row_target = self.safe_str(row.iloc[-1])
                # Se aplica un filtado para descartar las oraciones con stopwords
                # Split the sentence into individual words
                #words = row_source.split()
                #filtered_source = [word for word in words if word in stop_words]
                if filtered_source:
                    pass
                else:
                    yield row_source, row_target

    
    def tokenizar(self, row_source, row_target):

        prefix = "Given the two elements of a triplet infer the object: "

        source_encodings = self.tokenizer.encode_plus(
            prefix + row_source,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer.encode_plus(
            row_target,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Devolver en el mismo formato que JSONLDataset
        return {
            "source_ids": source_encodings['input_ids'].squeeze(0),
            "source_masks": source_encodings['attention_mask'].squeeze(0),
            "target_ids": target_encodings['input_ids'].squeeze(0)
        }

    def __iter__(self):
        """Generador que lee el archivo por chunks y devuelve muestras tokenizadas"""
        # Determinar si es CSV o JSONL
        if self.file_path.endswith('.jsonl') and not self.mix:
            reader = pd.read_json(self.path_sumarization, lines=True, chunksize=self.chunk_size)
        elif self.file_path.endswith('.csv') and not self.mix:
            reader = pd.read_csv(self.file_path ,chunksize=self.chunk_size, header=0)

        # Se crean los generadores
        print("Creando generador")
        gen_tripletas = self.devuelve_tripletas(reader) if self.mix or self.file_path.endswith('.csv') else None
        print("Generador creado")
        while True:
            try:
                row_source, row_target = next(gen_tripletas)
                #print(f'Tripleta {self.current_index}:\n{row_source}')
                #print("Source: ", row_source)
                yield self.tokenizar(row_source, row_target)
            except StopIteration:
                print("Tripletas consumidas")
                break

            if self.current_index >= 10000:
                print("Se alcanzaron 10 muestras")
                self.current_index = 0
                break
            
            self.current_index +=1

chunk_size = 1000
train_dataset = IterableJSONLDataset(file_path_train, chunk_size, tokenizer)
test_dataset = IterableJSONLDataset(file_path_test, chunk_size, tokenizer)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# Se declaran los argumentos del ajuste
training_args = Seq2SeqTrainingArguments(
    output_dir="./temp_output",
    eval_strategy="steps",
    eval_steps=200,
    max_steps=100000,
    save_steps=10000,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=True,
)
# Se pasan los parametros al trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Se incia el entrenamientno
trainer.train()