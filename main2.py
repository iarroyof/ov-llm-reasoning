# Se importan las librerias
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from rouge import Rouge
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.corpus import stopwords


def prueba_tripletas(file_path, model, tokenizer, device, chunk_size):
    # 1. Cargar datos y modelo
     # Determinar si es CSV o JSONL
    if file_path.endswith('.jsonl'):
        reader = pd.read_json(file_path, lines=True, chunksize=chunk_size)
    elif file_path.endswith('.csv'):
        reader = pd.read_csv(file_path ,chunksize=chunk_size, header=0)

    rouge = Rouge()

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    def safe_str(value):
        if pd.isna(value):
            return ""
        return str(value)

    def devuelve_tripletas(reader):
        # Create a set of stop words 
        stop_words = set(stopwords.words('english'))
        filtered_source = []

        for chunk in reader:
            for _,row in chunk.iterrows():
                row_source = safe_str(row.iloc[-3]) + ' ' + safe_str(row.iloc[-2])
                row_target = safe_str(row.iloc[-1])
                # Se aplica un filtado para descartar las oraciones con stopwords
                # Split the sentence into individual words
                words = row_source.split()
                #filtered_source = [word for word in words if word in stop_words]
                if filtered_source:
                    pass
                else:
                    yield row_source, row_target
    

    # Se ejecuta la pruba para n ejemplos dentro del range
    gen_tripletas = devuelve_tripletas(reader)
    #num = 100
    i = 0
    # Prueba de modelo previo
    #print("Prueba de modelo: ", trainer)
    #inputs = trainer.tokenizer.encode(
    #        "Hola",
    #        return_tensors="pt",
    #        max_length=512,
    #        truncation=True
    #    ).to(trainer.device)
    #outputs = trainer.model.generate(
    #    inputs,
    #    max_length=100,
    #    num_beams=4,
    #    early_stopping=True
    #)
    #print("Salida: ", trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    prefix = "Given the two elements of a triplet infer the object: "

    while True:
        try:
            row_source, row_target = next(gen_tripletas)
        except StopIteration:
            print("Tripletas consumidas")
            break
        # Se imprime la tripleta
        #print(f"Tripleta {i}:\n",row_source + ' ' + row_target)
        #print("Longitud del texto a la entrada(sin tokenizar): ", len(text))
        # Generar resumen
        inputs = tokenizer.encode(
            prefix + row_source,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        #print("Cantidad de tokens a la entrada: ", inputs.shape[1])
        #print("Longitud de texto a la entrada( despues de tokenizar): ", len(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True)))
        #print(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True))

        outputs = model.generate(
            inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        #print("Cantidad de tokens a la salida: ", outputs.shape[1])
        #print("Longitud de texto a la salida: ", len(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)))
        #print()
        #print(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))

        generated_triplet = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if i % 100 == 0:
            print("Tripleta generada:\n", generated_triplet)
            print("Tripleta de referencia:\n", row_target)
        
        # Calcular ROUGE
        try:
            scores = rouge.get_scores(generated_triplet, row_target)[0]
            rouge1_scores.append(scores['rouge-1']['f'])
            rouge2_scores.append(scores['rouge-2']['f'])
            rougeL_scores.append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            #rouge1_scores.append(0.0)
            #rouge2_scores.append(0.0)
            #rougeL_scores.append(0.0)
        
        if i >= 1000:
            break
        i += 1
    
    # 5. Calcular promedios
    final_metrics = {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }

    print("Resultados de evaluación:")
    print(f"ROUGE-1: {final_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {final_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {final_metrics['rougeL']:.4f}")



# Declaracion de rutas de datos
file_path_train = '/app/data/triplets_CC0_part1_and_part2_sin_vector.csv'
file_path_test = '/app/data/triplets_CC0_part3_with_header_sin_vector.csv'

# Se declara el nombre del modelo a usar
model_name = "t5-small"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declaraciones
# Se carga el tokenizador 
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Se carga el modelo que se va ajustar
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
# Se crea un batch de ejemplos usand DataCollector que es mas eficiente
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
# Metrica de evaluacion
rouge = Rouge()


print("="*100)
print("Pruebas antes del ajuste")
print("Iniciando pruebas de tripletas con archivo: ", file_path_test)
prueba_tripletas(file_path_test, model, tokenizer, device, 1000)


class IterableJSONLDataset(IterableDataset):

    # Funcion para inicializar parametros
    def __init__(self, file_path, chunk_size, tokenizer, max_samples = 100):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.current_index = 1
        self.source_max_length = 128
        self.target_max_length = 32
        self.prefix = "Given the two elements of a triplet infer the object: "

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
    
    def process_chunk(self, chunk):
        sources = []
        targets = []
        filtered_ob= []
        filtered_suj = []

        for _, row in chunk.iterrows():
            row_source = self.safe_str(row.iloc[-3]) + ' ' + self.safe_str(row.iloc[-2])
            row_target = self.safe_str(row.iloc[-1])
            
            # Filtrado de stopwords sujeto
            words = self.safe_str(row.iloc[-3]).split()
            filtered_suj = [word for word in words if word in stopwords.words('english')]

            # Filtrado de stopwords objeto
            words = self.safe_str(row.iloc[-1]).split()
            filtered_ob = [word for word in words if word in stopwords.words('english')]
            
            if filtered_suj or filtered_ob:
                pass
            else:
                sources.append(self.prefix + " ".join(row_source))
                targets.append(row_target)
                #print(f"Tripleta: {row_source} {row_target}")
        
        # Tokenización por lotes (mucho más eficiente)
        source_encodings = self.tokenizer(
            sources,
            max_length=self.source_max_length,
            truncation=True,
            padding=False,  
            return_tensors=None  
        )
        
        target_encodings = self.tokenizer(
            targets,
            max_length=self.target_max_length,
            truncation=True,
            padding=False,  # El collator se encargará del padding
            return_tensors=None
        )
        
        # Generar ejemplos en el formato adecuado
        for i in range(len(sources)):
            yield {
                "input_ids": source_encodings["input_ids"][i],
                "attention_mask": source_encodings["attention_mask"][i],
                "labels": target_encodings["input_ids"][i]
            }

    def __iter__(self):
        """Generador que lee el archivo por chunks y devuelve muestras tokenizadas"""
        # Determinar si es CSV o JSONL
        #if self.file_path.endswith('.jsonl'):
        #    reader = pd.read_json(self.file_path, lines=True, chunksize=self.chunk_size)
        #elif self.file_path.endswith('.csv'):
        reader = pd.read_csv(self.file_path ,chunksize=self.chunk_size, header=0)

        sample_count = 0
        for chunk in reader:
            for sample in self.process_chunk(chunk):
                if sample_count >= self.max_samples:  # Limitar muestras
                    return
                yield sample
                sample_count += 1

chunk_size = 1000
muestras = 10000
train_dataset = IterableJSONLDataset(file_path_train, chunk_size, tokenizer, muestras)
test_dataset = IterableJSONLDataset(file_path_test, chunk_size, tokenizer, muestras)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.get_scores(decoded_preds, decoded_labels, True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    gen_len = np.mean(prediction_lens)

    # Obtener paso y época actual
    step = trainer.state.global_step
    epoch = trainer.state.epoch

    # Extraer métricas detalladas
    rouge1_f1 = result["rouge-1"]["f"]
    rouge1_p = result["rouge-1"]["p"]
    rouge1_r = result["rouge-1"]["r"]
    rouge2_f1 = result["rouge-2"]["f"]
    rougeL_f1 = result["rouge-l"]["f"]
    
    # Imprimir en formato legible
    print(f"\n=== Paso {step} | Época {epoch:.1f} ===")
    print(f"ROUGE-1 F1: {rouge1_f1:.4f} (P: {rouge1_p:.4f}, R: {rouge1_r:.4f})")
    print(f"ROUGE-2 F1: {rouge2_f1:.4f}")
    print(f"ROUGE-L F1: {rougeL_f1:.4f}")
    print(f"Long. Promedio: {gen_len:.2f} tokens")
    
    # Ejemplo de generación
    print("\nEjemplo de generación:")
    print(f"Referencia: {decoded_labels[0]}")
    print(f"Predicción: {decoded_preds[0]}")
    print("="*50)

    return {
        "rouge1_f1": rouge1_f1,
        "rouge1_p": rouge1_p,
        "rouge1_r": rouge1_r,
        "rouge2_f1": rouge2_f1,
        "rougeL_f1": rougeL_f1,
        "gen_len": gen_len
    }

"""
# Se declaran los argumentos del ajuste
training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_final",
    evaluation_strategy="steps",
    eval_steps=1250,                               # Evaluar cada 5 pasos
    logging_steps=10,                            # Metricas cada paso
    max_steps=3750,                               # Máximo 20 pasos (100 muestras / batch_size=5 → 20 pasos)
    save_steps=20,                              # Guarda las modificacione al final 
    learning_rate=2e-5,
    per_device_train_batch_size=8,              # Batch más pequeño → más pasos/métricas
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    fp16=False,                                 #change to bf16=True for XPU
    push_to_hub=False,
)
"""
# Entrenamiento por epocas
training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_final",
    evaluation_strategy="epoch",
    num_train_epochs=3,                         # Entrenar por 3 épocas completas
    per_device_train_batch_size=8,              # Batch más pequeño → más pasos/métricas
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    predict_with_generate=True,                 # Usado para metricas ROUGE
    logging_steps=50,                            # Metricas cada paso
    fp16=False,                                 #change to bf16=True for XPU
    push_to_hub=False,
)

# Se pasan los parametros al trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Se incia el entrenamientno
trainer.train()


print("="*100)
print("Pruebas despues del entrenamiento")
print("Iniciando pruebas de tripletas con archivo: ", file_path_test)
prueba_tripletas(file_path_test, model, tokenizer, device, 1000)