#!/usr/bin/env python3
"""
Fine‑tune **T5‑small** on subject–predicate–object (SPO) triples with maximum backward‑compatibility to older 🤗 Transformers versions (no `predict_with_generate`).

Key points
-----------
* Same TSV → (input, target) preprocessing as the original TF‑GRU pipeline.
* W&B tracking + custom “probability of over‑fit” metric.
* Trainer without `predict_with_generate`; we call `model.generate()` manually for validation / hold‑out predictions.
* Script should run even on very old 4.x releases (down to ~4.0).
"""

import os
import re
import math
import string
import argparse
import logging
from functools import partial
from rouge import Rouge
import nltk
from nltk.corpus import stopwords
from evaluate import load
from bert_score import score
from sklearn.utils import shuffle
from rouge_score import rouge_scorer


import torch
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")

bertscore = load("bertscore")
scorer_rou = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def prepare_data(line: str,
                 start_token: str = "[start] ",
                 end_token: str = " [end]",
                 pmid: bool = True,
                 include_labels: bool = False,
                 include_sent: bool = False,
                 all_start_end: bool = True):
    """Convert one TSV row to (input, target) pair."""
    cols = line.rstrip("\n").split("\t")
    if pmid:
        cols.pop(0)
    predicate = " ".join(re.findall(r"[A-Z][a-z]*", cols[1])).lower() or cols[1]
    if not re.match(r"^-?\d+(?:\.\d+)?$", cols[4].strip()):
        extras = []
        i = 4
        while i < len(cols) and not re.match(r"^-?\d+(?:\.\d+)?$", cols[i].strip()):
            extras.append(cols.pop(i))
        cols[3] = " ".join([cols[3]] + extras)
    sample = [cols[0], predicate, cols[2], f"{start_token}{cols[3]}{end_token}", float(cols[4])]
    if include_labels:
        tgt = tuple(sample[-2:])
    else:
        sample.pop(-1)
        tgt = sample[-1]
    if include_sent:
        inp = " ".join([sample[0], sample[2], sample[1]])
    else:
        sample.pop(0)
        inp = " ".join([sample[1], sample[0]])
        if all_start_end:
            inp = f"{start_token}{inp}{end_token}"
    return inp, tgt

def prepare_data2(subject, relation, obj, all_start_end=True):
    """Devuelve tuplas con pares de input y tragets"""
    start_token = "[start] "
    end_token = " [end]"

    # Asegurarnos de que todos los datos son strings
    subject = str(subject)
    relation = str(relation)
    obj = str(obj)

    # La lógica de procesado de la relación se mantiene
    processed_relation = " ".join(re.findall(r"[A-Z][a-z]*", relation)).lower() or relation

    # Construcción de la entrada y el objetivo
    input_text = f"{subject} {processed_relation}"
    if all_start_end:
        input_text = f"{start_token}{input_text}{end_token}"
    
    target_text = obj

    return (input_text, target_text)

class OverfitCallback(TrainerCallback):
    def __init__(self, total_epochs: int, a=6.0, b=4.0, c=-2.0):
        self.total_epochs = total_epochs
        self.a, self.b, self.c = a, b, c
        self.epoch = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        train_loss = metrics.get("loss")
        val_loss   = metrics.get("eval_loss")
        if train_loss is None or val_loss is None:
            return
        rel_gap = (val_loss - train_loss) / max(train_loss, 1e-8)
        epoch_ratio = (self.epoch + 1) / self.total_epochs
        p_overfit = 1 / (1 + math.exp(-(self.a*rel_gap + self.b*epoch_ratio + self.c)))
        wandb.log({"epoch": self.epoch+1, "rel_gap": rel_gap, "epoch_ratio": epoch_ratio,
                   "p_overfit": p_overfit, "train_loss": train_loss, "eval_loss": val_loss})
        self.epoch += 1

def generate_text(model, tokenizer, texts, max_len, device):
    """Generate outputs for a list of input strings."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model.generate(**enc, max_length=max_len+10)
    return tokenizer.batch_decode(outs, skip_special_tokens=True)

def generate_text_2(model, tokenizer, texts, max_len, device, batch_size=8):
    """Generate outputs in batches to avoid OOM errors"""
    model.eval()
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        
        with torch.no_grad():                                  # Reduce memory (disable beam search)
            outs = model.generate(**enc, max_length=max_len+10, num_beams=1)
        
        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
        all_outputs.extend(dec)
        
        # Limpieza explícita de memoria
        del enc, outs
        torch.cuda.empty_cache()
    
    return all_outputs

def aleatorizarData(train_df, val_df):
    """Funcion para aleatorizar dos data frame en caso de que no esten aleatorizados"""

    train_df = shuffle(train_df)
    val_df = shuffle(val_df)
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    
    return train_df, val_df

def calcBert(hold_preds, hold_tgt):
    """Funcion para calcular la metrica berscore para precision, recall y f1score"""

    Bert_Pres, Bert_Recall, Bert_F1 = score(hold_preds, list(hold_tgt), lang="en", model_type="distilbert-base-uncased")

    print(f"Bert_Score Precision: {Bert_Pres.mean().item():.4f}")
    print(f"Bert_Score Recall: {Bert_Recall.mean().item():.4f}")
    print(f"Bert_Score F1Score: {Bert_F1.mean().item():.4f}")

def calcRouge(hold_preds, hold_tgt):
    """Funcion que calcula precision, recall y f1score de la metrica Rouge"""

    rouge_scores ={
        "recall-1" : [],
        "f1-1" : [],
        "precision-1": [],
        "recall-2" : [],
        "f1-2" : [],
        "precision-2": [],
        "recall-l" : [],
        "f1-l" : [],
        "precision-l": []
    }
    # Calcular ROUGE
    for i in range(len(hold_preds)):
        try:
            scores = scorer_rou.score(hold_preds[i], list(hold_tgt)[i])
            #scores = rouge.get_scores(hold_preds[i], list(hold_tgt)[i])[0]
            rouge_scores['recall-1'].append(scores['rouge1'].recall)
            rouge_scores['f1-1'].append(scores['rouge1'].fmeasure)
            rouge_scores["precision-1"].append(scores['rouge1'].precision)
            rouge_scores['recall-2'].append(scores['rouge2'].recall)
            rouge_scores['f1-2'].append(scores['rouge2'].fmeasure)
            rouge_scores["precision-2"].append(scores['rouge2'].precision)
            rouge_scores['recall-l'].append(scores['rougeL'].recall)
            rouge_scores['f1-l'].append(scores['rougeL'].fmeasure)
            rouge_scores["precision-l"].append(scores['rougeL'].precision)
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge_scores['recall-1'].append(0)
            rouge_scores['f1-1'].append(0)
            rouge_scores["precision-1"].append(0)
            rouge_scores['recall-2'].append(0)
            rouge_scores['f1-2'].append(0)
            rouge_scores["precision-2"].append(0)
            rouge_scores['recall-l'].append(0)
            rouge_scores['f1-l'].append(0)
            rouge_scores["precision-l"].append(0)
    # 5. Calcular promedios
    final_metrics = {
        'rouge1-r': sum(rouge_scores['recall-1']) / len(rouge_scores['recall-1']),
        'rouge2-r': sum(rouge_scores['recall-2']) / len(rouge_scores['recall-2']),
        'rougeL-r': sum(rouge_scores['recall-l']) / len(rouge_scores['recall-l']),
        'rouge1-f1': sum(rouge_scores['f1-1']) / len(rouge_scores["f1-1"]),
        'rouge2-f1': sum(rouge_scores['f1-2']) / len(rouge_scores["f1-2"]),
        'rougeL-f1': sum(rouge_scores['f1-l']) / len(rouge_scores["f1-l"]),
        'rouge1-pr': sum(rouge_scores['precision-1']) / len(rouge_scores["precision-1"]),
        'rouge2-pr': sum(rouge_scores['precision-2']) / len(rouge_scores["precision-2"]),
        'rougeL-pr': sum(rouge_scores['precision-l']) / len(rouge_scores["precision-l"])
    }

    print("Resultados de evaluación:")
    print(f"ROUGE-1 Recall: {final_metrics['rouge1-r']:.4f}")
    print(f"ROUGE-2 Recall: {final_metrics['rouge2-r']:.4f}")
    print(f"ROUGE-L Recall: {final_metrics['rougeL-r']:.4f}")
    print(f"ROUGE-1 f1score: {final_metrics['rouge1-f1']:.4f}")
    print(f"ROUGE-2 f1score: {final_metrics['rouge2-f1']:.4f}")
    print(f"ROUGE-L f1score: {final_metrics['rougeL-f1']:.4f}")
    print(f"ROUGE-1 precision: {final_metrics['rouge1-pr']:.4f}")
    print(f"ROUGE-2 precision: {final_metrics['rouge2-pr']:.4f}")
    print(f"ROUGE-L precision: {final_metrics['rougeL-pr']:.4f}")

def main(model_name):
    ap = argparse.ArgumentParser("Fine‑tune T5‑small for SPO generation")
    ap.add_argument("--trainData", default='data/filtered_train_triplets_shuffle.csv')
    ap.add_argument("--testData", default='data/filtered_test_triplets_shuffle.csv')
    #ap.add_argument("--trainData", required=True)
    #ap.add_argument("--testData",required=True)
    ap.add_argument("--holdoutData", default="/app/data/triplets_CC0_part3_with_header_sin_vector.csv") # Si no se requiere sustituir por ""
    ap.add_argument("--modelName", default=model_name)
    ap.add_argument("--seqLen", type=int, default=50)
    ap.add_argument("--batchSize", type=int, default=50)  # 32
    ap.add_argument("--nEpochs", type=int, default=4)
    ap.add_argument("--resPath", default=os.getcwd())
    args = ap.parse_args()

    run = wandb.init(project="t5_spo_generation", config=vars(args))
    cfg = run.config
    out_dir = os.path.join(cfg.resPath, run.project, run.id)
    os.makedirs(out_dir, exist_ok=True)

    # Lectura de archivos tsv junto con la funcion prepare_data
    #with open(cfg.trainData) as f: train_lines = f.readlines()
    #with open(cfg.testData)  as f: val_lines   = f.readlines()
    prep = partial(prepare_data2, all_start_end=True)
    #train_pairs = [prep(l) for l in train_lines]
    #val_pairs   = [prep(l) for l in val_lines]
    
    print("Modelo: ", cfg.modelName)
    # Lectura de archivos csv
    train_df = pd.read_csv(cfg.trainData, encoding='utf-8')
    val_df = pd.read_csv(cfg.testData, encoding='utf-8')
    print('Train Data: ',cfg.trainData)
    print('Test Data: ',cfg.testData)

    #train_df, val_df=aleatorizarData(train_df, val_df)

    train_results = train_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    # El resultado es una "Serie" de pandas, la convertimos a una lista de tuplas
    train_pairs = train_results.tolist()
    train_pairs = train_pairs[0:10000]

    val_results = val_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    val_pairs = val_results.tolist()
    hold_pairs = val_pairs[1200:1400]
    val_pairs = val_pairs[0:1000]

    train_inp, train_tgt = zip(*train_pairs)
    val_inp,   val_tgt   = zip(*val_pairs)

    if 'pubmed' in cfg.modelName:
        tokenizer = T5TokenizerFast.from_pretrained(cfg.modelName)
    else:
        tokenizer = T5Tokenizer.from_pretrained(cfg.modelName)
    model     = T5ForConditionalGeneration.from_pretrained(cfg.modelName)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("="*100)
    print("Probando holdoutdata previo al entrenamiento")

    # Hold‑out predictions
    if cfg.holdoutData and os.path.exists(cfg.holdoutData):
        #with open(cfg.holdoutData) as f: hold_lines = f.readlines()
        #hold_pairs = [prep(l) for l in hold_lines]
        hold_inp, hold_tgt = zip(*hold_pairs) if hold_pairs else ([], [])
        if hold_inp:
            logging.info("Generating hold‑out predictions…")
            hold_preds = generate_text(model, tokenizer, hold_inp, cfg.seqLen, device)
            #pd.DataFrame({"Subj_Pred": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt}).to_csv(
            #    os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)
            #Bert_Pres = bertscore.compute(predictions=hold_preds, references=list(hold_tgt), lang="en")    # solo calcula la presicion
            calcBert(hold_preds, hold_tgt)

            calcRouge(hold_preds, hold_tgt)


    def tok(batch):
        enc = tokenizer(batch["input"], max_length=cfg.seqLen, padding="max_length", truncation=True)
        dec = tokenizer(batch["target"], max_length=cfg.seqLen+1, padding="max_length", truncation=True)
        batch["input_ids"]      = enc.input_ids
        batch["attention_mask"] = enc.attention_mask
        batch["labels"]         = dec.input_ids
        return batch

    ds_train = Dataset.from_dict({"input": train_inp, "target": train_tgt}).map(tok, batched=True, remove_columns=["input","target"])
    ds_val   = Dataset.from_dict({"input": val_inp,   "target": val_tgt}).map(tok,   batched=True, remove_columns=["input","target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.nEpochs,
        per_device_train_batch_size=cfg.batchSize,
        per_device_eval_batch_size=cfg.batchSize,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        adafactor = True,
        optim = "adafactor"
    )

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=ds_train,
                      eval_dataset=ds_val,
                      tokenizer=tokenizer,
                      data_collator=collator,
                      callbacks=[OverfitCallback(cfg.nEpochs)])

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Validation predictions   #Verificar que no se esten acumulando gradientes y revisar si se genero el archivo de predictions.tsv
    # buscar si se puede poner adafactor como optimizador 
    logging.info("Generating validation predictions…")
    val_preds = generate_text_2(model, tokenizer, val_inp, cfg.seqLen, device)
    pd.DataFrame({"Subj_Pred": val_inp, "Obj": val_preds, "Obj_true": val_tgt}).to_csv(
        os.path.join(out_dir, "predictions.tsv"), sep="\t", index=False)

    print("="*100)
    print('Holdoutpairs predictions')
    # Hold‑out predictions
    if cfg.holdoutData and os.path.exists(cfg.holdoutData):
        #with open(cfg.holdoutData) as f: hold_lines = f.readlines()
        #hold_pairs = [prep(l) for l in hold_lines]
        hold_inp, hold_tgt = zip(*hold_pairs) if hold_pairs else ([], [])
        if hold_inp:
            logging.info("Generating hold‑out predictions…")
            hold_preds = generate_text(model, tokenizer, hold_inp, cfg.seqLen, device)
            pd.DataFrame({"Subj_Pred": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt}).to_csv(
                os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)
            #Bert_Pres = bertscore.compute(predictions=hold_preds, references=list(hold_tgt), lang="en")
            calcBert(hold_preds, hold_tgt)

            calcRouge(hold_preds, hold_tgt)

    #if cfg.holdoutData and os.path.exists(cfg.holdoutData):
        #print("="*100)
        #print("Pruebas antes del ajuste")
        #print("Iniciando pruebas de tripletas con archivo: ", cfg.holdoutData)
        #prueba_part_triplets(hold_pairs, model, tokenizer, device)
        #prueba_tripletas(cfg.holdoutData, model, tokenizer, device, 1000)

    wandb.finish()

def prueba_part_triplets(pair, model, tokenizer, device):

    rouge = Rouge()

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

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

    hold_inp, hold_tgt = zip(*pair) if pair else ([], [])

    for hold in hold_inp:
        # Se imprime la tripleta
        #print(f"Tripleta {i}:\n",row_source + ' ' + row_target)
        #print("Longitud del texto a la entrada(sin tokenizar): ", len(text))
        # Generar resumen
        inputs = tokenizer.encode(
            prefix + hold,
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

        if i % 1000 == 0:
            print("Tripleta generada:\n", generated_triplet)
            print("Tripleta de referencia:\n", hold_tgt[i])
        
        # Calcular ROUGE
        try:
            scores = rouge.get_scores(generated_triplet, hold_tgt[i])[0]
            rouge1_scores.append(scores['rouge-1']['f'])
            rouge2_scores.append(scores['rouge-2']['f'])
            rougeL_scores.append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
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

if __name__ == "__main__":
    models = ['t5-base',"Kevincp560/t5-base-finetuned-pubmed"]
    for modelname in models:
        main(modelname)