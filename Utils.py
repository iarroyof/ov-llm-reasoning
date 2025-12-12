import os
import re
import torch
import random
import wandb
import pandas as pd
import numpy as np
import evaluate

from scipy import stats
from evaluate import load
from rouge_score import rouge_scorer
from sklearn.utils import shuffle
from bert_score import score
from sacrebleu.metrics import BLEU


google_bleu = evaluate.load("google_bleu")
bertscore = load("bertscore")
scorer_rou = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def prepare_data2(subject, relation, obj, all_start_end=False):
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
    #input_text = f"complete the triplet subject: {subject} relation:{processed_relation} object:"
    input_text = f"{subject} {processed_relation}"
    if all_start_end:
        input_text = f"{start_token}{input_text}{end_token}"
    
    target_text = obj

    return (input_text, target_text)

def prepare_dataSNLI(premisa, answer, all_start_end=False):
    """Devuelve tuplas con pares de input y tragets"""
    start_token = "[start] "
    end_token = " [end]"

    # Asegurarnos de que todos los datos son strings
    input_text = str(premisa)
    target_text = str(answer)

    if all_start_end:
        input_text = f"{start_token}{input_text}{end_token}"

    return (input_text, target_text)

# Ambas funciones realizan lo mismo sin embargo la numero 2 genera las predicciones por chunks para no sobre cargar la memoria
# de la ram
def generate_text(model, tokenizer, texts, max_len, device):
    """Generate outputs for a list of input strings."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model.generate(**enc,
                              max_length=max_len+10,
                              repetition_penalty=1.3, # Penaliza repeticiones
                              no_repeat_ngram_size=2, # Evita que se repitan pares de palabras
                              num_beams=4, # Usa beam search para buscar mejores secuencias
                              early_stopping=True) # Detiene la generación cuando las 'beams' convergen
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

def aleatorizarData(train_df, test_df):
    """Funcion para aleatorizar dos data frame en caso de que no esten aleatorizados"""

    train_df = shuffle(train_df, random_state = 42)
    test_df = shuffle(test_df, random_state = 42)
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    
    return train_df, test_df

def aleatorizarsingle(df):
    """Funcion para aleatorizar dos data frame en caso de que no esten aleatorizados"""

    df = shuffle(df, random_state = 42)
    df.reset_index(inplace=True, drop=True)
    
    return df

def aleatorizar_column(hold_tgt):
    """Funcion que mezcala una columna, recibe como entrada una lista de objetos y devuleve la lista aleatorizada"""
    random.seed(42)
    tgt_aleatorizadas = list(hold_tgt)
    print("Antes de aleatorizar")
    print(tgt_aleatorizadas[:5])
    random.shuffle(tgt_aleatorizadas)
    print("Despues de aleatorizar")
    print(tgt_aleatorizadas[:5])

    return tgt_aleatorizadas

def calcBert(hold_preds, hold_tgt, run, save, tm, save_to_wandb):
    """Funcion para calcular la metrica berscore para precision, recall y f1score
    
    Returns:
        Bert_F1: Vector de valores f1 score para las tripletas
        
        bertscores (dict): Promedio de las bertsocres en Recal F1 y prec"""

    Bert_Pres, Bert_Recall, Bert_F1 = score(hold_preds, hold_tgt, lang="en", model_type="distilbert-base-uncased")
    #Bert_Pres, Bert_Recall, Bert_F1 = score(hold_preds, list(hold_tgt), lang="en")  #Sin modelo

    print(f"Bert_Score Precision: {Bert_Pres.mean().item():.4f}")
    print(f"Bert_Score Recall: {Bert_Recall.mean().item():.4f}")
    print(f"Bert_Score F1Score: {Bert_F1.mean().item():.4f}")

    bertscores = {
        "Precision": Bert_Pres.mean().item(),
        "Recall": Bert_Recall.mean().item(),
        "F1Score": Bert_F1.mean().item()
    }
    
    if save_to_wandb:
        # Guardando valores de F1 bertscores
        save_on_wandb(run, Bert_F1, tm, 'F1', 'BERTScore')
        # Guardando valores de recall bertscores
        save_on_wandb(run, Bert_Recall, tm, 'Recall', 'BERTScore')
        # Guardando valores de presicion bertscores
        save_on_wandb(run, Bert_Pres, tm, 'Presicion', 'BERTScore')

    return Bert_F1, bertscores

def save_on_wandb(run, list_result, tm, kindmetric, metric):
    my_table = wandb.Table(
        columns=[f"{kindmetric} {metric}"],
        data=[[x] for x in list(list_result)]
    )
    # Log the table to W&B
    run.log({f"{kindmetric} {metric} " + tm: my_table})

def cal_BLUE(gen, refer, inp):
    """La funcion recibe las respuestas generadas por el modelo y las referencias con las cuales se va acomparar"""
    #bleu = BLEU(smooth_method='exp')  # Changed to exp smoothing    # Metrica que solo funciona cuando hay mas de una palabra
    #references = [[t] for t in target_text]  # Proper reference format
    
    results = []
    i = 0
    for word, ref, entrada in zip(gen, refer, inp):
        #bleu_score = bleu.corpus_score([word], [ref]).score        # Metrica que solo funciona cuando hay mas de una palabra
        result = google_bleu.compute(predictions=[word], references=[[ref]])
        results.append(result['google_bleu'])
        if i % 10 == 0:
            print(f'Entrada: {entrada}')
            print(f"Word: {word}\nRef: {ref}\nBlueScore: {result}")
        i+=1

    print('prom_bleu', np.array(results).mean())

    return {'prom_bleu': np.array(results).mean()}

def cal_BLUE_colum(run, gen, refer, tm, save_to_wandb):

    results = []
    i = 0
    for word, ref in zip(gen, refer):
        #bleu_score = bleu.corpus_score([word], [ref]).score        # Metrica que solo funciona cuando hay mas de una palabra
        result = google_bleu.compute(predictions=[word], references=[[ref]])
        results.append(result['google_bleu'])

    if save_to_wandb:
        # Guardando metricas de Bleu
        save_on_wandb(run, np.array(results), tm, 'pr', 'Blue')

    return np.array(results)

def save_colum_csv(title_colum, title_arch, colum, out_dir):
    """Esta funcion esta pensafa para guardar los datos de una columna como los bertsocres en un archivo ya sea csv o tsv"""
    pd.DataFrame({title_colum: colum}).to_csv(os.path.join(out_dir, title_arch+".tsv"), sep="\t", index=False)

    print(f"archivo: {title_arch}.tsv, guardado en: {out_dir}")
    print(f'Ruta: {out_dir}/{title_arch}.tsv')

def p_value(a: np.ndarray, b: np.ndarray):
    """Two-sample independent t-test."""
    _, p_val = stats.ttest_ind(a, b)
    return p_val

def calc_gap(mu1: float, mu2: float, mode: str = "symmetric") -> float:
    """Compute percentage gap between two means."""
    diff = abs(mu1 - mu2)
    if mode == "absolute":
        return diff * 100.0
    # symmetric default: 2·diff/(mu1+mu2)×100
    return diff / ((mu1 + mu2) / 2) * 100.0

def gap_pvalue(bert_f1_score_Shuffle, bert_f1_score):
    """Recibe los f1 scores aleatorizados y no aleatorizados para deviolver un dic con el gap y pvalue"""

    s1 = np.array(bert_f1_score_Shuffle)
    s2 = np.array(bert_f1_score)

    mu1, mu2 = s1.mean(), s2.mean()

    ret = {
        'p_value': p_value(s1, s2),
        'gap': calc_gap(mu1, mu2)
    }
    print(f"p_value: {ret['p_value']}")
    print(f"gap: {ret['gap']}")

    return ret


def eval_holdoutdata(logging, model, tokenizer, cfg, device, run, out_dir, hold_pairs, SBertSr, RScores, BleuScores, bef_after, save_data, save_to_wandb):
    """Funcion que prueba un dataset de validacion
        bef_after(str) : Se encarga de llevar el control para el guardado de datos de si es antes o despues del ajuste fino
        save_data(boolean) : Se encarga de controlar si los datos de las predicciones son guardados o no
    """
    #with open(cfg.holdoutData) as f: hold_lines = f.readlines()
    #hold_pairs = [prep(l) for l in hold_lines]

    #Desempaquetado para la evaluacion, se reciben las enradas y las referencias
    hold_inp, hold_tgt = zip(*hold_pairs) if hold_pairs else ([], [])
    if hold_inp:
        logging.info("Generating hold‑out predictions…")
        
        #Se genera el texto de inferencia del modelo
        hold_preds = generate_text(model, tokenizer, hold_inp, cfg.seqLen, device)
        
        # Se guardan los datos que el modelo predijo con la tripleta y el objeto real del dataset de validacion
        if save_data: 
            pd.DataFrame({"Subj_Pred": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt}).to_csv(
                os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)
            print(f"Datos de validacion(Holdoutdata) guardados en: {out_dir}/test_predictions.tsv")
            #Bert_Pres = bertscore.compute(predictions=hold_preds, references=list(hold_tgt), lang="en")    # solo calcula la presicion
        
        # Se guardan los bertscores NO aleatorizados
        bert_f1_score, SBertSr[bef_after] = calcBert(hold_preds, list(hold_tgt), run = run, save=cfg.save_f1score, tm=f'{bef_after} ajuste tgts no aleatorizadas', save_to_wandb = save_to_wandb)
        f1R_1, f1R_2, f1R_l = calcRouge_F1(run, hold_preds, hold_tgt, tm=f'{bef_after} ajuste tgts no aleatorizadas', save_to_wandb = save_to_wandb)
        RecR_1, RecR_2, RecR_l = calcRouge_recall(run, hold_preds, hold_tgt, tm = f'{bef_after} ajuste tgts no aleatorizadas', save_to_wandb = save_to_wandb)
        PrR_1, PrR_2, PrR_l = calcRouge_presicion(run, hold_preds, hold_tgt, tm = f'{bef_after} ajuste tgts no aleatorizadas', save_to_wandb = save_to_wandb)
        Bleu_score = cal_BLUE_colum(run, hold_preds, hold_tgt, tm=f'{bef_after} ajuste tgts no aleatorizadas', save_to_wandb = save_to_wandb)

        # Se guardan los BertScores que contienen las metricas con los objetos NO aleatorizados en un archivo csv
        if cfg.save_f1score: 
            auxname = f"Obj_No_shuffle_{bef_after}_ajuste"
            if 'pubmed' in cfg.modelName:
                auxname = auxname + '_Pubmed'
            save_colum_csv("F1_BERT_Score", auxname, bert_f1_score, out_dir)
        
        # En este proceso se aleatorizan los objetos verdaderos y se vuelven a comparar contra los inferidos por el modelo
        # Finalmente se guardan los resultados de los bertScores con los objetos reales aleatorizados
        aux_gp = {}
        if cfg.shuffle:
            print("="*50)
            print("Resultados BERTScore ROUGEScore BLEUScore con goldlabes aleatorizadas")
            print("="*50)
            tgt_shuffled = aleatorizar_column(hold_tgt)
            bert_f1_score_Shuffle, _ = calcBert(hold_preds, tgt_shuffled, run = run, save=cfg.save_f1score, tm=f'{bef_after} ajuste tgts aleatorizadas', save_to_wandb = save_to_wandb)
            f1R_1_shuf, f1R_2_shuf, f1R_l_shuff = calcRouge_F1(run, hold_preds, tgt_shuffled, tm=f'{bef_after} ajuste tgts aleatorizadas', save_to_wandb = save_to_wandb)
            RecR_1_shuff, RecR_2_shuff, RecR_l_shuff = calcRouge_recall(run, hold_preds, tgt_shuffled, tm = f'{bef_after} ajuste tgts aleatorizadas', save_to_wandb = save_to_wandb)
            PrR_1_shuff, PrR_2_shuff, PrR_l_shuff = calcRouge_presicion(run, hold_preds, tgt_shuffled, tm = f'{bef_after} ajuste tgts aleatorizadas', save_to_wandb = save_to_wandb)
            Bleu_shuffle = cal_BLUE_colum(run, hold_preds, tgt_shuffled, tm=f'{bef_after} ajuste tgts aleatorizadas', save_to_wandb = save_to_wandb)

            if cfg.save_f1score:
                auxname = f"Obj_shuffle_{bef_after}_ajuste"
                if 'pubmed' in cfg.modelName:
                    auxname = auxname + '_Pubmed'
                save_colum_csv("F1_BERT_Score", auxname, bert_f1_score_Shuffle, out_dir)
                
            # Calcula el p_value y el gap
            print("Gap y p_value de los F1 berscores")
            SBertSr[bef_after].update(gap_pvalue(bert_f1_score_Shuffle, bert_f1_score))
            print("Gap y p_value de los F1 Rouge")
            aux_gp['fR1-1'] = gap_pvalue(f1R_1_shuf, f1R_1)
            aux_gp['fR1-2'] = gap_pvalue(f1R_2_shuf, f1R_2)
            aux_gp['fR1-l'] = gap_pvalue(f1R_l_shuff, f1R_l)
            print("Gap y p_value de los recall Rouge")
            _ = gap_pvalue(RecR_1_shuff, RecR_1)
            _ = gap_pvalue(RecR_2_shuff, RecR_2)
            _ = gap_pvalue(RecR_l_shuff, RecR_l)
            print("Gap y p_value de los presicion Rouge")
            _ = gap_pvalue(PrR_1_shuff, PrR_1)
            _ = gap_pvalue(PrR_2_shuff, PrR_2)
            _ = gap_pvalue(PrR_l_shuff, PrR_l)
            print("Gap y p_value de Bleu")
            _ = gap_pvalue(Bleu_shuffle, Bleu_score)
        
        # Calcula la metrica de Rouge
        RScores[bef_after] = calcRouge(hold_preds, hold_tgt)
        RScores[bef_after].update(aux_gp)
        # Se calcula el promedio de la metrica de bleu
        BleuScores[bef_after] = cal_BLUE(hold_preds, hold_tgt, hold_inp)
        print(f"Bleu Scores:\n{BleuScores}")

    return SBertSr, RScores, BleuScores

def preprocesado_datos(cfg, data_train, data_test, val_data, numdata_train):
    """Funcion que se encarga de la lectura, aleatorizado, procesado  y seleccion de los datos
    para probar resultados o entrenar el modelo.
    El paramtro numdata realiza el contro de los datos de entrenamiento que se estan seleccionando
    se se le pasa 0 se seleccionan todos los datos en otro casi toma el valor que se recibe"""

    # Cargando segunda version de SNLI
    if 'SNLI' in str(data_train):
        dataset = 'SNLI'
        data_train = f'data/{dataset}/{dataset}_train_v2.csv'
        data_test = f'data/{dataset}/{dataset}_test_v2.csv'
        val_data = f'data/{dataset}/{dataset}_val_v2.csv'

    train_df = pd.read_csv(data_train, encoding='utf-8')
    test_df = pd.read_csv(data_test, encoding='utf-8')
    val_df = pd.read_csv(val_data, encoding='utf-8')
    print('Train Data: ', data_train)
    print('Test Data: ', data_test)
    print('holdout Data: ', val_data)

    if not "shuffle" in cfg.trainData:
        print("Aleatorizando")
        train_df, test_df=aleatorizarData(train_df, test_df)
        val_df = aleatorizarsingle(val_df)

    if 'conceptnet' in data_train or 'triplets' in data_train:
        train_results = train_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    elif 'SNLI' in data_train:
        train_results = train_df.apply(lambda row: prepare_dataSNLI(row['premisa'], row['answer']), axis=1)
        
    # El resultado es una "Serie" de pandas, la convertimos a una lista de tuplas
    train_pairs = train_results.tolist()
    numdata_train = int(numdata_train)
    if  numdata_train != 0:
        print("Num train data: ", numdata_train)
        #print("Tipo de dato: ", type(numdata_train))
        train_pairs = train_pairs[0:numdata_train]
    
    if 'conceptnet' in data_test or 'triplets' in data_train:
        test_results = test_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    elif 'SNLI' in data_test:
        test_results = test_df.apply(lambda row: prepare_dataSNLI(row['premisa'], row['answer']), axis=1)

    test_pairs = test_results.tolist()
    hold_pairs = test_pairs[1200:1400]
    test_pairs = test_pairs[0:1000]    #No mover esta linea de codigo

    return train_pairs, test_pairs, hold_pairs

def calcRouge_F1(run, hold_preds, hold_tgt, tm, save_to_wandb):
    """Funcion que calcula el f1score de la metrica Rouge"""

    rouge_scores ={
        "f1-1" : [],
        "f1-2" : [],
        "f1-l" : []
    }
    # Calcular ROUGE
    for i in range(len(hold_preds)):
        try:
            scores = scorer_rou.score(hold_preds[i], list(hold_tgt)[i])
            #scores = rouge.get_scores(hold_preds[i], list(hold_tgt)[i])[0]
            rouge_scores['f1-1'].append(scores['rouge1'].fmeasure)
            rouge_scores['f1-2'].append(scores['rouge2'].fmeasure)
            rouge_scores['f1-l'].append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge_scores['f1-1'].append(0)
            rouge_scores['f1-2'].append(0)
            rouge_scores['f1-l'].append(0)
    if save_to_wandb:
        # Guardando valores de F1 Rougescores-1
        save_on_wandb(run, rouge_scores['f1-1'], tm, 'F1', 'ROUGEScore-1')
        # Guardando valores de F1 Rougescores-2
        save_on_wandb(run, rouge_scores['f1-2'], tm, 'F1', 'ROUGEScore-2')
        # Guardando valores de F1 Rougescores-1
        save_on_wandb(run, rouge_scores['f1-l'], tm, 'F1', 'ROUGEScore-L')

    return rouge_scores['f1-1'], rouge_scores['f1-2'], rouge_scores['f1-l']

def calcRouge_recall(run, hold_preds, hold_tgt, tm, save_to_wandb):
    """Funcion que calcula precision, recall y f1score de la metrica Rouge"""

    rouge_scores ={
        "recall-1" : [],
        "recall-2" : [],
        "recall-l" : []
    }
    # Calcular ROUGE
    for i in range(len(hold_preds)):
        try:
            scores = scorer_rou.score(hold_preds[i], list(hold_tgt)[i])
            #scores = rouge.get_scores(hold_preds[i], list(hold_tgt)[i])[0]
            rouge_scores['recall-1'].append(scores['rouge1'].recall)
            rouge_scores['recall-2'].append(scores['rouge2'].recall)
            rouge_scores['recall-l'].append(scores['rougeL'].recall)
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge_scores['recall-1'].append(0)
            rouge_scores['recall-2'].append(0)
            rouge_scores['recall-l'].append(0)
    
    if save_to_wandb:
        # Guardando valores de recall Rougescores-1
        save_on_wandb(run, rouge_scores['recall-1'], tm, 'recall', 'ROUGEScore-1')
        # Guardando valores de recall Rougescores-2
        save_on_wandb(run, rouge_scores['recall-2'], tm, 'recall', 'ROUGEScore-2')
        # Guardando valores de recall Rougescores-1
        save_on_wandb(run, rouge_scores['recall-l'], tm, 'recall', 'ROUGEScore-L')

    return rouge_scores['recall-1'], rouge_scores['recall-2'], rouge_scores['recall-l']

def calcRouge_presicion(run, hold_preds, hold_tgt, tm, save_to_wandb):
    """Funcion que calcula precision, recall y f1score de la metrica Rouge"""

    rouge_scores ={
        "precision-1": [],
        "precision-2": [],
        "precision-l": []
    }
    # Calcular ROUGE
    for i in range(len(hold_preds)):
        try:
            scores = scorer_rou.score(hold_preds[i], list(hold_tgt)[i])
            #scores = rouge.get_scores(hold_preds[i], list(hold_tgt)[i])[0]
            rouge_scores["precision-1"].append(scores['rouge1'].precision)
            rouge_scores["precision-2"].append(scores['rouge2'].precision)
            rouge_scores["precision-l"].append(scores['rougeL'].precision)
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge_scores["precision-1"].append(0)
            rouge_scores["precision-2"].append(0)
            rouge_scores["precision-l"].append(0)
    if save_to_wandb:
        # Guardando valores de F1 Rougescores-1
        save_on_wandb(run, rouge_scores['precision-1'], tm, 'precision', 'ROUGEScore-1')
        # Guardando valores de F1 Rougescores-2
        save_on_wandb(run, rouge_scores['precision-2'], tm, 'precision', 'ROUGEScore-2')
        # Guardando valores de F1 Rougescores-1
        save_on_wandb(run, rouge_scores['precision-l'], tm, 'precision', 'ROUGEScore-L')

    return rouge_scores['precision-1'], rouge_scores['precision-2'], rouge_scores['precision-l']

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

    return final_metrics

#---------------------------------------------------------------------------------------------------

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

