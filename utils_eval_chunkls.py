import os
import torch
import pandas as pd
import numpy as np
import evaluate

from evaluate import load
from rouge_score import rouge_scorer
from bert_score import score
from datetime import datetime


google_bleu = evaluate.load("google_bleu")
bertscore = load("bertscore")
scorer_rou = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from Utils import (
    prepare_data2,
    generate_text,
    generate_text_2,
    aleatorizar_column
)

from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq
)


def devuelve_valid_data(data):
    """Funcion que se regresa los valores que ya no fueron ocupado para validacion"""

 
    test_df = pd.read_csv(data, encoding='utf-8')
    print('Test Data: ', data)

    
    test_results = test_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)

    test_pairs = test_results.tolist()
    
    #Se toman de las docientas tripletas en adelante
    hold_pairs = test_pairs[1400:]


    return hold_pairs

def load_model(checkpoint_path, modelName):
    """
    Carga el modelo y tokenizador desde un checkpoint
    """
    try:
        if 'pubmed' in modelName:
            tokenizer = T5TokenizerFast.from_pretrained(checkpoint_path)
        elif 'bart' in modelName:
            tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
        else:
            tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
        
        if 'bart' in modelName:
            model = BartForConditionalGeneration.from_pretrained(checkpoint_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        
        # Mover a GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Modo evaluación

        print(f"Modelo cargado desde: {checkpoint_path}")
        print(f"Dispositivo: {device}")
        print(f"Modelo tipo: {type(model).__name__}")
        
        return model, tokenizer, device
    
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None

def eval_for_chunks(model, tokenizer, device, hold_pairs, name_root_file, modelName, seqLen, datasetName, id, particion, save_data=True, biomedic_part = True):
    """Funcion modificada de la original que prueba un dataset de validacion
        save_data(boolean) : Se encarga de controlar si los datos de las predicciones son guardados o no
        biomedic_part(boolean) : Se encarga de controlar si los datos son de caracter biomedico o no
    """

    #Desempaquetado para la evaluacion, se reciben las enradas y las referencias
    hold_inp, hold_tgt = zip(*hold_pairs) if hold_pairs else ([], [])
    if hold_inp:
        
        #Se genera el texto de inferencia del modelo
        if "t5-large" in modelName:
            print("Funcion Generate text 2")
            hold_preds = generate_text_2(model, tokenizer, hold_inp, seqLen, device)
        else:
            print("Funcion Generate text")
            hold_preds = generate_text(model, tokenizer, hold_inp, seqLen, device)
        
        #base_dir = "experimentos_compl"
        base_dir = str(name_root_file)
        # Descomentar esta linea y comentar la siguiente si se activa la hora
        #out_dir = os.path.join(base_dir, timestamp, str(cfg.modelName), str(cfg.datasetName)) # Se Genera el nombre de la carpeta con fecha y hora
        out_dir = os.path.join(base_dir, str(f'{modelName}_{particion}'), str(datasetName)) # Se Genera el nombre de la carpeta con fecha y hora
        if save_data:
            os.makedirs(out_dir, exist_ok=True) # Crea la carpeta si no existe exp/time/modelname/dataset

            if biomedic_part:
                pref_bio = "Bio"
            else:
                pref_bio = "otro"

            pd.DataFrame({"Subject": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt}).to_csv(
                os.path.join(out_dir, f"{pref_bio}_test_predictions_{id}.tsv"), sep="\t", index=False)
            
            print(f"Datos de validacion(Holdoutdata) guardados en: {out_dir}/{pref_bio}_test_predictions_{id}.tsv")
            #Bert_Pres = bertscore.compute(predictions=hold_preds, references=list(hold_tgt), lang="en")    # solo calcula la presicion
        
        # Se guardan los bertscores NO aleatorizados
        calcBert(hold_preds, list(hold_tgt), sh = False, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_F1(hold_preds, hold_tgt, sh = False, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_recall(hold_preds, hold_tgt, sh = False, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_presicion(hold_preds, hold_tgt, sh = False, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        cal_BLUE_colum(hold_preds, hold_tgt, sh = False, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        
        # En este proceso se aleatorizan los objetos verdaderos y se vuelven a comparar contra los inferidos por el modelo
        # Finalmente se guardan los resultados de los bertScores con los objetos reales aleatorizados
        print("="*50)
        print("Resultados BERTScore ROUGEScore BLEUScore con goldlabes aleatorizadas")
        print("="*50)
        tgt_shuffled = aleatorizar_column(hold_tgt) # Se aleatorizan las goldlabels

        calcBert(hold_preds, tgt_shuffled, sh = True, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_F1(hold_preds, tgt_shuffled, sh = True, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_recall(hold_preds, tgt_shuffled, sh  =True, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        calcRouge_presicion(hold_preds, tgt_shuffled, sh  =True, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)
        cal_BLUE_colum(hold_preds, tgt_shuffled, sh  =True, save_data = save_data, hold_inp = hold_inp, out_dir = out_dir, biomedic_part = biomedic_part, id = id)

def calcBert(hold_preds, hold_tgt, sh, save_data, hold_inp, out_dir, biomedic_part, id):
    """Funcion para calcular la metrica berscore para precision, recall y f1score
    
    Returns:
        Bert_F1: Vector de valores f1 score para las tripletas
        
        bertscores (dict): Promedio de las bertsocres en Recal F1 y prec"""

    Bert_Pres, Bert_Recall, Bert_F1 = score(hold_preds, hold_tgt, lang="en", model_type="distilbert-base-uncased")
    #Bert_Pres, Bert_Recall, Bert_F1 = score(hold_preds, list(hold_tgt), lang="en")  #Sin modelo

    print(f"Bert_Score Precision: {Bert_Pres.mean().item():.4f}")
    print(f"Bert_Score Recall: {Bert_Recall.mean().item():.4f}")
    print(f"Bert_Score F1Score: {Bert_F1.mean().item():.4f}")

    if save_data:
        # Guardando valores de F1 bertscores
        save_metrics(Bert_F1, 'F1', 'BERTScore', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de recall bertscores
        save_metrics(Bert_Recall, 'Recall', 'BERTScore', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de presicion bertscores
        save_metrics(Bert_Pres, 'Presicion', 'BERTScore', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)

def save_metrics(list_result, kindmetric, metric, biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh):
    "Funcion encargada de guardar los metricas junto con la emtrada y salida dentro del servidor"

    out_dir = os.path.join(out_dir, str(metric)) # Se Genera el nombre de la carpeta con fecha y hora
    os.makedirs(out_dir, exist_ok=True) # Crea la carpeta si no existe exp/time/modelname/dataset

    if biomedic_part:
        pref_bio = "Bio"
    else:
        pref_bio = 'otro'
    if sh:
        id = f'{id}_tgt_Shuffled'

    pd.DataFrame({"Subject": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt, f"{metric}_{kindmetric}" : list_result}).to_csv(
        os.path.join(out_dir, f"{pref_bio}_{metric}_{kindmetric}_{id}.tsv"), sep="\t", index=False)
    
    print(f"{metric}_{kindmetric} guardadas en: {out_dir}/{pref_bio}_{metric}_{kindmetric}_{id}.tsv")

def cal_BLUE_colum(gen, refer, sh, save_data, hold_inp, out_dir, biomedic_part, id):

    results = []
    i = 0
    for word, ref in zip(gen, refer):
        #bleu_score = bleu.corpus_score([word], [ref]).score        # Metrica que solo funciona cuando hay mas de una palabra
        result = google_bleu.compute(predictions=[word], references=[[ref]])
        results.append(result['google_bleu'])

    if save_data:
        # Guardando valores
        save_metrics(np.array(results), 'Presicion', 'Bleu', biomedic_part, gen, refer, hold_inp, out_dir, id, sh)

def calcRouge_F1(hold_preds, hold_tgt, sh, save_data, hold_inp, out_dir, biomedic_part, id):
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

    if save_data:
        # Guardando valores de F1 bertscores
        save_metrics(rouge_scores['f1-1'], 'F1', 'ROUGEScore-1', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de recall bertscores
        save_metrics(rouge_scores['f1-2'], 'F1', 'ROUGEScore-2', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de presicion bertscores
        save_metrics(rouge_scores['f1-l'], 'F1', 'ROUGEScore-L', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)

def calcRouge_recall(hold_preds, hold_tgt, sh, save_data, hold_inp, out_dir, biomedic_part, id):
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
    
    if save_data:
        # Guardando valores de F1 bertscores
        save_metrics(rouge_scores['recall-1'], 'Recall', 'ROUGEScore-1', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de recall bertscores
        save_metrics(rouge_scores['recall-2'], 'Recall', 'ROUGEScore-2', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de presicion bertscores
        save_metrics(rouge_scores['recall-l'], 'Recall', 'ROUGEScore-L', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)


def calcRouge_presicion(hold_preds, hold_tgt, sh, save_data, hold_inp, out_dir, biomedic_part, id):
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
    
    if save_data:
        # Guardando valores de F1 bertscores
        save_metrics(rouge_scores['precision-1'], 'Presicion', 'ROUGEScore-1', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de recall bertscores
        save_metrics(rouge_scores['precision-2'], 'Presicion', 'ROUGEScore-2', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
        # Guardando valores de presicion bertscores
        save_metrics(rouge_scores['precision-l'], 'Presicion', 'ROUGEScore-L', biomedic_part, hold_preds, hold_tgt, hold_inp, out_dir, id, sh)
