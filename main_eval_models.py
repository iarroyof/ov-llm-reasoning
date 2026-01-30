##### Codigo para realizar la evaluacion de varios modelos ya entrenados
##### con un dataset de tipo biomedico y distintas semillas, guardando las metricas

import os
import math
import string
import argparse

def main():
    ap = argparse.ArgumentParser("Evaluacion de modelo con varias semillas")
    ap.add_argument("--modelName", required=True)
    ap.add_argument("--seqLen", type=int, default=50)
    ap.add_argument("--batchSize", type=int, default=50)  # 32
    ap.add_argument("--nEpochs", type=int, default=5)
    ap.add_argument("--resPath", default=os.getcwd())
    ap.add_argument("--description", required=True)
    ap.add_argument("--datasetName", default=dataset)
    ap.add_argument("--shuffle", default=True)              #Parametro que control el aleatorizado de las goldlabes para toma de metricas
    ap.add_argument("--save_f1score", default=False)        #Parametro que controla el exportado de los bertscores en formato tsv
    ap.add_argument("--numTrainData_razon", default=400000)  # Si se colca cero se realiza el entrenamiento con el dataset completo
    ap.add_argument("--numTrainData_biomed", default=10000)
    ap.add_argument("--save_experiment", default='True')
    ap.add_argument("--nameFile", type=str, required=True)
    args = ap.parse_args()

    medical_data_train = 'data/filtered_train_triplets_shuffle.csv'
    medical_data_test = 'data/filtered_test_triplets_shuffle.csv'
    out_dir = os.path.join(cfg.resPath, run.project, run.id)
    

if __name__ == '__main__':
    t5_small = {'10k':{'atomic':'c7j4irjz',
        'conceptnet':'jhag9jeh',
        'SNLI':'3ixdn4af'},

        '200k':{'atomic':'5jmow6sf',
        'conceptnet':'5mwhsd5o',
        'SNLI':'obtanr7r'},

        'full':{'atomic':'dcmgfv43',
        'conceptnet':'ai7kls8o',
        'SNLI':'teszjxmy'}}

    t5_base = {'10k':{'atomic':'g8bm5ii0',
        'conceptnet':'cgd48lr9',
        'SNLI':'yinx5mgn'},

        '200k' : {'atomic':'l3phvzcm',
        'conceptnet':'tkjcv557',
        'SNLI':'ex1e5rsi'},

        'full' : {'atomic':'1v1poitr',
        'conceptnet':'z7laf3c5',
        'SNLI':'cjrugfub'}}

    t5_large = {'10k':{'atomic':'y5tdrtba',
        'conceptnet':'1s8eve3i',
        'SNLI':'tde4shhu'}}
    main(t5_small, t5_base, t5_large)