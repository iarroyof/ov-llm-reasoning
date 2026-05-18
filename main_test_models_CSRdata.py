##### Codigo para realizar la evaluacion de varios modelos ya entrenados
##### Con dataset de tipo test para las tareas de razonamiento de sentido comun (SNLI)
##### guardando las metricas
## Se espera que los pvalues sean muy cercanos a cero por el modelo ya vio estos datos


import os
import argparse

from utils_eval_chunkls import (
    devuelve_valid_data,
    load_model,
    eval_for_chunks
)


def main(models_dic):
    ap = argparse.ArgumentParser("Evaluacion de modelo con varias semillas")
    ap.add_argument("--modelName", required=False)
    ap.add_argument("--seqLen", type=int, default=50)
    ap.add_argument("--batchSize", type=int, default=50)  # 32
    ap.add_argument("--nEpochs", type=int, default=5)
    ap.add_argument("--resPath", default=os.getcwd())
    ap.add_argument("--description", required=True)
    ap.add_argument("--shuffle", default=True)              #Parametro que control el aleatorizado de las goldlabes para toma de metricas
    ap.add_argument("--save_f1score", default=False)        #Parametro que controla el exportado de los bertscores en formato tsv
    ap.add_argument("--nameFile", type=str, required=True)
    args = ap.parse_args()

    print("Descripcion del experimento: ", args.description)
    print("Modelo: ", args.modelName)

    for modelName in models_dic.keys():
        print("Modelo: ", modelName)
        for particion in models_dic[modelName].keys():
            print("Particion: ", particion)
            for dataset in models_dic[modelName][particion].keys():
                print("Dataset: ", dataset)
                id = models_dic[modelName][particion][dataset]
                print("Id: ", id)
                # Creacion de ruta----------------------------------------------
                path = f"t5_spo_generation/{id}/checkpoint-1000"
                # Se carga el modelo y el tokenizador
                print("Iniciando carga del modelo ubicado en: ", path)
                model_temp, tokenizer_temp, device = load_model(path, modelName)
                print("Modelo cargado correctamente!!")

                # Carga de tripletas ----------------------------------
                print("Iniciando carga de tripletas")
                # Lectura de archivos csv
                CSR_data_test = f'data/{dataset}/{dataset}_test.csv'
                # No son necesarios los datos de entrenamiento
                hold_pairs = devuelve_valid_data(CSR_data_test)
                # Se crea la capprtea en donde se guaradaran los datos
                out_dir = os.path.join(args.nameFile, f"{dataset}_{particion}")
                os.makedirs(out_dir, exist_ok=True)

                ############################################################
                # Evaluacion de las tripletas de dataset de CSR
                ############################################################
                print("Iniciando proceso de evaluacion de tripletas")
                print("="*100)
                # Hold‑out predictions
                eval_for_chunks(model_temp, tokenizer_temp, device, hold_pairs, out_dir, modelName, args.seqLen, dataset, id, particion, save_data = True, biomedic_part = False)

if __name__ == '__main__':
    models_dic_1 = {
    't5_small' : {'10k':{'atomic':'c7j4irjz',
        'conceptnet':'jhag9jeh',
        'SNLI':'3ixdn4af'},

        '200k':{'atomic':'5jmow6sf',
        'conceptnet':'5mwhsd5o',
        'SNLI':'obtanr7r'},

        'full':{'atomic':'dcmgfv43',
        'conceptnet':'ai7kls8o',
        'SNLI':'teszjxmy'}},

    't5_base' : {'10k':{'atomic':'g8bm5ii0',
        'conceptnet':'cgd48lr9',
        'SNLI':'yinx5mgn'},

        '200k' : {'atomic':'l3phvzcm',
        'conceptnet':'tkjcv557',
        'SNLI':'ex1e5rsi'},

        'full' : {'atomic':'1v1poitr',
        'conceptnet':'z7laf3c5',
        'SNLI':'cjrugfub'}},

    't5_large' : {'10k':{'atomic':'y5tdrtba',
        'conceptnet':'1s8eve3i',
        'SNLI':'tde4shhu'},

        '200k':{'atomic':'5adpy3g6',
        'conceptnet':'a4k3dyoh',
        'SNLI':'qzcczql1'},

        'full':{'atomic':'6b7urd9p',
        'conceptnet':'0tpxrys5',
        'SNLI':'l6uvbqdz'}},

    'bart_base' : {'10k':{'atomic':'rjvmoabv',
        'conceptnet':'ss2csjy9',
        'SNLI':'bmzps6e6'},

        '200k':{'atomic':'kbs5xkx8',
        'conceptnet':'7dm99ujf',
        'SNLI':'scjr7c1x'},

        'full':{'atomic':'xbxw3n8w',
        'conceptnet':'cz6kfnkc',
        'SNLI':'3xh0d615'}},

    'bart_large' : {'10k':{'atomic':'xgkhotag',
        'conceptnet':'9oi8f5vq',
        'SNLI':'8kzowcaz'},

        '200k':{'atomic':'y7avh2hb',
        'conceptnet':'3bu6i1wx',     ### cva7five   <- Bueno
        'SNLI':'w4aaielu'},

        'full':{'atomic':'rlj3m9o5',
        'conceptnet':'lhasqpo8',
        'SNLI':'un48iuft'}}}
    

    conceptnet_dic = {
        '200k':{
        'conceptnet':'cva7five',     ### cva7five   <- Bueno
        }}

    main(conceptnet_dic)