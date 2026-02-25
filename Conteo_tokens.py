import pandas as pd
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    BartTokenizer,
)
from sklearn.utils import shuffle

def prepare_data2(subject, relation, obj, all_start_end=False):
    """Devuelve tuplas con pares de input y tragets"""
    start_token = "[start] "
    end_token = " [end]"

    # Asegurarnos de que todos los datos son strings
    subject = str(subject)
    relation = str(relation)
    obj = str(obj)

    # La lógica de procesado de la relación se mantiene
    #processed_relation = " ".join(re.findall(r"[A-Z][a-z]*", relation)).lower() or relation

    # Construcción de la entrada y el objetivo
    #input_text = f"complete the triplet subject: {subject} relation:{processed_relation} object:"
    #input_text = f"{subject} {processed_relation}" # Descomentar si se activa la logica de procesado de la relacion
    input_text = f"{subject} {relation}"

    if all_start_end:
        input_text = f"{start_token}{input_text}{end_token}"
    
    target_text = obj

    return (input_text, target_text)

def aleatorizarData(train_df, test_df):
    """Funcion para aleatorizar dos data frame en caso de que no esten aleatorizados"""

    train_df = shuffle(train_df, random_state = 42)
    test_df = shuffle(test_df, random_state = 42)
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    
    return train_df, test_df

def preprocesado_datos(data_train, data_test, numdata_train):
    """Funcion que se encarga de la lectura, aleatorizado, procesado  y seleccion de los datos
    para probar resultados o entrenar el modelo.
    El paramtro numdata realiza el contro de los datos de entrenamiento que se estan seleccionando
    se se le pasa 0 se seleccionan todos los datos en otro casi toma el valor que se recibe"""

    train_df = pd.read_csv(data_train, encoding='utf-8')
    test_df = pd.read_csv(data_test, encoding='utf-8')
    print('Train Data: ', data_train)
    print('Test Data: ', data_test)

    if not "shuffle" in data_train:
        print("Aleatorizando")
        train_df, test_df=aleatorizarData(train_df, test_df)

    if 'conceptnet' in data_train or 'triplets' in data_train:
        train_results = train_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    elif 'SNLI' or 'atomic' in data_train:
        train_results = train_df.apply(lambda row: prepare_data2(row['S'], row['R'], row['O']), axis=1)
        
    # El resultado es una "Serie" de pandas, la convertimos a una lista de tuplas
    train_pairs = train_results.tolist()
    numdata_train = int(numdata_train)
    # Si no se especifica que se tomara un numero concreto de datos se toma una sola porcion
    if  numdata_train != 0:
        print("Num train data: ", numdata_train)
        #print("Tipo de dato: ", type(numdata_train))
        train_pairs = train_pairs[0:numdata_train]
    
    if 'conceptnet' in data_test or 'triplets' in data_train:
        test_results = test_df.apply(lambda row: prepare_data2(row['subject'], row['relation'], row['object']), axis=1)
    elif 'SNLI' or 'atomic' in data_test:
        test_results = test_df.apply(lambda row: prepare_data2(row['S'], row['R'], row['O']), axis=1)

    test_pairs = test_results.tolist()

    test_pairs = test_pairs[0:1000]    #No mover esta linea de codigo

    return train_pairs, test_pairs

def main():
    models = ["t5-small", 't5-base', 't5-large', 'facebook/bart-large', 'facebook/bart-base', "Kevincp560/t5-base-finetuned-pubmed"]
    datasets = ['conceptnet', 'atomic', 'SNLI']
    numdata_trains = [10000, 200000, 0]

    for model in models:
        if 'pubmed' in model:
            tokenizer = T5TokenizerFast.from_pretrained(model)
        elif 'bart' in model:
            tokenizer = BartTokenizer.from_pretrained(model)
        else:
            tokenizer = T5Tokenizer.from_pretrained(model)

        for numdata_train in numdata_trains:
            for dataset in datasets:
                trainData=f'data/{dataset}/{dataset}_train.csv'
                testData = f'data/{dataset}/{dataset}_test.csv'

                train_pairs, test_pairs = preprocesado_datos(trainData, testData, numdata_train)
                

                total_tokens = 0

                for x, y in train_pairs:
                    texto_combinado = f"{x} {y}"
                    tokens_totales_tupla = len(tokenizer.encode(texto_combinado))
                    
                    total_tokens += tokens_totales_tupla
                print('-'*30)
                print(f"Total de tokens para:\nRegistros: {numdata_train}\nDataset: {dataset}\nModelo: {model}\nTokens: {total_tokens}")
                print('-'*30)

if __name__ == '__main__':
    main()