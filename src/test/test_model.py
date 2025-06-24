from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
import pandas as pd
import nltk
from nltk.corpus import stopwords

def prueba_sumarization(file_path, trainer):
    # 1. Cargar datos y modelo
    if file_path.endswith('.jsonl'):
        dataset = pd.read_json(file_path, lines=True)
    else:
        dataset = pd.read_csv(file_path, header=0)

    rouge = Rouge()

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # 3. Función para generar resúmenes
    def generate_summary(text):
        inputs = trainer.tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(trainer.device)
        
        outputs = trainer.model.generate(
            inputs,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        return trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Se ejecuta la pruba para n ejemplos dentro del range
    num = len(dataset)
    #num = 100
    for i in range(num):
        row = dataset.iloc[i]
        text = row['Article']
        reference_summary = row['Abstract']

        #print("Longitud del texto a la entrada(sin tokenizar): ", len(text))
        # Generar resumen
        inputs = trainer.tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(trainer.device)
        
        #print("Cantidad de tokens a la entrada: ", inputs.shape[1])
        #print("Longitud de texto a la entrada( despues de tokenizar): ", len(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True)))
        #print(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True))

        outputs = trainer.model.generate(
            inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        #print("Cantidad de tokens a la salida: ", outputs.shape[1])
        #print("Longitud de texto a la salida: ", len(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)))
        #print()
        #print(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))

        generated_summary = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if i % 100 == 0:
            print("Resumen generado:\n", generated_summary)
            print("Resumen de referencia:\n", reference_summary)
        
        # Calcular ROUGE
        try:
            scores = rouge.get_scores(generated_summary, reference_summary)[0]
            rouge1_scores.append(scores['rouge-1']['f'])
            rouge2_scores.append(scores['rouge-2']['f'])
            rougeL_scores.append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            #rouge1_scores.append(0.0)
            #rouge2_scores.append(0.0)
            #rougeL_scores.append(0.0)

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



def prueba_tripletas(file_path, trainer, chunk_size):
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
    print("Prueba de modelo: ", trainer)
    inputs = trainer.tokenizer.encode(
            "Hola",
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(trainer.device)
    outputs = trainer.model.generate(
        inputs,
        max_length=100,
        num_beams=4,
        early_stopping=True
    )
    print("Salida: ", trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    prefix = "Given the two elements of a triplet give the object"

    while True:
        try:
            row_source, row_target = next(gen_tripletas)
        except StopIteration:
            print("Tripletas consumidas")
            break
        print(f"Tripleta {i}:\n",row_source + ' ' + row_target)
        #print("Longitud del texto a la entrada(sin tokenizar): ", len(text))
        # Generar resumen
        inputs = trainer.tokenizer.encode(
            prefix + row_source,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(trainer.device)
        
        #print("Cantidad de tokens a la entrada: ", inputs.shape[1])
        #print("Longitud de texto a la entrada( despues de tokenizar): ", len(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True)))
        #print(trainer.tokenizer.decode(inputs[0], skip_special_tokens=True))

        outputs = trainer.model.generate(
            inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
        #print("Cantidad de tokens a la salida: ", outputs.shape[1])
        #print("Longitud de texto a la salida: ", len(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)))
        #print()
        #print(trainer.tokenizer.decode(outputs[0], skip_special_tokens=True))

        generated_triplet = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if i % 10 == 0:
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
        
        if i >= 10:
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