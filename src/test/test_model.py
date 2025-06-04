from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
import pandas as pd

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

        # Generar resumen
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

        generated_summary = trainer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calcular ROUGE
        try:
            scores = rouge.get_scores(generated_summary, reference_summary)[0]
            rouge1_scores.append(scores['rouge-1']['f'])
            rouge2_scores.append(scores['rouge-2']['f'])
            rougeL_scores.append(scores['rouge-l']['f'])
        except Exception as e:
            print(f"Error calculando ROUGE: {str(e)}")
            # Añadir valores cero si hay error
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)

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