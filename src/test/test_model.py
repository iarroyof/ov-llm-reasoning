from transformers import T5ForConditionalGeneration, T5Tokenizer
import evaluate
import pandas as pd

def prueba_sumarization(file_path, trainer):
    # 1. Cargar datos y modelo
    if file_path.endswith('.jsonl'):
        dataset = pd.read_json(file_path, lines=True)
    else:
        dataset = pd.read_csv(file_path, header=0)


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

    # 4. Evaluar con métricas ROUGE
    rouge = evaluate.load('rouge')

    results = []
    # Se ejecuta la pruba para n ejemplos dentro del range
    for example in dataset.select(range(1000)):
        generated_summary = generate_summary(example['Article'])
        reference_summary = example['Abstract']
        
        results.append(rouge.compute(
            predictions=[generated_summary],
            references=[reference_summary]
        ))

    # 5. Calcular promedios
    final_metrics = {
        'rouge1': sum(r['rouge1'] for r in results) / len(results),
        'rouge2': sum(r['rouge2'] for r in results) / len(results),
        'rougeL': sum(r['rougeL'] for r in results) / len(results)
    }

    print("Resultados de evaluación:")
    print(f"ROUGE-1: {final_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {final_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {final_metrics['rougeL']:.4f}")