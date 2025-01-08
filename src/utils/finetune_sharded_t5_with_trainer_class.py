import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
import evaluate
import os

# Flags to control configurations
USE_8BIT = True  # Set to True for 8-bit quantization, False for 4-bit
INCREASE_LORA_PRECISION = False  # Set to True to use float32 for LoRA parameters

# Initialize ROUGE metric
rouge = evaluate.load('rouge')

def create_insomnia_dataset():
    return [
        {
            "input_text": "Answer this medical question: What is chronic insomnia and how does it differ from acute insomnia?",
            "target_text": "Chronic insomnia is a sleep disorder lasting at least 3 months, with sleep difficulties occurring at least 3 times per week. Unlike acute insomnia, which is brief and often triggered by life events, chronic insomnia is persistent and can significantly impact daily functioning, mental health, and physical well-being."
        },
        {
            "input_text": "Answer this medical question: What are the main risk factors for developing chronic insomnia?",
            "target_text": "Main risk factors include: psychological conditions (anxiety, depression), medical conditions (chronic pain, respiratory disorders), irregular sleep schedules, excessive caffeine or alcohol consumption, certain medications, genetics, age (more common in older adults), and female gender. Lifestyle factors and stress also play significant roles."
        },
        {
            "input_text": "Answer this medical question: How does chronic insomnia affect cardiovascular health?",
            "target_text": "Chronic insomnia increases the risk of cardiovascular problems by elevating blood pressure, heart rate, and stress hormones. It can lead to hypertension, irregular heartbeat, and increased risk of heart disease. The chronic sleep deprivation also affects the body's ability to regulate glucose and inflammation."
        },
        {
            "input_text": "Answer this medical question: What are the common comorbidities associated with chronic insomnia?",
            "target_text": "Common comorbidities include anxiety disorders, depression, diabetes, hypertension, chronic pain conditions, respiratory disorders, and gastrointestinal problems. There's often a bidirectional relationship where these conditions can both cause and be worsened by chronic insomnia."
        },
        {
            "input_text": "Answer this medical question: How is chronic insomnia diagnosed and what criteria are used?",
            "target_text": "Diagnosis involves sleep history evaluation, sleep diaries, and meeting criteria: difficulty initiating/maintaining sleep or early awakening, occurring 3+ nights weekly for 3+ months, causing significant distress/impairment, and adequate sleep opportunity. Physical exams and sleep studies may be used to rule out other conditions."
        },
        {
            "input_text": "Answer this medical question: What are the treatment options for chronic insomnia?",
            "target_text": "Treatment options include cognitive behavioral therapy for insomnia (CBT-I), sleep hygiene education, stimulus control therapy, relaxation techniques, and in some cases, medication. CBT-I is considered first-line treatment as it addresses underlying causes and helps develop healthy sleep habits without medication dependence."
        },
        {
            "input_text": "Answer this medical question: How does chronic insomnia impact cognitive function and mental health?",
            "target_text": "Chronic insomnia impairs attention, memory, decision-making, and reaction time. It increases risk of anxiety and depression, creates emotional instability, and can worsen existing mental health conditions. Long-term sleep deprivation may also contribute to cognitive decline and increased risk of neurodegenerative disorders."
        },
        {
            "input_text": "Answer this medical question: What lifestyle modifications can help manage chronic insomnia?",
            "target_text": "Key lifestyle modifications include maintaining consistent sleep-wake schedules, creating a relaxing bedtime routine, avoiding screens before bed, limiting caffeine and alcohol, regular exercise (but not close to bedtime), stress management techniques, and ensuring a comfortable sleep environment with proper temperature, darkness, and quiet."
        },
        {
            "input_text": "Answer this medical question: How does chronic insomnia affect hormonal balance and metabolism?",
            "target_text": "Chronic insomnia disrupts hormonal balance by affecting cortisol, growth hormone, and melatonin production. It can lead to increased ghrelin (hunger hormone) and decreased leptin (satiety hormone), contributing to weight gain. It also impairs insulin sensitivity and glucose metabolism, increasing diabetes risk."
        },
        {
            "input_text": "Answer this medical question: What are the long-term consequences of untreated chronic insomnia?",
            "target_text": "Untreated chronic insomnia can lead to increased risk of cardiovascular disease, diabetes, obesity, weakened immune system, chronic pain conditions, cognitive decline, mental health disorders, and reduced quality of life. It can also impact work performance, relationships, and increase accident risk due to fatigue."
        }
    ]

def compute_metrics(eval_pred):
    """Compute ROUGE metrics for evaluation"""
    predictions, labels = eval_pred
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels (ignored index) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}

    return {
        'rouge1': round(result['rouge1'], 4),
        'rouge2': round(result['rouge2'], 4),
        'rougeL': round(result['rougeL'], 4),
    }


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    weight_precision = "fp16"
    print("Loading model and tokenizer...")

    # Load the original config
    model_id = "google/t5-11b-ssm-nq"
    tokenizer = T5Tokenizer.from_pretrained(model_id)

    config = T5Config.from_pretrained(model_id)
    # Set up quantization config
    bnb_config = BitsAndBytesConfig(
    load_in_8bit=USE_8BIT,
    load_in_4bit=not USE_8BIT,
    bnb_4bit_quant_type="nf4" if not USE_8BIT else "fp4",  # Fixed invalid None
    bnb_4bit_use_double_quant=True if not USE_8BIT else False,
    bnb_4bit_compute_dtype=torch.float16 if not USE_8BIT else None  # Valid parameter
    )

    # Load model with quantization
    model = T5ForConditionalGeneration.from_pretrained(
        "iarroyof/t5-11b-ssm-nq-sharded",
        device_map="auto",
        max_memory={0: "40GB", 1: "40GB", "cpu": "30GB"},
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    # Test zero-shot performance
    print("\nTesting zero-shot performance...")
    model.eval()
    test_questions = [
        "What is chronic insomnia?",
        "How does chronic insomnia affect overall health?",
        "What treatments are available for chronic insomnia?"
    ]

    with torch.no_grad():
        for question in test_questions:
            inputs = tokenizer(
                f"Answer this medical question: {question}",
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.7
            )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nQ: {question}")
            print(f"A: {answer}")

    # Prepare for fine-tuning
    print("\nPreparing for fine-tuning...")

    # Create dataset
    dataset_dict = {
        "input_text": [],
        "target_text": []
    }
    for item in create_insomnia_dataset():
        dataset_dict["input_text"].append(item["input_text"])
        dataset_dict["target_text"].append(item["target_text"])

    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset = dataset.train_test_split(test_size=0.2)

    # Tokenize datasets
    tokenized_train = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        batched=True
    )

    tokenized_eval = dataset["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=dataset["test"].column_names,
        batched=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA configuration
    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Optionally increase LoRA precision
    if INCREASE_LORA_PRECISION:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.data = param.data.float()  # Convert to float32
                param.requires_grad = True  # Ensure trainability

    # Verify trainable parameters
    trainable_params = 0
    non_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable: {name} - {param.shape}")
        else:
            non_trainable_params += param.numel()

    print(f"Total Parameters: {trainable_params + non_trainable_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")

    # Training arguments
    training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,  # Enable mixed precision
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    #report_to=["tensorboard"],
    )
    # Create trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        ),
    compute_metrics=compute_metrics
    )
    # Fine-tune
    print("\nStarting fine-tuning...")
    trainer.train()

    # Save fine-tuned model
###    output_dir = os.path.join("insomnia_model", "final")
###    trainer.save_model(output_dir)
###    tokenizer.save_pretrained(output_dir)

    # Test fine-tuned model
    print("\nTesting fine-tuned model...")
    model.eval()
    with torch.no_grad():
        for question in test_questions:
            inputs = tokenizer(
                f"Answer this medical question: {question}",
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                do_sample=True,
                temperature=0.7
            )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
