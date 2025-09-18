import re
import torch

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

# Ambas funciones realizan lo mismo sin embargo la numero 2 genera las predicciones por chunks para no sobre cargar la memoria
# de la ram
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

