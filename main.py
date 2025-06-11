"""
Main training script for neural text-to-text models on reasoning tasks.
Supports various architectures (T5, BART, PEGASUS) with configurable training parameters.
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Type

import torch
import wandb
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.nn.modules import Module
from transformers import PreTrainedTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer


import pandas as pd
from torch.utils.data import Dataset, IterableDataset
import json

# Local imports
from src.trainers import (
    T5ReasoningTrainer, 
    T5LargeReasoningTrainer,
    BartReasoningTrainer,
    PegasusReasoningTrainer,
    BaseNeuralReasoningTrainer  # Fixed class name
)
from src.data import ElasticSearchDataset
from src.utils.memory import log_gpu_memory_usage
from src.utils import ClearCache
from src.utils import es_settings
from src.utils.cache_utils import save_split_cache, load_split_cache
from src.utils.triplet_filter import FilterMethod
from src.utils.gpu_monitor import gpu_wait
from src.test.test_model import prueba_sumarization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
DETAILED_LOG_WB = False

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    source_len: int
    target_len: int
    model_name: str
    epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str
    quantization: Optional[str] = None
    max_memory: Optional[dict] = None

@dataclass
class ElasticSearchConfig:
    """Configuration for ElasticSearch data source."""
    url: str
    index: str
    page_size: int
    n_sentences: int
    n_articles: int
    article_ids_file: str

@dataclass
class LocalDataConfig:
    """Configuration for LocalData source"""
    file_path_train: str
    file_path_test :str
    path_sumarization_test :str
    path_sumarization_train :str
    chunk_size: int
    test_ratio: float = 0.3
    seed: int = 42
    # n_samples: int = 10000

class JSONLDataset(Dataset):

    # Funcion para inicializar parametros
    def __init__(self, Df, tokenizer, source_len, target_len):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.Df = Df
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

    def __len__(self):
        return len(self.Df)

    def __getitem__(self, index):
        """Obtiene un numero n de elementos del chunk."""
        sample_data = self.Df.iloc[index]['Article']
        sample_target = self.Df.iloc[index]['Abstract']
        source_encodings = self.tokenizer.encode_plus(
            sample_data,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer.encode_plus(
            sample_target,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            "source_ids": source_encodings['input_ids'].squeeze(0),
            "source_masks": source_encodings['attention_mask'].squeeze(0),
            "target_ids": target_encodings['input_ids'].squeeze(0)
        }
        #return source_encodings, target_encodings

class IterableJSONLDataset(IterableDataset):

    # Funcion para inicializar parametros
    def __init__(self, file_path, file_path_sumarization, mix, chunk_size, tokenizer, source_len, target_len):
        """
        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of lines to read at a time.
        """
        self.file_path = file_path
        self.path_sumarization = file_path_sumarization
        self.mix = mix
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.train = None
        self.test = None
        self.current_index = 1

    def __len__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            if self.file_path.endswith('.csv'):
                return total_lines - 1
            return total_lines
    
    def safe_str(self, value):
            if pd.isna(value):
                return ""
            return str(value)
    
    def devuelve_tripletas(self, reader):
        for chunk in reader:
            for _,row in chunk.iterrows():
                row_source = self.safe_str(row.iloc[-3]) + ' ' + self.safe_str(row.iloc[-2])
                row_target = self.safe_str(row.iloc[-1])

                yield row_source, row_target


    def devuelve_resumenes(self, reader):
        for chunk in reader:
            for _, row in chunk.iterrows():
                sumarzation_text = self.safe_str(row['Article'])
                abstract_text = self.safe_str(row['Abstract'])
                print("Article\n", sumarzation_text)
                print("Abstract\n", abstract_text)
                yield "summarize: "+sumarzation_text, abstract_text
    
    def tokenizar(self, row_source, row_target):
        source_encodings = self.tokenizer.encode_plus(
            row_source,
            max_length=self.source_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encodings = self.tokenizer.encode_plus(
            row_target,
            max_length=self.target_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Devolver en el mismo formato que JSONLDataset
        return {
            "source_ids": source_encodings['input_ids'].squeeze(0),
            "source_masks": source_encodings['attention_mask'].squeeze(0),
            "target_ids": target_encodings['input_ids'].squeeze(0)
        }

    def __iter__(self):
        """Generador que lee el archivo por chunks y devuelve muestras tokenizadas"""
        # Determinar si es CSV o JSONL
        if self.file_path.endswith('.jsonl') and not self.mix:
            readerjson = pd.read_json(self.path_sumarization, lines=True, chunksize=self.chunk_size)
        elif self.file_path.endswith('.csv') and not self.mix:
            reader = pd.read_csv(self.file_path ,chunksize=self.chunk_size, header=0)
        elif self.mix:
            print("Iniciando la lectura de archivos")
            readerjson = pd.read_json(self.path_sumarization, lines=True, chunksize=self.chunk_size)
            reader = pd.read_csv(self.file_path ,chunksize=self.chunk_size, header=0)
            print("Lectura de archivos completada")
        
        # Se crean los generadores
        print("Creando generadores")
        gen_resumenes = self.devuelve_resumenes(readerjson) if self.mix or self.file_path.endswith('.jsonl') else None
        gen_tripletas = self.devuelve_tripletas(reader) if self.mix or self.file_path.endswith('.csv') else None
        print("Generadores creados")
        

        # Se crea condicional para determinar la mezcla de datos
        if self.mix:
            while True:
                # Obtiene resumen
                if self.current_index == 1:
                    row_source, row_target = next(gen_resumenes)
                    yield self.tokenizar(row_source, row_target)

                elif self.chunk_size % 64 == 0:
                    row_source, row_target = next(gen_tripletas)
                    yield self.tokenizar(row_source, row_target)
                    if self.current_index == 64:
                        self.current_index = 0
                
                self.current_index +=1
        elif not self.mix:
            generator = gen_tripletas if self.file_path.endswith('.jsonl') else gen_tripletas
            for row_source, row_target in generator:
                yield self.tokenizar(row_source, row_target)


def t5_collate_fn(batch):
    """Función para agrupar muestras en lotes"""
    return {
        "source_ids": torch.stack([item["source_ids"] for item in batch]),
        "source_masks": torch.stack([item["source_masks"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch])
    }

def get_trainer_class(model_name: str) -> Type[BaseNeuralReasoningTrainer]:  # Fixed return type
    """
    Determine appropriate trainer class based on model architecture.
    """
    model_name_lower = model_name.lower()
    if '11b' in model_name_lower:
        return T5LargeReasoningTrainer
    elif any(name in model_name_lower for name in ['t5', 'flan', 'mt5', 'umt5']):
        return T5ReasoningTrainer
    elif 'bart' in model_name_lower:
        return BartReasoningTrainer
    elif 'pegasus' in model_name_lower:
        return PegasusReasoningTrainer
    
    raise ValueError(f"Unsupported model architecture: {model_name}")

def setup_datasets(
    config: ElasticSearchConfig,
    trainer: BaseNeuralReasoningTrainer,
    batch_size: int,
    source_len: int,
    target_len: int,
    force_recollect: bool = False,  # New parameter
    cache_dir: str = "cache"  # New parameter
) -> Tuple[DataLoader, DataLoader]:
    """
    Set up training and validation datasets with caching support.
    
    Args:
        config: ElasticSearch configuration
        trainer: Model trainer instance
        batch_size: Batch size for training
        source_len: Maximum source sequence length
        target_len: Maximum target sequence length
        force_recollect: If True, ignore cache and recollect IDs
        cache_dir: Directory for caching splits
    """
    # Prepare split parameters
    split_params = {
        'url': config.url,
        'index': config.index,
        'n_sentences': config.n_sentences,
        'test_ratio': 0.3,
        'seed': 42,
        'filter_method': FilterMethod.STOPWORDS
    }
    
    # Handle article ID filtering
    if config.article_ids_file not in [None, '', 'Not Found']:
        try:
            with open(config.article_ids_file, 'r') as f:
                article_ids = [line.strip() for line in f]
                
            if len(article_ids) <= 10 or len(article_ids) < config.n_articles:
                raise ValueError(
                    f"Insufficient articles ({len(article_ids)}) for analysis. "
                    f"Minimum required: max(10, {config.n_articles})"
                )
                
            random.seed(42)
            random.shuffle(article_ids)
            split_params['filter_article_ids'] = article_ids[:config.n_articles]
            
        except FileNotFoundError:
            logger.warning(f"Article IDs file not found: {config.article_ids_file}")
        except Exception as e:
            logger.error(f"Error processing article IDs: {str(e)}")
            raise
    
    # Try to load from cache if not force_recollect
    train_ids = test_ids = None
    if not force_recollect:
        cache_result = load_split_cache(split_params, cache_dir)
        if cache_result is not None:
            train_ids, test_ids = cache_result
            logger.info("Successfully loaded split from cache")
    
    # Create new split if necessary
    if train_ids is None or test_ids is None:
        logger.info("Collecting new train/test split...")
        train_ids, test_ids = ElasticSearchDataset.create_train_test_split(**split_params)
        # Cache the new split
        save_split_cache(train_ids, test_ids, split_params, cache_dir)
        logger.info("New split saved to cache")
    
    logger.info(f"Dataset split - Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    # Rest of the function remains the same...
    true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
    
    dataset_params = {
        'url': config.url,
        'index': config.index,
        'tokenizer': trainer.tokenizer,
        'true_sample_f': true_sample,
        'es_page_size': config.page_size,
        'batch_size': batch_size,
        'source_len': source_len,
        'target_len': target_len,
        'cache_size_limit': config.page_size,
        'seed': 42
    }
    
    train_dataset = ElasticSearchDataset(
        **dataset_params,
        selected_doc_ids=train_ids
    )
    val_dataset = ElasticSearchDataset(
        **dataset_params,
        selected_doc_ids=test_ids
    )
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0, collate_fn=t5_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0, collate_fn=t5_collate_fn)
    
    return train_loader, val_loader

def setup_local_datasets(
    config: LocalDataConfig,
    trainer: BaseNeuralReasoningTrainer,
    batch_size: int,
    source_len: int,
    target_len: int
) -> Tuple[DataLoader, DataLoader]:
    
    # Parameters for local splits
    split_params = {
        'file_path_train':config.file_path_train,
        'file_path_test':config.file_path_test,
        'test_ratio': config.test_ratio,
        'seed':config.seed,
        'chunk_size': config.chunk_size
    }

    #if ".jsonl" in config.file_path:
        
    #    chunk_iterator = pd.read_json(config.file_path, lines=True, chunksize=500, encoding='utf-8')
    #    print(f'El archivo {config.file_path} se abrio correctamente')
    #    for chunk in chunk_iterator:
    #        test_size = int(len(chunk) * config.test_ratio)
    #        train = chunk[test_size:]
    #        test = chunk[:test_size]
    #        train.reset_index(drop=True, inplace=True)
    #        test.reset_index(drop=True, inplace=True)

    #    train_dataset = JSONLDataset(
    #        train, 
    #        trainer.tokenizer, 
    #        source_len, 
    #        target_len)
        
    #    test_dataset = JSONLDataset(
    #        test,
    #        trainer.tokenizer, 
    #        source_len, 
    #        target_len)
    #else:
    # For data in csv format and diferent files to load
    # La variable bolleana incia si es que se quieren mezclar los datasets
    train_dataset = IterableJSONLDataset(config.file_path_train, config.path_sumarization_train, True, config.chunk_size, trainer.tokenizer, source_len, target_len)
    test_dataset = IterableJSONLDataset(config.file_path_test, config.path_sumarization_test, True, config.chunk_size, trainer.tokenizer, source_len, target_len)

    # Configurar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=t5_collate_fn
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=t5_collate_fn
    )
    
    return train_loader, val_loader

def train_model(
    trainer: BaseNeuralReasoningTrainer,  # Fixed type hint
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Tuple[float, dict]:
    """
    Train the model and evaluate performance.
    """
    # Setup optimizer
    optimizer_class = AdamW if config.optimizer == "adamw" else Adam
    optimizer = optimizer_class(
        trainer.model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01 if config.optimizer == "adamw" else 0
    )
    
    # Configure wandb monitoring
    if DETAILED_LOG_WB:
        wandb.watch(
            trainer.model,
            log="all",
            log_freq=100,
            log_graph=True
        )
    
    # Set training parameters
    trainer.score_type = 'all'
    trainer.gen_method = 'beam'
    
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        with ClearCache():# Get configurations from wandb
            trainer.train(optimizer, train_loader, epoch)
    
    logger.info("Training completed. Running final evaluation...")
    with ClearCache():# Get configurations from wandb
        # Se selecciona si se desea imprimir las metricas por paso
        final_loss, final_scores = trainer.test(val_loader, por_paso=True)
    
    return final_loss, final_scores

#@gpu_wait
def main():
    """Main training pipeline."""
    with wandb.init() as run:
        with ClearCache():
            # Get configurations from wandb
            training_config = TrainingConfig(
                source_len=wandb.config["source_seq_len"],
                target_len=wandb.config["target_seq_len"],
                model_name=wandb.config["hf_model_name"],
                epochs=wandb.config["epochs"],
                learning_rate=wandb.config["learning_rate"],
                batch_size=wandb.config["batch_size"],
                optimizer=wandb.config["optimizer"],
                quantization=wandb.config.get("quantization"),
                max_memory=wandb.config.get("max_memory")
            )
            #es_config = ElasticSearchConfig(
            #    url=es_settings.get("url", "http://192.168.241.210:9200"),
            #    index=es_settings.get("index", "triplets"),
            #    page_size=es_settings.get("es_page_size", 500),
            #    n_sentences=es_settings.get("n_sentences", 10000),
            #    n_articles=es_settings.get("n_articles", 10000),
            #    article_ids_file=es_settings.get("article_ids_file", "Not Found")
            #)
            ld_config = LocalDataConfig(
                file_path_train = '/app/data/triplets_CC0_part1_and_part2_sin_vector.csv',
                file_path_test = '/app/data/triplets_CC0_part3_with_header_sin_vector.csv',
                path_sumarization_test = '/app/data/articles_and_abstracts_CC0_part3.jsonl',
                path_sumarization_train = '/app/data/articles_and_abstracts_CC0_part_1_2.jsonl',
                chunk_size = 900,
                test_ratio = 0.3,
                seed = 42
            )            
            # Get caching options from settings or wandb config
            force_recollect = wandb.config.get("force_recollect", False)
            cache_dir = es_settings.get("cache_dir", "data/cache_art_ids")
            
            #logger.info(f"ElasticSearch configuration: {es_config}")
            logger.info(f"Cache settings - Dir: {cache_dir}, Force recollect: {force_recollect}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            logger.info(f"Initial system state: {log_gpu_memory_usage()}")
        
            trainer_class = get_trainer_class(training_config.model_name)
            trainer = (
                trainer_class.from_pretrained(
                    model_name=training_config.model_name,
                    device=device,
                    quantization=training_config.quantization,
                    max_memory=training_config.max_memory
                )
                if issubclass(trainer_class, T5LargeReasoningTrainer)
                else trainer_class.from_pretrained(training_config.model_name, device)
            )

            # Variable para controlar el resto del procceso
            band = True
            #Se realiza el calculo estadistico
            #print("Realizando resumen estadistico")
            #resumen_estadistico(trainer)

            # Realizar el testeo antes de realizar el entrenamiento
            #path_sumarization = '/app/data/articles_and_abstracts_CC0_part3.jsonl'
            #print("Iniciando pruebas de sumarization con archivo: ", path_sumarization)
            #prueba_sumarization(path_sumarization, trainer)
            
            if band:
                # Setup datasets with caching options for elasticsearch
                #train_loader, val_loader = setup_datasets(
                #    es_config,
                #    trainer,
                #    training_config.batch_size,
                #    training_config.source_len,
                #    training_config.target_len,
                #    force_recollect=force_recollect,
                #    cache_dir=cache_dir
                #)
                # Setup datasets for local datasets
                print("Iniciando la creacion de los dataloader")
                train_loader, val_loader = setup_local_datasets(
                    ld_config,
                    trainer,
                    training_config.batch_size,
                    training_config.source_len,
                    training_config.target_len
                )
                print("Iniciando con el entrenamiento del modelo")
                final_loss, final_scores = train_model(
                    trainer,
                    train_loader,
                    val_loader,
                    training_config
                )
            
                path_sumarization = '/app/data/articles_and_abstracts_CC0_part3.jsonl'
                print("Iniciando pruebas de simarization despues de ajuste con archivo ", path_sumarization)
                prueba_sumarization(path_sumarization, trainer)
            else:
                final_loss = 0 
                final_scores = 0

            wandb.run.summary.update({
                "final_test_loss": final_loss,
                "final_test_scores": final_scores
            })

def resumen_estadistico(trainer):
    """Programa para añadir columnas a los archivos de csv de tripletas de cancer de pulmon"""

    def long(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            if file_path.endswith('.csv'):
                return total_lines - 1
            return total_lines
    
    def conut_tokens(texto):
        inputs = trainer.tokenizer.encode(
            texto,
            return_tensors="pt",
            max_length=4096,
            truncation=False
        ).to(trainer.device)

        return inputs

    # Cargar las rutas para abrir los archivos de las tripletas
    file_path_train = '/app/data/triplets_CC0_part1_and_part2_sin_vector.csv'
    file_path_test = '/app/data/triplets_CC0_part3_with_header_sin_vector.csv'
    path_sumarization_test = '/app/data/articles_and_abstracts_CC0_part3.jsonl'
    path_sumarization_train = '/app/data/articles_and_abstracts_CC0_part_1_2.jsonl'

    print(f"Total de lineas en {file_path_train} : {long(file_path_train)}")
    print(f"Total de lineas en {file_path_test} : {long(file_path_test)}")
    
    # Iniciar iteracion para obtener conteo de los datos en la columna
    reader = pd.read_csv(file_path_train ,chunksize= 1000, header=0)
    long_word_obj = []
    long_word_verb = []
    long_word_suj = []
    long_token_obj = []
    long_token_verb = []
    long_token_suj = []
    for chunk in reader:
        for _,row in chunk.iterrows():
        
            def safe_str(value):
                if pd.isna(value):
                    return ""
                return str(value)
            
            objeto = safe_str(row.iloc[-1])
            verbo = safe_str(row.iloc[-2])
            sujeto = safe_str(row.iloc[-3])

            long_token_obj.append(conut_tokens(objeto).shape[1])
            long_token_verb.append(conut_tokens(verbo).shape[1])
            long_token_suj.append(conut_tokens(sujeto).shape[1])

            long_word_obj.append(len(objeto))
            long_word_verb.append(len(verbo))
            long_word_suj.append(len(sujeto))
        

    # Realizar el calculo estadistico
    data_word = {'Sujeto': long_word_suj,
            'Verbo': long_word_verb,
            'Objeto': long_word_obj}
    df = pd.DataFrame(data_word)
    resumen_estadistico = df.describe()
    print("Resumen Estadistico con base en palabras")
    print(resumen_estadistico)

    # Se realiza el calculo estadustico para tokens
    data_token = {'Sujeto': long_token_suj,
            'Verbo': long_token_verb,
            'Objeto': long_token_obj}
    df = pd.DataFrame(data_token)
    resumen_estadistico_tokens = df.describe()
    print("Resumen Estadistico con base en tokens")
    print(resumen_estadistico_tokens)

    # Calculo de estadisticas para los elementos de resumen
    print(f"Total de lineas en {path_sumarization_train} : {long(path_sumarization_train)}")
    print(f"Total de lineas en {path_sumarization_test} : {long(path_sumarization_test)}")
    
    # Iniciar iteracion para obtener conteo de los datos en la columna
    reader = pd.read_json(path_sumarization_train, lines=True, chunksize=1000)
    long_sumarzation_words = []
    long_abstract_words = []
    long_sumarzation_tokens = []
    long_abstract_tokens = []
    for chunk in reader:
        for _,row in chunk.iterrows():
        
            def safe_str(value):
                if pd.isna(value):
                    return ""
                return str(value)
            
            sumarzation_text = safe_str(row['Article'])
            abstract_text = safe_str(row['Abstract'])

            long_sumarzation_tokens.append(conut_tokens("summarize: " + sumarzation_text).shape[1])
            long_abstract_tokens.append(conut_tokens(abstract_text).shape[1])

            long_sumarzation_words.append(len(sumarzation_text))
            long_abstract_words.append(len(abstract_text))
        

    # Realizar el calculo estadistico para palabras
    data = {'Article': long_sumarzation_words,
            'Abstract': long_abstract_words}
    df = pd.DataFrame(data)
    resumen_estadistico = df.describe()
    print("Resumen Estadistico con base en palabras")
    print(resumen_estadistico)

    # Se realiza el calculo estadustico para tokens
    data_token = {'Article': long_sumarzation_tokens,
            'Abstract': long_abstract_tokens}
    df = pd.DataFrame(data_token)
    resumen_estadistico_tokens = df.describe()
    print("Resumen Estadistico con base en tokens")
    print(resumen_estadistico_tokens)
  

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with error:")
        raise
    finally:
        logger.info(f"Final system state: {log_gpu_memory_usage()}")
