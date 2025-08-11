# En este codigo se explorar el ajuste fino sin que se guarde en wandb y verificar los resultados de entrenamiento
import pandas as pd

df_train = pd.read_csv("filtered_train_triplets_shuffle.csv")
df_test = pd.read_csv("filtered_test_triplets_shuffle.csv")

df_train.head()
df_test.head()
