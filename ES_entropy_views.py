from phenvs import PhraseEntropyViews
from elastic_pytorch_loader.es_dataset import ElasticSearchDataset

from pdb import set_trace as st

url = "http://192.168.241.210:9200"
index = 'triplets'
# Amount of sentences to load from ElasticSearch in memory 
es_page_size = 100
# Total amount of sentences to get their triplets
max_docs2load = 200
generate = False

#true_sample = lambda x: (' '.join((x[0], x[1])), x[2]) if len(x) >= 3 else x
true_sample = lambda x: (x[0], ' '.join((x[1], x[2]))) if len(x) >= 3 else x

batched_data = ElasticSearchDataset(url=url, index=index,
        es_page_size=es_page_size, tokenizer=None,
        true_sample_f=true_sample, max_documents=max_docs2load,
        batch_size=100, yield_raw_triplets=True)
pev = PhraseEntropyViews()

# Assuming you have your batched_data ready
entropy_views = pev.fit_entropies(batched_data, return_results=True, n_jobs=1)

pev.cluster_entropy_levels(entropy_views, mode='full', plot=True, n_epochs=5, batches_to_plot=200)
#pev.cluster_entropy_levels(entropy_views, mode='2dembed', plot=True, n_epochs=5)