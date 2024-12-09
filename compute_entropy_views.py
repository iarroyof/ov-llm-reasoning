# This was prototype of https://colab.research.google.com/drive/1-FwFkgE6xnjGqgqASTfoAj3KEeOT8V6e?authuser=1#scrollTo=cUX13ZmHcojq&line=10&uniqifier=1
# Training data was declared on https://colab.research.google.com/drive/1-FwFkgE6xnjGqgqASTfoAj3KEeOT8V6e?authuser=1#scrollTo=PlDNfl98DDAB&line=211&uniqifier=1
from phenvs import PhraseEntropyViews

high_info_phrases = [
    "The Hubble Space Telescope has observed over 1 million celestial objects since its launch in 1990.",
    "CRISPR-Cas9 gene editing technology allows for precise modifications of DNA sequences.",
    "The Great Barrier Reef is the world's largest coral reef system, stretching over 2,300 kilometers.",
    "Blockchain technology utilizes distributed ledger systems to ensure data integrity and transparency.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Artificial intelligence algorithms can now generate photorealistic images from text descriptions.",
    "The Large Hadron Collider is the world's largest and most powerful particle accelerator.",
    "Renewable energy sources accounted for 28% of global electricity generation in 2020.",
    "The deepest part of the ocean, the Challenger Deep, reaches a depth of nearly 11,000 meters.",
    "Neuroplasticity allows the brain to form new neural connections throughout life.",
    "The International Space Station orbits the Earth at an average altitude of 400 kilometers.",
    "Machine learning models can predict protein structures with near-atomic accuracy.",
    "The Amazon rainforest produces 20% of the world's oxygen and contains 10% of known species.",
    "Quantum entanglement enables instantaneous communication between particles regardless of distance.",
    "The human microbiome contains trillions of microorganisms that influence health and disease.",
    "Graphene is a two-dimensional material consisting of a single layer of carbon atoms.",
    "The world's oldest known living tree is a Great Basin bristlecone pine, over 4,800 years old.",
    "CERN's particle physics experiments have confirmed the existence of the Higgs boson.",
    "The human brain contains approximately 86 billion neurons and 100 trillion synapses.",
    "Exoplanets are planets that orbit stars outside our solar system, with over 5,000 confirmed to date.",
    "The global average temperature has increased by 1.1°C since the pre-industrial era.",
    "Fusion energy research aims to replicate the process that powers the sun for clean energy production.",
    "The Internet of Things (IoT) is expected to connect over 75 billion devices by 2025.",
    "The Svalbard Global Seed Vault stores over 1 million seed samples as a backup for global biodiversity.",
    "Gravitational waves, predicted by Einstein, were first directly observed in 2015.",
    "The human gut contains over 100 trillion bacteria, weighing up to 2 kilograms.",
    "The Great Pacific Garbage Patch is a collection of marine debris spanning an area of 1.6 million square kilometers.",
    "CERN's Large Hadron Collider accelerates protons to 99.9999991% the speed of light.",
    "The world's longest known cave system is Mammoth Cave in Kentucky, with over 650 kilometers of passages.",
    "Optical quantum computers can perform certain calculations exponentially faster than classical computers.",
    "The human body contains enough carbon to make 900 pencils.",
    "The Voyager 1 spacecraft is currently the farthest human-made object from Earth, at over 23 billion kilometers away.",
    "The world's smallest known vertebrate is the Paedocypris progenetica fish, measuring just 7.9 millimeters.",
    "The human eye can distinguish between 10 million different colors.",
    "The world's largest known organism is a quaking aspen clone covering 43 hectares in Utah.",
    "The fastest supercomputer can perform over 1 quintillion calculations per second.",
    "The Greenland ice sheet contains enough water to raise global sea levels by 7.2 meters if melted completely.",
    "The human skeleton renews itself completely every 10 years.",
    "The world's deepest known cave is the Veryovkina Cave in Abkhazia, Georgia, with a depth of 2,212 meters.",
    "The total length of DNA in a single human cell, if stretched out, would be about 2 meters.",
    "The world's loudest animal relative to its size is the water boatman, which produces 99.2 decibels by rubbing its penis against its abdomen.",
    "The Atacama Desert in Chile is the driest place on Earth, with some areas receiving no rainfall for decades.",
    "The human body contains enough iron to make a 3-inch nail.",
    "The world's largest known single flower is the Rafflesia arnoldii, which can grow up to 1 meter in diameter.",
    "The fastest land animal, the cheetah, can reach speeds of up to 120 kilometers per hour.",
    "The human brain processes visual information 60,000 times faster than text.",
    "The world's oldest known living animal is a quahog clam named Ming, estimated to be over 500 years old.",
    "The total weight of all ants on Earth is estimated to be equal to the total weight of all humans.",
    "The human heart beats approximately 100,000 times per day, pumping about 7,500 liters of blood.",
    "The world's largest desert is Antarctica, with an area of 14 million square kilometers."
]

low_info_phrases = [
    "Things are not always what they seem.",
    "Everyone has their own opinion on the matter.",
    "It's important to think outside the box.",
    "The situation is more complex than it appears.",
    "There are many factors to consider in this case.",
    "We need to look at the big picture.",
    "It's a challenging issue with no easy solutions.",
    "The truth is often somewhere in the middle.",
    "We should take things one step at a time.",
    "It's better to be safe than sorry.",
    "There's no one-size-fits-all solution.",
    "We need to think critically about this.",
    "It's important to keep an open mind.",
    "The devil is in the details.",
    "We should focus on the positives.",
    "There's always room for improvement.",
    "It's not what you know, it's who you know.",
    "We need to be proactive, not reactive.",
    "The grass isn't always greener on the other side.",
    "It's all about perspective.",
    "We should expect the unexpected.",
    "There are pros and cons to every decision.",
    "It's important to stay flexible and adaptable.",
    "We need to think long-term.",
    "There's no such thing as a free lunch.",
    "It's better to ask for forgiveness than permission.",
    "We should learn from our mistakes.",
    "There's more than one way to skin a cat.",
    "It's not over until it's over.",
    "We need to prioritize our efforts.",
    "There's always light at the end of the tunnel.",
    "It's important to maintain a work-life balance.",
    "We should never judge a book by its cover.",
    "There's no time like the present.",
    "It's not about the destination, it's about the journey.",
    "We need to think outside the box.",
    "There's no 'I' in team.",
    "It's important to give credit where credit is due.",
    "We should always strive for excellence.",
    "There's more than meets the eye.",
    "It's better to be prepared for the worst and hope for the best.",
    "We need to keep our options open.",
    "There's no use crying over spilled milk.",
    "It's important to stay positive.",
    "We should never underestimate the power of teamwork.",
    "There's always a silver lining.",
    "It's not rocket science.",
    "We need to think on our feet.",
    "There's no substitute for hard work.",
    "It is what it is, in the end."
]

high_info = [
     "The results of the experiment show a significant improvement.",
    "We used a convolutional neural network for image classification.",
    "The novel method enhances the accuracy of predictions.",
    "Quantum computing is a rapidly advancing field.",
    "The integration of AI in healthcare has led to significant advancements in disease diagnosis and treatment.",
    "Climate change is accelerating the melting of polar ice caps, causing sea levels to rise at an alarming rate.",
    "The development of renewable energy sources is crucial for mitigating the impacts of climate change.",
    "Research indicates a strong correlation between socioeconomic status and educational attainment.",
    "The human brain is capable of astonishing feats of learning, memory, and problem-solving.",
    "Advances in genetic engineering have the potential to revolutionize agriculture and medicine.",
    "The exploration of Mars offers valuable insights into the possibility of extraterrestrial life.",
    "Artificial intelligence is rapidly transforming industries such as finance, transportation, and customer service.",
    "The study of human behavior provides essential knowledge for addressing social and psychological issues.",
    "Economic inequality is a complex problem with far-reaching consequences for societies worldwide.",
    "The universe is vast and filled with countless galaxies, stars, and planets.",
    "The human immune system is a complex network of cells and organs that protect against disease.",
    "The development of sustainable cities is essential for addressing urban challenges and improving quality of life.",
    "The study of history helps us understand the present and shape the future.",
    "The global population is aging rapidly, with significant implications for healthcare and social services.",
    "The discovery of new materials has the potential to drive innovation in various fields.",
    "The study of animal behavior provides insights into human psychology and social interactions.",
    "The impact of social media on mental health is a growing concern.",
    "The development of clean energy technologies is essential for achieving a sustainable future.",
    "The study of human language reveals the complexity of human thought and communication.",
    "The exploration of deep space has led to groundbreaking discoveries about the universe.",
    "The human body is a remarkable machine with intricate systems and functions.",
    "The study of economics helps us understand how societies allocate resources and produce goods and services.",
    "The impact of technology on education is transforming the way students learn.",
    "The study of psychology provides insights into the human mind and behavior.",
    "The development of new drugs is essential for combating diseases such as cancer and Alzheimer's.",
    "The study of sociology helps us understand the structure and dynamics of human societies.",
    "The impact of climate change on biodiversity is a serious threat to the planet's ecosystems.",
    "The exploration of the ocean depths has revealed a world of incredible diversity and wonder.",
    "The human brain is capable of remarkable plasticity and adaptation.",
    "The study of anthropology provides insights into the origins and development of human cultures.",
    "The development of autonomous vehicles has the potential to revolutionize transportation.",
    "The impact of globalization on labor markets is a complex and multifaceted issue.",
    "The study of political science helps us understand the processes of governance and power.",
    "The development of renewable energy technologies is creating new economic opportunities.",
    "The study of philosophy explores fundamental questions about existence, knowledge, and morality.",
    "The impact of artificial intelligence on the job market is a subject of ongoing debate.",
    "The study of genetics is advancing our understanding of human health and disease.",
    "The exploration of space offers the potential for new discoveries and scientific breakthroughs.",
    "The human body's ability to heal itself is a remarkable process.",
    "The study of history helps us understand the causes and consequences of conflict.",
    "The development of sustainable agriculture is essential for feeding a growing population.",
    "The study of psychology is informing the development of new mental health treatments.",
    "The impact of climate change on food security is a growing concern.",
    "The exploration of the human genome has led to significant medical advances.",
    "The study of sociology helps us understand social inequality and discrimination.",
    "The development of new materials is driving innovation in the aerospace industry.",
    "The study of anthropology provides insights into the diversity of human cultures.",
    "The impact of technology on privacy is a complex and controversial issue.",
    "The exploration of the human mind is a fascinating and ongoing endeavor."
]

low_info = [
    "In the end, it is what it is.",
    "There is a lot of things to consider.",
    "This is a very important matter.",
    "It is what it is, in the end.",
    "There are many things to consider.",
    "It is what it is, in the end.",
    "There are many things to consider.",
    "This is a very important matter.",
    "In the end, it all comes down to this.",
    "There are a lot of factors to take into account.",
    "It's a complex situation, to say the least.",
    "This is something that needs to be addressed.",
    "It's important to think about all the possibilities.",
    "There are many different perspectives on this issue.",
    "It's a difficult situation to be in.",
    "This is a problem that needs to be solved.",
    "It's important to weigh all the options carefully.",
    "There are many things that could happen.",
    "It's a challenging task, to say the least.",
    "This is something that requires careful consideration.",
    "It's important to gather all the information before making a decision.",
    "There are many different approaches to this problem.",
    "It's a complicated matter, to be sure.",
    "This is something that we need to discuss further.",
    "It's important to keep an open mind.",
    "There are many different possibilities to explore.",
    "It's a difficult decision to make.",
    "This is something that we need to be aware of.",
    "It's important to be prepared for anything.",
    "There are many different factors at play.",
    "It's a challenging situation to navigate.",
    "This is something that we need to work on together.",
    "It's important to communicate effectively.",
    "There are many different ways to look at this.",
    "It's a difficult concept to grasp.",
    "This is something that we need to learn from.",
    "It's important to be patient.",
    "There are many different opinions on this subject.",
    "It's a difficult process, to say the least.",
    "This is something that we need to improve upon.",
    "It's important to stay focused.",
    "There are many different challenges to overcome.",
    "It's a difficult balance to strike.",
    "This is something that we need to be careful about.",
    "It's important to be realistic.",
    "There are many different variables to consider.",
    "It's a difficult situation to control.",
    "This is something that we need to adapt to.",
    "It's important to be flexible.",
    "There are many different perspectives to consider.",
    "It's a difficult time for everyone.",
    "This is something that we need to support.",
    "It's important to be supportive.",
    "There are many different outcomes possible.",
    "It's a difficult world to live in."
]

high_info_sents = high_info_phrases + high_info
low_info_sents = low_info_phrases + low_info
sents = high_info_sents + low_info_sents

# prompt: De la celda de código anterior, toma la lista de cadenas 'sents' y crea un diccionario con dos llaves: 'sents' y 'label', conde label correspondería a 'high_info' y 'low_info', segun sea el caso de su lista de origen

data = []
for sent in high_info_sents:
  data.append({'sentence': sent, 'label': 'high_info'})
for sent in low_info_sents:
  data.append({'sentence': sent, 'label': 'low_info'})

# prompt: Now randomize the obtained 'data' and put it into an approapriated structure to make parallel computations by batch, such as in the entropy computations of the last code cell

import random

random.shuffle(data)

batch_size = 32  # Adjust batch size as needed
batched_data = []
for i in range(0, len(data), batch_size):
  batched_data.append(data[i:i + batch_size])



pev = PhraseEntropyViews()

# Assuming you have your batched_data ready
entropy_views = pev.fit_entropies(batched_data, return_results=True, n_jobs=1)

#pev.cluster_entropy_levels(entropy_views, mode='full', plot=True, n_epochs=5)
pev.cluster_entropy_levels(entropy_views, mode='2dembed', plot=True, n_epochs=5)
