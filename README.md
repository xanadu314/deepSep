# deep-Sep

In this study, we develop a deep learning-based algorithm, deep-Sep, for quickly and precisely identifying bacterial selenoprotein genes in genomic sequences. This algorithm uses a Transformer-based neural network architecture to build an optimal model for Sec-encoding UGA codon in bacteria, and a homology search-based strategy to remove additional false positives.

## 1. Enviroment setup
You can use the requirements.txt file to install all the necessary Python packages on your server. You will need at least one NVIDIA GPU.

pip install -r requirements.txt 

Alternatively, you can use Conda to install all the essential packages.

## 2. Data processing
You can use the create_ORF.py to collect all TGA triplets in the six reading frames (both strands) of each query genome and analyze the regions upstream and downstream of each TGA triplet for the presence of a reasonable ORF. Meanwhile, for those ORFs containing possible in-frame TGA codons, 300 nts immediately downstream of the TGA codon are collected.

## 3. Model Training
You can see the detail in the model_train_val.py.

## 4. Inference
Run deepSep with test.py and homology.py.