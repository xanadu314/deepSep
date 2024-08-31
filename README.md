# deep-Sep

In this study, we develop a deep learning-based algorithm, deep-Sep, for quickly and precisely identifying bacterial selenoprotein genes in genomic sequences. This algorithm uses a Transformer-based neural network architecture to build an optimal model for Sec-encoding UGA codon in bacteria, and a homology search-based strategy to remove additional false positives.

## 1. Enviroment setup
You can use the `requirements.txt` file to install all the necessary Python packages on your server. You will need at least one NVIDIA GPU.

`pip install -r requirements.txt`

Alternatively, you can use Conda to install all the essential packages.

## 2. Data processing
You can use the `create_ORF.py` to collect all TGA triplets in the six reading frames (both strands) of each query genome and analyze the regions upstream and downstream of each TGA triplet for the presence of a reasonable ORF. Meanwhile, for those ORFs containing possible in-frame TGA codons, 300 nts immediately downstream of the TGA codon are collected.

<<<<<<< HEAD
## 3. Model Training
You can see the detail in the `model_train_val.py`.

## 4. Inference
Run deep-Sep with `test.py` and `homology.py`.
=======
## 3. Train Model 
You can download from Google Driver [Model](https://drive.google.com/drive/u/1/folders/12DUpJQpV-LEk0Z0BGuES3p3HLdpRt7zd). Then you need to put these things (about neural network model and db based on homology search) into the "model" folder.

You can see the detail in the `model_train_val.py`.

## 4. Test Model 
You can see the detail in the `test.py`.

You can use the /data/test.csv to test the model by running
`
k=3
i=300

test_df=data/test.csv
tokenizer=model/tokenizer
pretrained_model=model/checkpoint-11007
finetuned_model=model
test_save_path=test_output

python test.py \
    -k ${k} \ 
    -max_length ${i} \
    -test_df ${test_df} \
    -tokenizer ${tokenizer} \
    -pretrained_model ${pretrained_model} \
    -finetuned_model ${finetuned_model} \
    -output_path ${test_save_path} \
    -hidden_size 768 
`
Based on the above command, the neural network model will be tested based on the `data/test.csv`, and the test results will be stored in the `test_output` folder

## 5. Prediction
<!-- Run deepSep with test.py and homology.py. -->
You can get prediction by running
`
sequence=TTGGAGACCTGGAGACCATGCGCTTATCAACCTGATGACGCTGCGATATTAGAAGATTTTGATATCACACATCTCAAAAACACATTGGAGGTCATTATGAAATTATACGAAAAACTCAATGAAATTAAGCAGAAGTCTATA

python prediction/main.py -sequence ${sequence}
`

Based on the above command, a `tmp` folder will be automatically generated under the `prediction` folder based on the provided sequence, and the collected ORFs will be stored in this folder, and the prediction results of the neural network, the Diamond homology search, and the filtered results will be generated.
>>>>>>> aef5bd5 (add some instructions about commands.)
