# deep-Sep

In this study, we develop a deep learning-based algorithm, deep-Sep, for quickly and precisely identifying bacterial selenoprotein genes in genomic sequences. This algorithm uses a Transformer-based neural network architecture to build an optimal model for Sec-encoding UGA codon in bacteria, and a homology search-based strategy to remove additional false positives.

This is our online-server [web](http://deepsep.metalbioinfolab.net:7001/). We welcome you to try it out.

## 1. Enviroment setup
You can use the `requirements.txt` file to install all the necessary Python packages on your server. You will need at least one NVIDIA GPU.

`pip install -r requirements.txt`

Alternatively, you can use Conda to install all the essential packages.

## 2. Data processing
### 2.1
You can download the data from Google Driver [Data](https://drive.google.com/drive/folders/1J0bXIK0r7z6h-LKqm4q2Xt2SWTR64qg2?usp=drive_link). Then you need to put these data into the `data` folder.

### 2.2
You can use the `prediction/create_ORF.py` to collect all TGA triplets in the six reading frames (both strands) of each query genome and analyze the regions upstream and downstream of each TGA triplet for the presence of a reasonable ORF. Meanwhile, for those ORFs containing possible in-frame TGA codons, 300 nts immediately downstream of the TGA codon are collected.

## 3. Train Model 
You can download from Google Driver [Model](https://drive.google.com/drive/u/1/folders/12DUpJQpV-LEk0Z0BGuES3p3HLdpRt7zd). Then you need to put these things (about neural network model and db based on homology search) into the `model` folder.

You can see the detail in the `model_train_val.py`.

## 4. Test Model 
You can see the detail in the `test.py`.

You can use the /data/test.csv to test the model by running
```
python test.py \
    -k 3 \
    -max_length 300 \
    -test_df data/test.csv \
    -tokenizer model/tokenizer \
    -pretrained_model model/checkpoint-11007 \
    -finetuned_model model \
    -output_path test_output \
    -hidden_size 768
```

Based on the above command, the neural network model will be tested based on the `data/test.csv`, and the test results will be stored in the automatically generated `test_output` folder.

## 5. Prediction
<!-- Run deepSep with test.py and homology.py. -->
You can get prediction by running
```
python prediction/main.py \
    -sequence TTGGAGACCTGGAGACCATGCGCTTATCAACCTGATGACGCTGCGATATTAGAAGATTTTGATATCACACATCTCAAAAACACATTGGAGGTCATTATGAAATTATACGAAAAACTCAATGAAATTAAGCAGAAGTCTATAGCGAATATACCACCTGAATTGATTGCAATCATGCTTAAAAGCACCGAAGAACTGGTACAATCAGGAATCGCTGATAAGGCGATCAGCGTTGGTGAAGCTCTACCGGAGTTTACACTTCCCGATGCAAATGGCAATCTGATCAGTTCAAGAGATCTTCTTGCAAAAGGCCCTCTTGCCATCAGTTTTTATCGGGGTATATGGTGACCTTACTGTAACGTTGAGCTGGAAGCTCTGCAGGAAGTCTACGGTCAGGTACTGGGACTCGGAGGTTCATTCATCGCTATTTCCCCCCAACTGAGTAAATATACACAACAGGTTGTAAAAAAGAATAACCTCACTTTTCCGGTACTGGTCGATGCAGATAACGGGTATGCTGAGAAACTTGGCCTGACTTTCCCCTTGCCGGAAAAACTCAAGGAAGTGTATAAAGGTTTCGGCATTGATCTTGAGCGCTTCAATGGTAACAATTCATGGAAGCTCCCAATGTCCGGAAGGTTTATCATTGCCAGCGACGGTATTATCAGATCGACTGAGGTGCATCCGGACCATACCATCAGGCCGGAACCAAAAGAAATCGTTGATCTTCTGAAATCAATTGCTTAG \
    -program_path ../diamond_v2.1.8/diamond (Change to the path where your Diamond program is located)
```

Based on the above command, a `tmp` folder will be automatically generated under the `prediction` folder based on the provided sequence, and the collected ORFs will be stored in `ORFs` folder, and the prediction results of the neural network (called `dl_result.csv`), the Diamond homology search that will be stored in the `homology` folder, and the filtered results (called `final_results.csv`) will be generated .
