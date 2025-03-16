# deep-Sep

In this study, we have developed a deep learning-based algorithm named deep-Sep, which is designed to quickly and accurately identify bacterial selenoprotein genes within genomic sequences. This algorithm uses a Transformer-based neural network architecture to construct an optimal model specifically for the Sec-encoding UGA codon in bacteria, and a homology search-based strategy to further minimize false positives. The online server platform is available at http://deepsep.metalbioinfolab.net:7001/.

## Detailed steps for configuring and utilizing deep-Sep (Local Version)
## 1. Enviroment setup
To set up the necessary environment for deep-Sep, you can use the `requirements.txt` file to install all required Python packages on your server. Please note that you will need at least one NVIDIA GPU for optimal performance.

To install the packages using pip, simply run:
`pip install -r requirements.txt`

Alternatively, you can use Conda to install the essential packages.

## 2. Data processing
### 2.1 Downloading the test sample data:
You can download the `test_sample.csv` file to use as a starting point for your analysis.

### 2.2 Creating open reading frames (ORFs):
Use the `prediction/create_ORF.py` script to process your query genome data. This script will identify all TGA triplets in the six reading frames of both strands and analyze the upstream and downstream regions for the presence of a reasonable ORF. For ORFs that contain potential in-frame TGA codons, the script will collect the 300 nucleotides immediately downstream of the TGA codon.

## 3. Train Model (if needed)
If you need to train the model from scratch, you can download the necessary files from Google Driver [Model](https://drive.google.com/drive/folders/14DFJasrvLaHuodaN_oUF5AXtDV2YwgIA?usp=drive_link) and place them in the `model` folder (including the neural network model and the database based on homology search). However, for most users, pre-trained models will be sufficient.

Details on training can be found in the `model_train_val.py` script.

## 4. Test Model 
To test the trained model, use the `test.py` script. This will evaluate the model's performance on the `test_sample.csv` file.

```
python test.py \
    -k 3 \
    -max_length 300 \
    -test_df test_sample.csv \
    -tokenizer model/tokenizer \
    -pretrained_model model/checkpoint-11007 \
    -finetuned_model model \
    -output_path test_output \
    -hidden_size 768
```

This will generate test results in the `test_output` folder.

## 5. Prediction
<!-- Run deepSep with test.py and homology.py. -->
To make predictions on new sequences, use the `main.py` script in the `prediction` folder.

```
python prediction/main.py \
    -sequence [your_sequence_here] \
    -program_path [path_to_diamond_program]
```
Replace `[your_sequence_here]` with the actual sequence you want to predict and `[path_to_diamond_program]` with the path to your Diamond program installation.

The script will: \
·&nbsp;&nbsp;Generate a `temporary (tmp)` folder under the `prediction` folder. \
·&nbsp;&nbsp;Store the collected ORFs in the `ORFs` folder. \
·&nbsp;&nbsp;Generate the neural network prediction results in `dl_result.csv`. \
·&nbsp;&nbsp;Perform a Diamond homology search and store the results in the `homology` folder. \
·&nbsp;&nbsp;Generate the final filtered results in `final_results.csv`.
