# MetaphorGenNAACL2021

      pip install torch==1.3.1
      cd fairseq
      pip install --editable .
      
-     Change the encoder.json path to correct path in fairseq/fairseq/data/encoder/gpt2_bpe_utils.py line 131
-     Training and validation data for generator inside /fairseq/metaphor folder
-     Training and validation data for discriminator inside /fairseq/glue_data/metaphor 




All preprocessed versions shared as well
-      Preprocessed data for generator is inside /fairseq/metaphor folder . You can see bpe and idx files
-      Preprocessed data for discriminator is inside /fairseq/metaphor-bin folder .

If you want to use your own metaphor data for generator
-     Create train and val source and target files for a finetuning seq2seq model. You can see my data format
-     The input format has the TEXT portion to be replaced enclosed in <V>. You can emulate the same
-     run sh preprocess1.sh and sh preprocess2.sh


If you want to use your own metaphor data for discriminator
-     Create train.tsv and dev.tsv in same tab seperated format 
-     ./examples/roberta/preprocess_GLUE_tasks.sh glue_data metaphor

Training
-     To run bart model sh trainbart.sh
-     To run roberta model sh roberta_train.sh


Inference using our finetuned model
-     For inference for MERMAID use inference.py
-     You have to change WP_scorers.tsv to reflect your own coefficients (can be positive or negative anything) and also directories 
-     You have to edit inference.py and WP_scorers.tsv to your checkpoint locations download checkpoints from the link 
      <br/>https://drive.google.com/drive/folders/1j6HNNBc_Ess-FSSbZNwA_09WOvE2C1jf?usp=sharing


