conda create --name metaphor python=3.6

conda activate metaphor

#point your LD_LIBRARY_PATH to your miniconda or anaconda library

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/tuhinc/miniconda3/lib/



Clone this repo.

  - install nltk
      
    - cd comet-commonsense
    
       To run the setup scripts to acquire the pretrained model files from OpenAI, as well as the ATOMIC and ConceptNet datasets

      ```
      bash scripts/setup/get_atomic_data.sh
      bash scripts/setup/get_conceptnet_data.sh
      bash scripts/setup/get_model_files.sh
      ```

      Then install dependencies (assuming you already have Python 3.6 ):

      ```
      pip install torch==1.3.1
      pip install tensorflow
      pip install ftfy==5.1
      conda install -c conda-forge spacy
      python -m spacy download en
      pip install tensorboardX
      pip install tqdm
      pip install pandas
      pip install ipython
      pip install inflect
      pip install pattern
      pip install pyyaml==5.1
      
      ```
      <h1> Making the Data Loaders </h1>

      Run the following scripts to pre-initialize a data loader for ATOMIC or ConceptNet:

      ```
      python scripts/data/make_atomic_data_loader.py
      python scripts/data/make_conceptnet_data_loader.py
      ```
      
      <h1> Download pretrained COMET </h1>
      
      First, download the pretrained models from the following link:

      ```
      https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB
      ```

      Then untar the file:

      ```
      tar -xvzf pretrained_models.tar.gz
      
    
 Make sure your directory resembles this 
 https://github.com/tuhinjubcse/SarcasmGeneration-ACL2020/blob/master/comet-commonsense/directory.md
 
 
 
  - <h1> Finally to generate Symbols </h1>
  
    - Run python generateSymbols.py $input
    - This will print the output in the console


Email me at tc2896@columbia.edu for any problems/doubts. Further you can raise issues on github, or suggest improvements.

## poem dataset

```bash
cd data/poem
python preprocess.py
cd ../..
python poem.py
```

Please put the poetry corpus file `gutenberg-poetry-v001.ndjson` into path `data/poem`

In the code `preprocess.py`, `stanfordcorenlp` is needed. Please download http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip and unzip to `data/corenlp`.
