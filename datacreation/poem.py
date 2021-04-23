from vua_detection_mlm.roberta_bert.main_poem import main
import os
import argparse


device = "0" # gpu device id
input_file = "data/poem/poem_dataset.csv" # after running `python preprocess.py`
output_dir = "" # directory of output files
metaphor_model_path = "" # metaphor detection model file
threshold = 0.95

main(threshold, input_file, output_dir, metaphor_model_path, device)

os.system('python comet-commonsense/scripts/generate/generate_conceptnet_arbitrary_trans.py --model_file comet-commonsense/pretrained_models/conceptnet_pretrained_model.pickle --input_file ' + os.path.join(output_dir, "mlm_ans.json") + ' --output_file ' + os.path.join(output_dir, "output.json") + ' --device ' + device + ' --sampling_algorithm beam-5')
