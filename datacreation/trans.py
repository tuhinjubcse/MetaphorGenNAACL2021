# from vua_detection_mlm.roberta_bert.main import main
from vua_detection_mlm.roberta_bert.main_bertonly import main
import os
import argparse


device = "0" # gpu device id
input_file = "" # a file with lines of sentences
output_dir = "" # directory of output files
metaphor_model_path = "" # metaphor detection model file

main(input_file, output_dir, metaphor_model_path, device)

os.system('python comet-commonsense/scripts/generate/generate_conceptnet_arbitrary_trans.py --model_file comet-commonsense/pretrained_models/conceptnet_pretrained_model.pickle --input_file ' + os.path.join(output_dir, "mlm_ans.json") + ' --output_file ' + os.path.join(output_dir, "output.json") + ' --device ' + device + ' --sampling_algorithm beam-5')
