from datasets import get_dataset_config_names
import pickle
import torch
import shutil
import os
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from datasets import load_dataset,concatenate_datasets
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53").to("cuda")
configs = get_dataset_config_names("google/fleurs")
configs = ['ca_es','hr_hr','da_dk','nl_nl','en_us','fi_fi','fr_fr','de_de','el_gr','hu_hu','ga_ie','it_it','es_419','mt_mt','pt_br','sv_se','cy_gb']
print(configs)
for name in configs:
  # ds_train = load_dataset("google/fleurs", name,split="train")
  ds_validation = load_dataset("google/fleurs", name,split="validation")
  ds_test = load_dataset("google/fleurs", name,split="test")

  ds = concatenate_datasets([ds_validation,ds_test])#,ds_validation,ds_test,ds_train
  print(len(ds))
  ds_validation = []
  ds_test = []
  proj = torch.tensor([])
  for audio_array in ds[:]["audio"]:
    # print(audio_array["array"])
    input_values = feature_extractor(audio_array["array"], return_tensors="pt",sampling_rate = 16000).input_values
    # batch_size, raw_sequence_length = input_values.shape
    # sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
    # mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
    # mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)
    with torch.no_grad():
      outputs = model(input_values.to("cuda"))#, mask_time_indices=mask_time_indices)
    sequence_length = outputs.projected_quantized_states.shape[1]
    proj = torch.cat((proj,outputs.projected_quantized_states.to("cpu")),dim = 1)
  proj = proj.squeeze()
  torch.save(proj, f'{name}.pt')
    
  # counter = {}
  # for letter in proj:
  #   letter = str(letter)
  #   if letter not in counter:
  #       counter[letter] = 0
  #   counter[letter] += 1
  # counter = {k: v /proj.shape[0]  for k, v in counter.items()}
  # with open( f'{name}.pkl', 'wb+') as f:
  #   pickle.dump(counter, f)
  # print(max(counter.values()))
  if os.path.exists('/root/.cache/huggingface/datasets'):
    # removing the file using the os.remove() method
   shutil.rmtree('/root/.cache/huggingface/datasets')
  else:
    # file not found message
    print("File not found in the directory")
  print("Done!")
