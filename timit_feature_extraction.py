import librosa
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
import torch
import librosa.display
import os
from collections import Counter
import pickle



def xlsr_codes(audio_path,codebook,pickle_path,feature_extractor,model):
    wav,sample = torchaudio.load(audio_path)
    input_values = feature_extractor(wav.squeeze(), return_tensors="pt",sampling_rate = 16000).input_values
    with torch.no_grad():
        codevector_probs,codevectors,codevector_idx = model(input_values)
    output = codevector_idx.view(-1,2)[:,codebook-1].tolist()
    file_pt = audio_path.split('/')[-4:-1]
    if not os.path.exists(pickle_path+'/'.join(file_pt)):
        os.makedirs(pickle_path+'/'.join(file_pt))
    with open(pickle_path+'/'.join(file_pt)+'/'+audio_path.split('/')[-1].split('.')[0]+'.pkl', 'wb') as f:
        pickle.dump(output, f)
    return output
def codes_low_high(audio_path,xlsr_codes):
    y, sr = librosa.load(audio_path)
    tot_samp = librosa.time_to_samples(librosa.get_duration(y=y, sr=sr),sr=16000)
    low = 0,
    high = 0
    time = 0
    time_1 = 0.025
    low_lst =[]
    high_lst =[]
    # print(len(xlsr_codes))
    for i,k in enumerate(xlsr_codes):
        low,high = librosa.time_to_samples(time,sr = 16000),librosa.time_to_samples(time_1,sr = 16000)
        low_lst.append(int(low))
        high_lst.append(int(high))
        # print(f'{low}-{high}:{i}')
        # if i ==0:
        #     time += 0.025
        #     time_1 += 0.025
        # else:
        time += 0.020
        time_1 += 0.020
        # print(low_lst)
    if high<tot_samp:
        # low_lst.append(tot_samp-librosa.time_to_samples(0.025,sr=16000))
        low_lst.append(tot_samp)
        high_lst.append(tot_samp)
    return low_lst,high_lst
def val_ret(lst,val):
    return lst[next(x[0]-1 for x in enumerate(lst) if x[1] > val)]
def phn_ind(phn_path,low_lst,high_lst,xlsr_codes,phn_dict ={}):
    with open(phn_path) as f:
        lines,lines_1,phn = zip(*[(line.split()) for line in f])
    # plt.figure()
    # librosa.display.waveshow(y, sr=sr )
    # plt.show()
    # fig, ax = plt.subplots()
    # librosa.display.waveshow(y, sr=sr )
    # ax.text(x=0, y=1.0, s="|".join([s for s in phn]), transform=ax.transAxes, fontsize=20)
    # plt.show()
    for val,p in enumerate(phn):
        l = val_ret(low_lst,int(lines[val]))
        if val < len(phn)-1:
            h = val_ret(high_lst,int(lines_1[val]))
        else:
            h = high_lst[-1]
        if p in phn_dict:
            phn_dict[p] += xlsr_codes[low_lst.index(l):high_lst.index(h)+1]
        else:
            phn_dict[p] = xlsr_codes[low_lst.index(l):high_lst.index(h)+1]
        print(f'{lines[val]}-{lines_1[val]}_{p}:{xlsr_codes[low_lst.index(l):high_lst.index(h)+1]}')

    return phn_dict
def main():
    # feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    # model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    codebook = 2
    rootdir = '/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/TIMIT/timit/'
    pickle_path = f'/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/timit_pkl_only_english/codebook{codebook}/'
    phn_dict ={}
    extract = 1
    phn_to_codebook_entry = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if 'wav'in os.path.join(subdir, file):
                print(os.path.join(subdir, file))
                audio_path = os.path.join(subdir, file)
                phnpath = audio_path.replace('wav','phn')
                print(phnpath)
                if extract ==1:
                    output = xlsr_codes(audio_path,codebook,pickle_path,feature_extractor,model)
                else:
                    pass
                if phn_to_codebook_entry:
                    low_lst,high_lst = codes_low_high(audio_path,output)
                    phn_dict = phn_ind(phnpath,low_lst,high_lst,output,phn_dict)
    if phn_to_codebook_entry:
        with open('saved_dictionary_codebook_1.pkl', 'wb') as f:
            pickle.dump(phn_dict, f)
        for dict_val in phn_dict.keys():
            print(f'{dict_val}-{dict(Counter(phn_dict[dict_val]))}')
if __name__ == "__main__":
   main()