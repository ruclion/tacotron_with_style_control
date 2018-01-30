import os

wav_root = 'audioBook/All_Slices_wav_24k'
soxExeFile = 'resample/resample.pl'

# $cmd = "$soxExeFile $inWavePath/$waveFile $outRawPath/$waveFile.tmp.wav rate -s -a 16000 dither -s ";
# $cmd = "$soxExeFile $outRawPath/$waveFile.tmp.wav $outRawPath/$rawFile remix 1";
# $cmd = "rm $outRawPath/$waveFile.tmp.wav";

def get_path_lst(root, cur_list=[]):
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            get_path_lst(item_path, cur_list)
        if os.path.isfile(item_path):
            if item_path.endswith("wav"):
                cur_list.append(item_path)
    return cur_list

path_lst = get_path_lst(wav_root)
for wav_path in path_lst:
    tmp_out_wav_path = wav_path[:-4] + '_sr16k.tmp.wav'
    out_wav_path = wav_path[:-4] + '_sr16k.wav'
    cmd1 = soxExeFile + ' ' + wav_path + ' ' + tmp_out_wav_path + ' rate -s -a 16000 dither -s '
    cmd2 = soxExeFile + ' ' + tmp_out_wav_path + ' ' + out_wav_path + ' remix 1'
    cmd3 = 'rm ' + tmp_out_wav_path
    # print(cmd1)
    # print(cmd2)
    # print(cmd3)
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)

