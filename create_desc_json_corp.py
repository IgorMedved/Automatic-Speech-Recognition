"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]

Where speaker.trans.txt has in each line, file_id transcription
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave


def main(data_directory, output_file):
    labels = []
    durations = []
    keys = []
    
    labels_file = os.path.join(data_directory, '{}.csv'.format(data_directory.split('/')[2]))
    #file_path = os.path.join(data_directory, file)
    print (labels_file)
    for line in open(labels_file):
        split = line.strip().split(',')
        file_id = split[0].split('/')[1].split('.')[0]
        #print (file_id)
        label = ''.join([char if char !="'" else '' for char in split[1]])
        audio_file = os.path.join(data_directory,
                                          file_id) + '.wav'
        #print (audio_file)
        audio = wave.open(audio_file)
        duration = float(audio.getnframes()) / audio.getframerate()
        audio.close()
        keys.append(audio_file)
        durations.append(duration)
        labels.append(label)
    
    #base, ext = os.path.splitext(file)
    #    if ext == '.wav':
        
            
        
            
                
    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i]})
            out_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()
    main(args.data_directory, args.output_file)
