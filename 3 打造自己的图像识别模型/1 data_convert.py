# coding:utf-8
from __future__ import absolute_import
import argparse
import os
import logging
from tfrecord import main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tensorflow-input-dir', default='pic/')
    parser.add_argument('-o', '--tensorflow-output-dir', default='tfrecord/')
    parser.add_argument('--train-shards', default=2, type=int)
    parser.add_argument('--validation-shards', default=2, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--dataset-name', default='satellite', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.input_directory = args.tensorflow_input_dir
    args.train_directory = os.path.join(args.input_directory, 'train')
    args.validation_directory = os.path.join(args.input_directory, 'validation')
    args.output_directory = args.tensorflow_output_dir
    if os.path.exists(args.output_directory) is False:
        os.makedirs(args.output_directory)
    args.labels_file = os.path.join(args.output_directory, 'label.txt')
    if os.path.exists(args.labels_file) is False:
        logging.warning('Can\'t find label.txt. Now create it.')
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')
    main(args)
