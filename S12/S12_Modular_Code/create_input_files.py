from utils import create_input_files
import os
import argparse

if __name__ == '__main__':
    # Create input files (along with word map)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='flickr8k',
        choices=['coco', 'flickr8k', 'flickr30k'],
        help='dataset type'
    )
    parser.add_argument(
        '--json', default=os.path.join(BASE_DIR, 'data', 'dataset_flickr8k.json'),
        help='path to karpathy json file'
    )
    parser.add_argument(
        '--image_folder', default=os.path.join(BASE_DIR, 'data', 'images'),
        help='path to folder containing images'
    )
    parser.add_argument(
        '--captions_per_image', type=int, default=5,
        help='number of captions to sample per image'
    )
    parser.add_argument(
        '--min_word_freq', type=int, default=5,
        help='words occuring less frequently than this threshold are binned as <unk>s'
    )
    parser.add_argument(
        '--output_folder', default=os.path.join(BASE_DIR, 'data'),
        help='folder to save files'
    )
    parser.add_argument(
        '--max_len', type=int, default=50,
        help='don\'t sample captions longer than this length'
    )
    args = parser.parse_args()
    create_input_files(dataset=args.dataset,
                       karpathy_json_path=args.json,
                       image_folder=args.image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)
