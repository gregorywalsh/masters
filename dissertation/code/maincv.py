import argparse
import torch

from random import choice
from functools import reduce

from extractors.pamap2 import PAMAP2Extractor
from models.mlp import MLP
from models.cnn import CNN
from models.rnn import RNN, CRNN
from training.trainer import Trainer


######### OPTIONS ##########
# Execution
argparser = argparse.ArgumentParser()
argparser.add_argument('--min_epochs', default=20, type=int)
argparser.add_argument('--max_epochs', default=300, type=int)
argparser.add_argument('--patience', default=20, type=int)
argparser.add_argument('--use_cpu', action='store_true', default=False)
argparser.add_argument('--verbose', action='store_true', default=False)
argparser.add_argument('--lyceum', action='store_true', default=False)
# Data
argparser.add_argument('--refresh_data', action='store_true', default=False)
args = argparser.parse_args()


# Initialise Script
if args.lyceum:
    raw_data_folder = '/lyceum/gw2g17/project/project_data/PAMAP2_Dataset/Protocol/'
    output_folder = '/lyceum/gw2g17/project/Output/PAMAP2'
else:
    raw_data_folder = '/Users/gregwalsh/project_data/PAMAP2_Dataset/Protocol/'
    output_folder = '/Users/gregwalsh/Google Drive/Study/Data Science Masters/Modules/Project/Output/PAMAP2'

sensor_pattern = '(acc6)'

filenames = [
    'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
    'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat',
    # 'subject109.dat',
]

subject_ids = [int(filename[-5]) for filename in filenames]

window_shift = 33
downsampling_factor = 3
window_secs = 5.12

extractor = PAMAP2Extractor(
    raw_data_folder=raw_data_folder,
    filenames=filenames,
    output_folder_path=raw_data_folder,
    required_sensor_regex_pattern=sensor_pattern,
    refresh_data=args.refresh_data
)

if torch.cuda.is_available() and not args.use_cpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Using device:', device)

model_types = ['cnn', 'rnn', 'mlp', 'crnn']

while True:

    for model_type in model_types:

        if model_type in ['mlp', 'cnn', 'crnn']:
            std_methods = ['ranked', 'meanstd', 'popmeanstd']
        elif model_type == 'rnn':
            std_methods = ['popminmaxsymmetric', 'rankedsymmetric']
        else:
            raise ValueError('{} is an invalid model name.'.format(model_type))

        for std_method in std_methods:

            if std_method in ['ranked', 'meanstd', 'rankedsymmetric']:

                extractor.standardise(
                    method=std_method,
                    write_out=False)

            for test_subject_id in subject_ids:

                validation_subject_id = subject_ids[test_subject_id - 2]

                if std_method in ['popmeanstd', 'popminmaxsymmetric']:
                    extractor.standardise(
                        method=std_method,
                        write_out=False,
                        test_subject_id=test_subject_id,
                        validation_subject_id=validation_subject_id
                    )

                data_params = dict()
                data_params['standardisation_method'] = std_method
                data_params['weighted_sampler'] = choice([True, False])

                learning_params = dict()
                learning_params['momentum'] = choice([0, 0.5, 0.9, 0.99])
                if model_type == 'rnn':
                    learning_params['learning_rate'] = choice([1.e-1, 1.e-2, 1.e-3])
                else:
                    learning_params['learning_rate'] = choice([1.e-1, 1.e-2, 1.e-3, 1.e-4])
                learning_params['weight_decay'] = choice([1, 1.e-2, 1.e-4, 0])
                learning_params['minibatch_size'] = choice([16, 32, 64, 128])

                test_dataset, training_dataset, validation_dataset = extractor.generate_datasets(
                    device=device,
                    test_subject_id=test_subject_id,
                    validation_subject_id=validation_subject_id,
                    window_secs=window_secs,
                    downsampling_factor=downsampling_factor,
                    window_shift=window_shift
                )

                trainer = Trainer(
                    learning_params=learning_params,
                    data_params=data_params,
                    device=device,
                    output_path=output_folder,
                    test_dataset=test_dataset,
                    training_dataset=training_dataset,
                    validation_dataset=validation_dataset
                )

                output_len = 12
                if model_type == 'cnn':
                    input_len = next(iter(trainer.train_dataloader))[0].shape[1]
                    num_input_channels = next(iter(trainer.train_dataloader))[0].shape[2]
                    encoder_params = CNN.generate_params()
                    decoder_params = MLP.generate_params()
                    model = CNN(
                        num_input_channels=num_input_channels,
                        input_size=input_len,
                        output_len=output_len,
                        encoder_params=encoder_params,
                        decoder_params=decoder_params,
                    )
                elif model_type == 'rnn':
                    num_input_channels = next(iter(trainer.train_dataloader))[0].shape[2]
                    rnn_params = RNN.generate_params()
                    model = RNN(
                        output_len=output_len,
                        num_input_channels=num_input_channels,
                        rnn_params=rnn_params
                    )
                elif model_type == 'crnn':
                    input_len = next(iter(trainer.train_dataloader))[0].shape[1]
                    num_input_channels = next(iter(trainer.train_dataloader))[0].shape[2]
                    encoder_params = CNN.generate_params()
                    rnn_params = RNN.generate_params()
                    model = CRNN(
                        input_len=input_len,
                        output_len=output_len,
                        num_input_channels=num_input_channels,
                        encoder_params=encoder_params,
                        rnn_params=rnn_params
                    )
                elif model_type == 'mlp':
                    input_len = reduce(lambda x, y: x * y, next(iter(trainer.train_dataloader))[0].shape[1:])
                    params = MLP.generate_params()
                    model = MLP(
                        input_len=input_len,
                        output_len=output_len,
                        params=params,
                        **params
                    )
                else:
                    raise ValueError('"{}" is not a valid model.'.format(model_type))

                model = model.to(device)

                # Train Network
                trainer.train(
                    model=model,
                    min_epochs=args.min_epochs,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    verbose=args.verbose,
                    test_subject_id=test_subject_id,
                    validation_subject_id=validation_subject_id
                )
