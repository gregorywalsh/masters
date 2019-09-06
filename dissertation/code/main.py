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
argparser.add_argument('--dataset', default='pamap')
argparser.add_argument('--std_method', default='popmmshift')
argparser.add_argument('--test_subject', default=6, type=int)
argparser.add_argument('--experiment_num', default=20, type=int)
argparser.add_argument('--min_epochs', default=20, type=int)
argparser.add_argument('--max_epochs', default=200, type=int)
argparser.add_argument('--patience', default=20, type=int)
argparser.add_argument('--use_cpu', action='store_true', default=False)
argparser.add_argument('--model_type', default='mlp')
argparser.add_argument('--verbose', action='store_true', default=False)
argparser.add_argument('--lyceum', action='store_true', default=False)
# Data
argparser.add_argument('--refresh_data', action='store_true', default=False)
args = argparser.parse_args()


# Initialise Script
# if args.dataset.lower() == 'pamap':
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

extractor.standardise(
    method=args.std_method,
    write_out=False)

# Run Experiments
test_subject_id = args.test_subject
validation_subject_id = subject_ids[args.test_subject - 2]

# elif args.dataset.lower() == 'opportunity':
#     if args.lyceum:
#         raw_data_folder = '/lyceum/gw2g17/project/project_data/OPPORTUNITY/OpportunityUCIDataset/'
#         output_folder = '/lyceum/gw2g17/project/Output/OPPORTUNITY'
#     else:
#         raw_data_folder = '/Users/gregwalsh/project_data/OPPORTUNITY/OpportunityUCIDataset/'
#         output_folder = '/Users/gregwalsh/Google Drive/Study/Data Science Masters/Modules/Project/Output/OPPORTUNITY'
#
#     sensor_columns = [
#         38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59,
#         64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85,
#         90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109,
#         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
#         125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 250
#     ]
#     sensor_columns = [x - 1 for x in sensor_columns]
#
#     filenames = [
#         'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
#         'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
#         'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat',
#         'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
#     ]
#
#     subject_ids = [1, 2, 3, 4]
#
#     window_shift = 33
#     downsampling_factor = 3
#     window_secs = 5.12
#
#     extractor = OpportunityExtractor(
#         raw_data_folder=raw_data_folder,
#         filenames=filenames,
#         output_folder_path=raw_data_folder,
#         required_sensor_columns=sensor_columns,
#         refresh_data=args.refresh_data
#     )
#
#     extractor.standardise(
#         method=args.std_method,
#         write_out=False)
#
#     # Run Experiments
#     test_subject_id = args.test_subject
#     validation_subject_id = subject_ids[args.test_subject - 2]
#
# else:
#     raise ValueError('"{}" is not a valid dataset name for arg "--dataset"'.format(args.dataset))

if torch.cuda.is_available() and not args.use_cpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Using device:', device)

for i in range(args.experiment_num):

    data_params = {}
    data_params['standardisation_method'] = args.std_method
    data_params['weighted_sampler'] = choice([True, False])

    learning_params = {}
    learning_params['momentum'] = choice([0, 0.5, 0.9, 0.99])
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
    if args.model_type == 'cnn':
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
    elif args.model_type == 'rnn':
        num_input_channels = next(iter(trainer.train_dataloader))[0].shape[2]
        rnn_params = RNN.generate_params()
        model = RNN(
            output_len=output_len,
            num_input_channels=num_input_channels,
            rnn_params=rnn_params
        )
    elif args.model_type == 'crnn':
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
    elif args.model_type == 'mlp':
        input_len = reduce(lambda x, y: x * y, next(iter(trainer.train_dataloader))[0].shape[1:])
        params = MLP.generate_params()
        model = MLP(
            input_len=input_len,
            output_len=output_len,
            params=params,
            **params
        )
    else:
        raise ValueError('"{}" is not a valid model.'.format(args.model_type))

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

print('\n' + '#' * 50 + '\n')