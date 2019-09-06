import torch
import numpy as np
import time
import datetime
import torch.nn.functional as F
import json
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import fbeta_score, confusion_matrix
from random import randint

from models.rnn import RNN, CRNN

headers = [
    'id',
    'model',
    'val_id',
    'test_id',
    'datetime',
    'model hyperparams',
    'data hyperparams',
    'learning hyperparams',
    'test - f1m',
    'val - f1m',
    'train - f1m',
    'test - f1w',
    'val - f1w',
    'train - f1w',
    'test - acc',
    'val - acc',
    'train - acc',
]

class Trainer:

    def __init__(self, learning_params, data_params, device, output_path,
                 test_dataset, training_dataset, validation_dataset, is_cv=False):
        self.learning_params = learning_params
        self.data_params = data_params
        self.output_path = output_path
        self.device = device
        self.is_cv = is_cv

        if data_params['weighted_sampler']:
            sample_weights = training_dataset.get_sample_weights(class_weights=training_dataset.get_class_weights())
            sampler_args={'sampler': WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)}
            print('Sampler in use')
        else:
            sampler_args = {'shuffle': True}

        self.train_dataloader = DataLoader(
            dataset=training_dataset,
            batch_size=self.learning_params['minibatch_size'],
            drop_last=True,
            **sampler_args
        )

        self.validation_dataloader = DataLoader(
            dataset=validation_dataset,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
        )

    def train(self, model, min_epochs, max_epochs, patience, verbose, test_subject_id, validation_subject_id):

        print('Training Network\n')
        print('Model params')
        print(model.get_params(), '\n')
        print('Learning params')
        print(self.learning_params, '\n')
        print('Data params')
        print(self.data_params, '\n')

        self.model = model
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        self.test_subject_id = test_subject_id
        self.validation_subject_id = validation_subject_id
        self.model_id = ('{}_{}').format(int(time.time()), randint(0,1000))

        self.scores = {key1:{key2:[] for key2 in ['f1m', 'f1w', 'acc']} for key1 in ['test', 'train', 'validation']}
        self.confusion_matrices = {key:[] for key in ['test', 'train', 'validation']}
        self.remaining_patiences = {key:self.patience for key in ['f1m', 'f1w', 'acc']}
        self.best_validation_epochs = {key: 0 for key in ['f1m', 'f1w', 'acc']}
        self.best_validation_scores = {key: 0 for key in ['f1m', 'f1w', 'acc']}

        if self.learning_params['momentum'] == 0:
            optimiser_params = {'nesterov': False}
        else:
            optimiser_params = {'nesterov': True}
        self.optimiser = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.learning_params['learning_rate'],
            weight_decay=self.learning_params['weight_decay'],
            momentum=self.learning_params['momentum'],
            **optimiser_params
        )

        for epoch in range(self.max_epochs):
            self.train_epoch(epoch=epoch)
            self.evaluate(loader=self.validation_dataloader)
            self.evaluate(loader=self.test_dataloader)

            self.remaining_patiences['f1m'] -= 1
            if self.remaining_patiences['f1m'] >= 0 and self.scores['validation']['f1m'][-1] > self.best_validation_scores['f1m']:
                self.best_validation_scores['f1m'] = self.scores['validation']['f1m'][-1]
                self.best_validation_epochs['f1m'] = epoch
                self.remaining_patiences['f1m'] = self.patience

            if epoch + 1 > self.min_epochs - self.patience:
                if self.remaining_patiences['f1m'] <= 0:
                    break

        self.print_results(last_epoch=epoch)
        self.save_final_results()
        self.save_scores()


    def print_results(self, last_epoch):
        print('\n' + '#' * 50 + '\n')
        print('Trained for {}/{} epochs.\n'.format(last_epoch + 1, self.max_epochs))
        for loader_key in ['validation', 'test']:
            print('\t{}:\n\tF1m: {:.3f}\n\tF1w: {:.3f}\n\tAcc: {:.2f}%\n'.format(
                loader_key.upper(),
                self.scores[loader_key]['f1m'][self.best_validation_epochs['f1m']],
                self.scores[loader_key]['f1w'][self.best_validation_epochs['f1m']],
                self.scores[loader_key]['acc'][self.best_validation_epochs['f1m']]
            ))
        print('\tBEST EPOCHS:\n\tF1m: {}\n\tF1w: {}\n\tAcc: {}'.format(
            self.best_validation_epochs['f1m'] + 1,
            self.best_validation_epochs['f1m'] + 1,
            self.best_validation_epochs['f1m'] + 1
        ))

    def train_epoch(self, epoch):
        self.model.train()
        loader = self.train_dataloader
        for g in self.optimiser.param_groups:
            g['lr'] = g['lr'] * (1 / (1 + 1e-2 * epoch))

        if self.verbose:
            print('EPOCH {}'.format(epoch+1))
            print('\tLearning Rate: {:.2e}'.format(g['lr']))

        num_correct = 0
        targets = []
        predictions = []
        for i, (x, y) in enumerate(self.train_dataloader):
            self.optimiser.zero_grad()
            if isinstance(self.model, (RNN, CRNN)):
                out, _ = self.model(x, None)
            else:
                out = self.model(x)
            cost = F.nll_loss(out, y)
            cost.backward()
            self.optimiser.step()

            targets.append(y)
            predicted_classes = out.max(dim=1)[1]
            predictions.append(predicted_classes)
            num_correct = num_correct + int(y.eq(predicted_classes).sum(dim=0))

            if self.verbose:
                if i % (len(self.train_dataloader) // 5) == 0:
                    print('\t{:.0f}% complete - loss: {:.6f}'.format(100.0 * (i+1) / len(self.train_dataloader), cost.item()))

        if self.verbose:
            print()

        self.update_scores(predictions=predictions, targets=targets, num_correct=num_correct, loader=loader)
        self.print_scores(loader=loader)

    def evaluate(self, loader):
        self.model.eval()
        num_correct = 0
        targets = []
        predictions = []
        with torch.no_grad():
            for x, y in loader:
                if isinstance(self.model, (RNN, CRNN)):
                    out, _ = self.model(x, None)
                else:
                    out = self.model(x)

                targets.append(y)
                predicted_classes = out.max(dim=1)[1]
                predictions.append(predicted_classes)
                num_correct = num_correct + int(y.eq(predicted_classes).sum(dim=0))

        self.update_scores(predictions=predictions, targets=targets, num_correct=num_correct, loader=loader)
        self.print_scores(loader=loader)

    def update_scores(self, predictions, targets, num_correct, loader):

        if loader is self.train_dataloader:
            loader_key = 'train'
        elif loader is self.validation_dataloader:
            loader_key = 'validation'
        elif loader is self.test_dataloader:
            loader_key = 'test'

        statistics = dict()
        statistics['f1w'] = fbeta_score(
            beta=1,
            y_true=np.concatenate(predictions),
            y_pred=np.concatenate(targets),
            average='weighted'
        )
        statistics['f1m'] = fbeta_score(
            beta=1,
            y_true=np.concatenate(predictions),
            y_pred=np.concatenate(targets),
            average='macro'
        )
        statistics['acc'] = num_correct * 100.0 / len(loader.dataset)

        for statistic_key, value in statistics.items():
            self.scores[loader_key][statistic_key].append(value)

        con_matrix = confusion_matrix(
            y_pred=np.concatenate(predictions),
            y_true=np.concatenate(targets)
        )
        self.confusion_matrices[loader_key].append(con_matrix)

    def print_scores(self, loader):
        if self.verbose:
            if loader is self.train_dataloader:
                loader_key = 'train'
            elif loader is self.validation_dataloader:
                loader_key = 'validation'
            elif loader is self.test_dataloader:
                loader_key = 'test'

            print(
                '\t{}\n\tF1m: {:.3f}\n\tF1w: {:.3f}\n\tAcc: {:.1f}%\n'.format(
                    loader_key.upper(),
                    self.scores[loader_key]['f1m'][-1],
                    self.scores[loader_key]['f1w'][-1],
                    self.scores[loader_key]['acc'][-1]
                )
            )

    def save_final_results(self):
        generated_datetime = datetime.datetime.fromtimestamp(int(self.model_id.split('_')[0]))
        line = [
            self.model_id,
            type(self.model).__name__,
            self.validation_subject_id,
            self.test_subject_id,
            generated_datetime,
            self.model.get_params(),
            self.data_params,
            self.learning_params,
        ]
        for metric_keys in ['f1m', 'f1w', 'acc']:
            for loader_key in ['test', 'validation', 'train']:
                line.append(self.scores[loader_key][metric_keys][self.best_validation_epochs['f1m']])
        else:
            filename = 'results_cv.txt'
        if os.path.exists('{}/results/{}'.format(self.output_path, filename)):
            append_write = 'a'
        else:
            append_write = 'w'
        with open('{}/results/{}'.format(self.output_path, filename), append_write) as f:
            if append_write == 'w':
                f.write('\t'.join([str(x) for x in headers]) + '\n')
            f.write('\t'.join([str(x) for x in line]) + '\n')

    def save_scores(self):

        params = dict()
        params['model'] = self.model.get_params()
        params['learning'] = self.learning_params
        params['data'] = self.data_params
        # with open('{}/params/{}.json'.format(self.output_path, self.model_id), 'w+') as f:
        #     json.dump(obj=params, fp=f, sort_keys=True, indent=4)
        with open('{}/scores/{}.json'.format(self.output_path, self.model_id), 'w+') as f:
            json.dump(obj=self.scores, fp=f, sort_keys=True, indent=4)

        for dataset_key, conv_matrices in self.confusion_matrices.items():
            epoch = self.best_validation_epochs['f1m']
            file_path = self.output_path
            file_path += '/confusion_matrices/{}/{}_{}.csv'.format('f1m', self.model_id, dataset_key)
            np.savetxt(fname=file_path, X=conv_matrices[epoch])
