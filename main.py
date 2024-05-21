import torch
import os
from model import MVIB
from utils import set_seed
from wrapper import train_CSMVIB
from config import get_args
from dataload import ImportData
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    kfold = StratifiedKFold(n_splits=args.nsplits, shuffle=True, random_state=1)

    dataset_folder = './datasets/'
    file_list = os.listdir(dataset_folder)

    for file_name in file_list:
        file_path = os.path.join(dataset_folder, file_name)
        data = loadmat(file_path)
        input_data = torch.tensor(data['X'][0][0]).float()
        labels = torch.tensor(np.squeeze(data['Y']) - 1).long() if np.min(data['Y']) != 0 else torch.tensor(np.squeeze(data['Y'])).long()
        data_set = ImportData(data_path=file_path)
        set_seed(123)

        input_dims = [data['X'][0][i].shape[1] for i in range(data['X'][0].shape[0])]
        class_num = len(np.unique(labels))
        sample_number = data['X'][0][0].shape[0]
        view_number = len(input_dims)

        results = []

        for fold_idx, (train_idxs, test_idxs) in enumerate(kfold.split(input_data, labels)):
            net = MVIB(input_dims=input_dims, class_num=class_num).to(device)
            train_subset = torch.utils.data.Subset(data_set, train_idxs)
            test_subset = torch.utils.data.Subset(data_set, test_idxs)
            trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=0)
            testloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batchsize, shuffle=False, num_workers=0)
            report = train_CSMVIB(args, trainloader, testloader, fold_idx, net, class_num)
            results.append(report)

        avg_report = np.mean(results, axis=0)
        std_report = np.std(results, axis=0)

        print(f'Average and Standard Deviation Report: {100 * avg_report} ± {100 * std_report}')

        save_results(file_path, sample_number, class_num, view_number, input_dims, args, avg_report, std_report, results)

def save_results(file_path, sample_number, class_num, view_number, input_dims, args, avg_report, std_report, results):
    file_name = f'./results/{Path(file_path).stem}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + "\n")
        f.write(f'Dataset: {Path(file_path).stem}\n')
        f.write('=' * 60 + "\n")
        f.write('Average Report and Standard Deviation Report:\nacc, precision, recall, f1_score\n')
        for avg, std in zip(avg_report, std_report):
            f.write(f"{avg * 100:.2f} ± {std * 100:.2f}\n")
        # f.writelines([' '.join(f"{result * 100:.2f}" for result in experiment) + '\n' for experiment in results])
        f.write('=' * 60 + "\n")

    print(f'Results saved to {file_name}.')

if __name__ == '__main__':
    main()
