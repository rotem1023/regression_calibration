import os
import numpy as np

def get_results_dir():
    return os.path.join(os.path.dirname(__file__), '../src/models/results/predictions')


class Results:
    def __init__(self, base_model, level, group, predictions, sds, labels):
        self.base_model = base_model
        self.level = level
        self.group = group
        self.predictions = predictions
        self.sds = sds
        self.labels = labels

def load_results(base_model, level, group):
    results_dir = get_results_dir()
    predictions = np.load(f'{results_dir}/{base_model}_lumbar_{group}_{level}_predictions.npy')
    sds = np.load(f'{results_dir}/{base_model}_lumbar_{group}_{level}_sds.npy')
    labels = np.load(f'{results_dir}/{base_model}_lumbar_{group}_{level}_labels.npy')
    return Results(base_model=base_model, level=level, group=group, predictions=predictions, sds=sds, labels=labels)

if __name__ == '__main__':
    results = load_results('efficientnetb4', 4, 'test')
    print(results.predictions)
    print(results.sds)
    print(results.labels)
    print(results.base_model)
    print(results.level)
