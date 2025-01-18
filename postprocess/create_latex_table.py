import os
import re
class Stats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

class Results:
    def __init__(self, base_model, alpha, level, q, length, coverage):
        self.base_model = base_model
        self.alpha = alpha
        self.level = level
        self.q = q
        self.length = length
        self.coverage = coverage

def _results_dir():
    return os.path.join(os.path.dirname(__file__), '../src/models/results')


def extract_numbers(line):
    """
    Extracts the two numbers (mean and std) from the given line format.

    Parameters:
        line (str): The input string in the format
                    'q CP mean: <number>, q CP std: <number>'

    Returns:
        tuple: A tuple containing the mean and std as floats.
    """
    # Use regex to find all numbers in the string
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    if len(matches) >= 2:
        mean = float(matches[0])
        std = float(matches[1])
        return mean, std
    else:
        raise ValueError("Could not find two numbers in the input line.")

def parse_cp_results(base_model, alpha, level, results):
    results = results.split('\n')
    cp_q_mean, cp_q_std = extract_numbers(results[0])
    cp_len_mean, cp_len_std = extract_numbers(results[1])
    cp_cov_mean, cp_cov_std = extract_numbers(results[2])

    cqr_mean, cqr_std = extract_numbers(results[3])
    cqr_len_mean, cqr_len_std = extract_numbers(results[4])
    cqr_cov_mean, cqr_cov_std = extract_numbers(results[5])

    cp_results = Results(base_model, alpha, level, Stats(cp_q_mean, cp_q_std), Stats(cp_len_mean, cp_len_std), Stats(cp_cov_mean, cp_cov_std))
    g_results = Results(base_model, alpha, level, Stats(cqr_mean, cqr_std), Stats(cqr_len_mean, cqr_len_std), Stats(cqr_cov_mean, cqr_cov_std))
    return cp_results, g_results


def load_cp_results(base_model, alpha, level):
    results_dir = _results_dir()
    # read txt file
    with open(f'{results_dir}/lumbar_dataset_model_{base_model}_alpha_{alpha}_level_{level}_iterations_20.txt', 'r') as file:
        results = file.read()
    return parse_cp_results(base_model, alpha, level, results)


def write_line(file, model_name, g_results, cp_results):
    digits_len = 3
    digits_cov = 2
    file.write(f'& {model_name} &')
    # g results
    file.write(f' {g_results.length.mean:.{digits_len}f} $\pm$ {g_results.length.std:.{digits_len}f} &')
    file.write(f' {(100*g_results.coverage.mean):.{digits_cov}f} $\pm$ {(100*g_results.coverage.std):.{digits_cov}f} &')
    # cqr results
    file.write(f' NA $\pm$ NA &')
    file.write(f' NA $\pm$ NA &')
    # cp results
    file.write(f' {cp_results.length.mean:.{digits_len}f} $\pm$ {cp_results.length.std:.{digits_len}f} &')
    file.write(f' {(100*cp_results.coverage.mean):.{digits_cov}f} $\pm$ {(100*cp_results.coverage.std):.{digits_cov}f} \\\\ \n')


if __name__ == '__main__':
    level = 5
    alpha = 0.1
    cp_results_dense, g_results_dense = load_cp_results('densenet201', alpha, level)
    cp_results_efficient, g_results_efficient = load_cp_results('efficientnetb4', alpha, level)
    # write the results to a latex table
    os.makedirs('./tables', exist_ok=True)
    file_name = f'./tables/results_level_{level}_alpha_{alpha}.txt'
    with open(file_name, 'w') as file:
        file.write("\multirow{2}{*}{" + f'RSNA{level}' +"}")
        file.write("\n")
        write_line(file, 'DenseNet201', g_results_dense, cp_results_dense)
        write_line(file, 'EfficientNet-B4', g_results_efficient, cp_results_efficient)
