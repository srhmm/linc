from statistics import mean

import numpy as np
import pandas as pd

import argparse



def main(args):
    fnm = f'data_tuebingen/pairs/pair0049.txt'
    f2 = f'data_tuebingen/pairs/pair0042.txt'
    f3 = f'data_tuebingen/pair0049.txt'
    ozone_temp = pd.read_csv(fnm, header = None, delimiter=r'[\s]+')
    temp_day = pd.read_csv(f2, header = None, delimiter=r'[\s]+')
    import matplotlib.pyplot as plt
    plt.scatter(ozone_temp[0], ozone_temp[1])
    plt.scatter(temp_day[0], temp_day[1])

    #cutoffs = [0, 90, 90+91, 90+91+92, 90+91+92+92]
    #mean_temps = [mean(temp_day[1][(cutoffs[i - 1]):(cutoffs[i])]) for i in range(1,len(cutoffs))]

    ranges_seasons = [[i for i in range(366) if i in range(60) or i in range(335,366)] , range(60,152), range(152, 244), range(244, 335)]

    ranges_seasons = [[i for i in range(366) if i in range(121) or i in range(304, 366)], range(121, 304)]

    mean_temps = [mean(temp_day[1][rng]) for rng in ranges_seasons]

    ozone_temp[2] = [np.argmin(np.abs(ozone_temp[1][i]-mean_temps)) for i in range(len(ozone_temp[1]))]

    print()
    #Xs = [np.array(X[:min(len(X), 1000)])] # np.array(X[:int(np.floor(len(X)/2))], dtype=float),  np.array(X[int(np.floor(len(X)/2)):], dtype=float)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        help="Enable to run a smaller, test version",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    main(args)
