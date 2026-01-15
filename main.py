import pandas as pd
import sys

def load_payoff_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df = df.map(lambda x: [float(s) for s in x.split(" ") if s != ""])
    return df

def get_payoff_matrix_file():
    if len(sys.argv) < 2:
        print("You must supply the csv file containing the payoff matrix as a command line argument.")
        return None
    return sys.argv[1]

if __name__ == "__main__":
    file = get_payoff_matrix_file()
    if file:
        print(f"Reading {file} and building payoff matrix...")

        payoff = load_payoff_matrix(file)
        print(payoff)