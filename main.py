from enum import Enum
import random
import pandas as pd
import sys
import matplotlib.pyplot as plt


class Players(Enum):
    P1 = "row"
    P2 = "col"


class PayoffMatrix:
    def __init__(self, path):
        self.df = pd.read_csv(path, index_col=0)
        self.df = self.df.map(lambda x: [float(s) for s in x.split(" ") if s != ""])

    def evaluate(self, s1, s2):
        payoffs = self.df.loc[s1, s2]
        payoff1 = payoffs[0]
        payoff2 = payoffs[1]
        tie = payoff1 == payoff2

        return {
                Players.P1: {"Strategy": s1, "Payoff": payoffs[0]},
                Players.P2: {"Strategy": s2, "Payoff": payoffs[1]},
                "Winner": None if tie else (Players.P1 if payoffs[0] > payoffs[1] else Players.P2),
                "Loser": None if tie else (Players.P1 if payoffs[0] < payoffs[1] else Players.P2),
                "Draw": tie
                }


class Player:
    def __init__(self, payoff):
        # Use two separate dictionaries for row and columns weights
        # so 2x3 or any non nxn payoff matrix works as well.
        self.weights = {Players.P1: {}, Players.P2: {}}
        self.log = {Players.P1: {}, Players.P2: {}}

        for row in payoff.df.index:
            self.weights[Players.P1][row] = 1 / len(payoff.df.index)
            self.log[Players.P1][row] = [1 / len(payoff.df.index)]

        for col in payoff.df.columns:
            self.weights[Players.P2][col] = 1 / len(payoff.df.columns)
            self.log[Players.P2][col] = [1 / len(payoff.df.columns)]

        self.lr = 0.07
        self.scores = []

    def getStrategy(self, player):
        r = random.random()
        lower = 0

        weights = self.weights[player]
        for key in weights:
            if lower <= r < lower + weights[key]:
                return key
            lower += weights[key]
        return list(weights.keys())[-1] # r must equal 1 which would correspond to the last in weights

    def versus(self, p2, payoff):
        s1 = self.getStrategy(Players.P1)
        s2 = p2.getStrategy(Players.P2)
        return payoff.evaluate(s1, s2)

    def log_score(self, score):
        self.scores.append(score)

        for p in self.weights:
            for strat in self.weights[p]:
                self.log[p][strat].append(self.weights[p][strat])

    def _update_factor(self, score, player):
        scores = [s[player] for s in self.scores]
        l = len(scores)
        if l < 1:
            return 0

        range_ = max(scores) - min(scores)
        if range_ == 0:
            return 0
        
        avg = sum(scores) / l
        return self.lr * (score - avg) / range_

    def _normalize_weights(self, player):
        s = sum([self.weights[player][k] for k in self.weights[player]])
        for k in self.weights[player]:
            w = self.weights[player][k]
            self.weights[player][k] = min(max(w, 0), 1)

        for k in self.weights[player]:
            self.weights[player][k] /= s

    def update(self, p1Results, p2Results):
        p1Strategy = p1Results["Strategy"]
        p2Strategy = p2Results["Strategy"]
        p1Score = p1Results["Payoff"]
        p2Score = p2Results["Payoff"]
        
        self.log_score({
                Players.P1: p1Score,
                Players.P2: p2Score
            })

        p1Factor = self._update_factor(p1Score, Players.P1)
        p2Factor = self._update_factor(p2Score, Players.P2)
        self.weights[Players.P1][p1Strategy] += p1Factor
        self.weights[Players.P2][p2Strategy] += p2Factor
        self._normalize_weights(Players.P1)
        self._normalize_weights(Players.P2)

    def plot(self, plt, player):
        series = self.log[player]
        for key in series:
            plt.plot(series[key], label=f"{key}, as {"P1" if player == Players.P1 else "P2"} ({player.value})")


class RoundRobin:
    def __init__(self, payoff, numPlayers=10):
        self.payoff = payoff
        self.numPlayers = numPlayers
        self.players = [Player(payoff) for i in range(numPlayers)]

    def matchup(self, p1, p2):
        p1 = self.players[p1]
        p2 = self.players[p2]
        game1 = p1.versus(p2, self.payoff)
        game2 = p2.versus(p1, self.payoff)
        p1.update(game1[Players.P1], game2[Players.P2])
        p2.update(game2[Players.P1], game1[Players.P2])

    def run_session(self):
        l = len(self.players)
        midpoint = l // 2 # assume l is even
        for rnd in range(l-1):
            for i in range(midpoint):
                p1 = 0 if i == 0 else (i + rnd - 1) % (l - 1) + 1
                p2 = (l - i - 2 + rnd) % (l - 1) + 1
                self.matchup(p1, p2)

    def plot(self, plt):
        for i in range(len(self.players)):
            self.players[i].plot(plt, Players.P1)
            self.players[i].plot(plt, Players.P2)
            plt.legend()
            plt.show()



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You must supply the csv file containing the payoff matrix as a command line argument.")
        file = None
    else:
        file = sys.argv[1]
    
    if file:
        print(f"Reading {file} and building payoff matrix...")
        payoff = PayoffMatrix(file)
        print(payoff.df)
        rr = RoundRobin(payoff, 10)
        for i in range(50):
            rr.run_session()
        print(rr.plot(plt))
