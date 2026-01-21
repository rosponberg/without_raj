"""
Nash Equilibrium through Simulation

Author        Rowan Sponberg
Date          01/20/2026

This program uses repeated simulated gameplay to allow Nash equilibrium
strategies to emerge without analytically solving the payoff matrix.
Players iteratively update their strategy preferences based solely on
their realized payoffs relative to historical performance.

The program is run from the command line and accepts a CSV file
representing a payoff matrix. The simulation performs multiple
round-robin sessions among a fixed group of players and visualizes
strategy convergence using matplotlib.

AI Usage:     
    Docstring generation (manually edited for correctness)

Sources:
    ChatGPT
"""

from enum import Enum
import random
import pandas as pd
import sys
import matplotlib.pyplot as plt


class Players(Enum):
    """
    Enum representing player roles within the payoff matrix.

    P1 corresponds to the row player.
    P2 corresponds to the column player.
    """
    P1 = "row"
    P2 = "col"


class PayoffMatrix:
    """
    Represents a two-player payoff matrix loaded from a CSV file.

    The CSV is expected to contain strategy labels as row and column
    headers, with each cell storing two numeric payoffs (one per player).
    """

    def __init__(self, path):
        """
        Loads and parses the payoff matrix from a CSV file.

        Args:
            path (str): Path to the CSV file containing the payoff matrix.
        """
        self.df = pd.read_csv(path, index_col=0)
        self.df = self.df.map(
            lambda x: [float(s) for s in x.split(" ") if s != ""]
        )
        self.name = path

    def evaluate(self, s1, s2):
        """
        Evaluates a game outcome for two strategies.

        Args:
            s1 (str): Strategy chosen by the row player.
            s2 (str): Strategy chosen by the column player.

        Returns:
            dict: A dictionary containing strategies, payoffs, winner,
                  loser, and draw status.
        """
        payoffs = self.df.loc[s1, s2]
        payoff1 = payoffs[0]
        payoff2 = payoffs[1]
        tie = payoff1 == payoff2

        return {
            Players.P1: {"Strategy": s1, "Payoff": payoff1},
            Players.P2: {"Strategy": s2, "Payoff": payoff2},
            "Winner": None if tie else (
                Players.P1 if payoff1 > payoff2 else Players.P2
            ),
            "Loser": None if tie else (
                Players.P1 if payoff1 < payoff2 else Players.P2
            ),
            "Draw": tie
        }


class Player:
    """
    Represents a player that adapts strategy preferences
    through repeated gameplay.

    Each player maintains independent preference distributions
    for row and column roles and updates them using payoff feedback.
    """

    def __init__(self, payoff):
        """
        Initializes player preferences and logging structures.

        Args:
            payoff (PayoffMatrix): The payoff matrix used in the game.
        """
        self.weights = {Players.P1: {}, Players.P2: {}}
        self.log = {Players.P1: {}, Players.P2: {}}

        for row in payoff.df.index:
            self.weights[Players.P1][row] = 1 / len(payoff.df.index)
            self.log[Players.P1][row] = [self.weights[Players.P1][row]]

        for col in payoff.df.columns:
            self.weights[Players.P2][col] = 1 / len(payoff.df.columns)
            self.log[Players.P2][col] = [self.weights[Players.P2][col]]

        self.lr = 0.05
        self.scores = []

    def getStrategy(self, player):
        """
        Selects a strategy probabilistically based on current preferences.

        Args:
            player (Players): The role (row or column) being played.

        Returns:
            str: The chosen strategy.
        """
        r = random.random()
        lower = 0
        weights = self.weights[player]

        for key in weights:
            if lower <= r < lower + weights[key]:
                return key
            lower += weights[key]

        return list(weights.keys())[-1]

    def versus(self, p2, payoff):
        """
        Plays a single game against another player.

        Args:
            p2 (Player): The opposing player.
            payoff (PayoffMatrix): The payoff matrix.

        Returns:
            dict: Game outcome produced by PayoffMatrix.evaluate().
        """
        s1 = self.getStrategy(Players.P1)
        s2 = p2.getStrategy(Players.P2)
        return payoff.evaluate(s1, s2)

    def log_score(self, score):
        """
        Logs payoffs and current strategy preferences.

        Args:
            score (dict): Payoffs from the most recent game.
        """
        self.scores.append(score)

        for p in self.weights:
            for strat in self.weights[p]:
                self.log[p][strat].append(self.weights[p][strat])

    def _update_factor(self, score, player):
        """
        Computes the reinforcement adjustment factor.

        Args:
            score (float): Payoff from the current game.
            player (Players): Player role associated with the score.

        Returns:
            float: Adjustment value for the strategy preference.
        """
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
        """
        Normalizes strategy preferences so they sum to 1.0.

        Args:
            player (Players): Player role whose preferences are normalized.
        """
        s = sum(self.weights[player].values())
        for k in self.weights[player]:
            self.weights[player][k] = max(self.weights[player][k], 0)

        for k in self.weights[player]:
            self.weights[player][k] /= s

    def update(self, p1Results, p2Results):
        """
        Updates strategy preferences after a game.

        Args:
            p1Results (dict): Outcome for the row role.
            p2Results (dict): Outcome for the column role.
        """
        p1Strategy = p1Results["Strategy"]
        p2Strategy = p2Results["Strategy"]
        p1Score = p1Results["Payoff"]
        p2Score = p2Results["Payoff"]

        self.log_score({
            Players.P1: p1Score,
            Players.P2: p2Score
        })

        self.weights[Players.P1][p1Strategy] += \
            self._update_factor(p1Score, Players.P1)
        self.weights[Players.P2][p2Strategy] += \
            self._update_factor(p2Score, Players.P2)

        self._normalize_weights(Players.P1)
        self._normalize_weights(Players.P2)

    def plot(self, plt, player):
        """
        Plots strategy preference evolution over time.

        Args:
            plt (module): matplotlib.pyplot module.
            player (Players): Player role to plot.
        """
        series = self.log[player]
        role = "P1" if player == Players.P1 else "P2"

        for key in series:
            plt.plot(
                series[key],
                label=f"{key}, as {role} ({player.value})"
            )


class RoundRobin:
    """
    Manages a round-robin tournament among multiple players.
    """

    def __init__(self, payoff, numPlayers=10):
        """
        Initializes players and tournament parameters.

        Args:
            payoff (PayoffMatrix): The payoff matrix.
            numPlayers (int): Number of players in the tournament.
        """
        self.payoff = payoff
        self.numPlayers = numPlayers
        self.players = [Player(payoff) for _ in range(numPlayers)]

    def matchup(self, p1, p2):
        """
        Conducts a two-game matchup between two players.

        Args:
            p1 (int): Index of the first player.
            p2 (int): Index of the second player.
        """
        p1 = self.players[p1]
        p2 = self.players[p2]

        game1 = p1.versus(p2, self.payoff)
        game2 = p2.versus(p1, self.payoff)

        p1.update(game1[Players.P1], game2[Players.P2])
        p2.update(game2[Players.P1], game1[Players.P2])

    def run_session(self):
        """
        Runs a full round-robin session.
        """
        l = len(self.players)
        midpoint = l // 2

        for rnd in range(l - 1):
            for i in range(midpoint):
                p1 = 0 if i == 0 else (i + rnd - 1) % (l - 1) + 1
                p2 = (l - i - 2 + rnd) % (l - 1) + 1
                self.matchup(p1, p2)

    def plot(self, plt):
        """
        Plots strategy evolution for all players.
        
        Args:
            plt (matplotlib.pyplot): the matplotlib pyplot object to plot on
        """
        for player in self.players:
            plt.title(self.payoff.name)
            player.plot(plt, Players.P1)
            player.plot(plt, Players.P2)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You must supply the csv file containing the payoff matrix.")
        sys.exit(1)

    file = sys.argv[1]
    print(f"Reading {file} and building payoff matrix...")

    payoff = PayoffMatrix(file)
    rr = RoundRobin(payoff, 10)

    for _ in range(50):
        rr.run_session()

    rr.plot(plt)

