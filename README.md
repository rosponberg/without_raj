# Nash Equilibrium through Simulation

**Author:** Rowan Sponberg  
**Course:** Modeling & Simulation (Fall 2025)  
**Project:** Final Project – Nash Equilibria through Simulation  

---

## Overview

This project explores how **Nash equilibrium strategies can emerge through repeated gameplay**, rather than being identified analytically. Players iteratively update their strategy preferences based only on realized payoffs and historical performance.

Players do **not** know the payoff matrix in advance. Instead, they learn through experience by reinforcing strategies that perform better than their historical average. Over many games, this process causes players’ strategies to converge toward Nash equilibria.

The project supports multiple games, including those with:
- A single pure-strategy Nash equilibrium
- Multiple pure-strategy Nash equilibria
- A mixed-strategy Nash equilibrium

---

## How the Simulation Works

- A fixed group of **10 players** participates in repeated gameplay.
- Each session consists of a **round-robin tournament** (45 games per session).
- The same players persist across sessions, carrying forward their learned strategy preferences.
- Each player maintains a probability distribution over available strategies.
- After each game, players update their preferences based on:
  - The payoff earned in the current game
  - Their historical average payoff
- Preferences always sum to 1.0 and represent the probability of choosing each strategy.

This process is repeated for **50 sessions (2250 total games)** to allow convergence.

---

## Supported Games

The project was tested on the following games:

1. **Prisoner’s Dilemma**  
   - Single pure-strategy Nash equilibrium  
   - Expected convergence: *(confess, confess)*

2. **Stag and Hare**  
   - Two pure-strategy Nash equilibria  
   - Expected convergence: *(stag, stag)* or *(hare, hare)*

3. **Battle of the Sexes**  
   - Two pure-strategy Nash equilibria  
   - Expected convergence depends on early randomness

4. **Rock-Paper-Scissors (RPS)**  
   - No pure-strategy Nash equilibrium  
   - Expected convergence: mixed strategy

Each game is represented by a CSV file containing its payoff matrix.

---

## Running the Program
To run the program you must supply the payoff matrix file in the command line arguments. For example:

`python main.py battle_sexes.csv`

### Requirements
- Python 3.x
- `pandas`
- `matplotlib`

## AI Usage
AI was used to generate the docstrings and this readme file. Each were manually edited for accuracy and clarity.
