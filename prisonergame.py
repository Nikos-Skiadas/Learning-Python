"""Exercise:

Must:
-   Improve interface messages. For example if bad strategy, list the good ones.
-   Also, should we have a default strategy? If yes, which? How to handle this?
-   Implement other strategies and properly list them:
    -   tit_for_tat
    -   win-stay-lose-switch
    -   always-cooperate
    -   always-defect
    -   ...

Bonus:
-   Support fuzzy strategies (with probabilities). Should probabilities complement each other?? Which is so?? If NOT, why??
-   Start working on an object oriented implementation, that may include (up to you) the following entities:
    -   `Strategy` (do we really need this or is `Player` enough?)
    -   `Player` (perhaps track their score and select a strategy?)
    -   `Game` (perhaps track the turns and run the game loop?)
    -   ...
    Think how those entities interact with one another. For example:
    -   a `Player` must have a `Strategy`
    -   a `Game` must have two `Players`
    -   ...
"""


import sys

default_strategy = "tit_for_tat"

# Whether algorithm will cooperate:
# Collection of standard strategies:
strategies = {
    "tit_for_tat": {
        'C': {
            'C': True,
            'D': False,
        },
        'D': {
            'C': False,
            'D': True,
        },
    },
    "win_stay_lose_switch": {
        'C': {
            'C': True,
            'D': False,
        },
        'D': {
            'C': True,
            'D': False,
        },
    },
    "always_cooperate": {
        'C': {
            'C': True,
            'D': True,
        },
        'D': {
            'C': True,
            'D': True,
        },
    },
    "always_defect": {
        'C': {
            'C': False,
            'D': False,
        },
        'D': {
            'C': False,
            'D': False,
        },
    },
}


# Prison Dilemma with tit for tat method
def prison_dilemma(strategy_name: str):
    profits = {
        'C': {
            'C': 3,
            'D': 0,
        },
        'D': {
            'C': 5,"tit_for_tat"

            'D': 1,
        },
    }
    maximum_profits = profits['D']

    strategy = strategies[strategy_name]
    maximum_profits = profits['D']
    algorithm_move = 'C'

    turn = 1
    actual_profit = 0
    maximum_profit = 0

    print("Type 'C' for Cooperate or 'D' for Defect. EOF for endgame.\n")

    while True:
        print(f"Round {turn}:")

        # Input to take player move and upper to capitalize:
        try:
            print(f"player 1: {algorithm_move}")

            player_move = input("player 2: \033[K")


            if player_move not in ['C', 'D']:
                print("Invalid input. Please type 'C' for Cooperate or 'D' for Defect.\n")
                continue

            # Score algorithm:
            maximum_profit += maximum_profits[player_move]
            actual_profit += profits[algorithm_move][player_move]

            # Next turn:
            algorithm_move = 'C' if strategy[algorithm_move][player_move] else 'D'
            turn += 1

        except KeyError:
            print("\033[4A")
            continue

        except EOFError:
            break

    print("\nGame over.")
    print(f"Rounds: {turn - 1}")
    print(f"Maximum possible profit: {maximum_profit}")
    print(f"Actual profit: {actual_profit}")


if __name__ == "__main__":
    try:
        prison_dilemma(sys.argv[1] if len(sys.argv) > 1 else default_strategy)

    except KeyError:
        print("non-existent strategy")
