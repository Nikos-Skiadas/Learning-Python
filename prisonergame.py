# import sys

# Prison Dilemma with tit for tat method
def prison_dilemma():
    profits: dict[str, dict[str, int]] = {
        'C': {
            'C': 3,
            'D': 0,
        },
        'D': {
            'C': 5,
            'D': 1,
        },
    }
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

            # Score algorithm:
            maximum_profit += maximum_profits[player_move]
            actual_profit += profits[algorithm_move][player_move]

            # Next turn:
            algorithm_move = player_move
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
	prison_dilemma()
