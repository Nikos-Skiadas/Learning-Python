# import sys

# Prison Dilemma with tit for tat method
def prison_dilemma():
    algorithm_move = 'C'

    actual_profit = 0
    maximum_profit = 0

    print("Type 'C' for Cooperate or 'D' for Defect. Leave input empty to stop playing.")

    while True:
        print(algorithm_move)

        player_move = input().upper() # input to take player move and upper to capitalize

        if not player_move:
            break

        if player_move not in ['C', 'D']:

            print("Invalid input. Please type 'C' or 'D'.")
            continue

		# Maximum profit
        if player_move == 'C':
            maximum_profit += 5
        else:
            maximum_profit += 1

        if algorithm_move == 'C' and player_move == 'C':
            actual_profit += 3
        elif algorithm_move == 'C' and player_move == 'D':
            actual_profit += 0
        elif algorithm_move == 'D' and player_move == 'C':
            actual_profit += 5
        elif algorithm_move == 'D' and player_move == 'D':
            actual_profit += 1

        algorithm_move = player_move

    print("\nGame over.")
    print(f"Maximum possible profit: {maximum_profit}")
    print(f"Actual profit: {actual_profit}")


if __name__ == "__main__":
	prison_dilemma()
