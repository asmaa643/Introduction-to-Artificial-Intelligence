
*****
Comments:
Question 7:
Function Description

The evaluation function for the 2048 game assesses the quality of a game state using key heuristics,
chosen based on gameplay analysis and testing.
As written in the Hints and Observations, we computed values for features about the states that we think are
important, we watch videos and learn strategies, and combined those features by multiplying them by different
values and adding the results together. Weights were initially equal and then adjusted through gameplay 
simulations to improve performance.

First we kept the same strategy from question 1: Merges: merge tiles in subsequent moves is crucial for creating 
	higher-value tiles and freeing up space, thus we Count adjacent equal tiles.
Adding new useful strategies:
- Free Tiles: Count zero tiles on the board, more empty tiles allow for new tile placement and reduce the risk 
	of running out of moves.
- Tile Value: Sum all tile values on the board, aiding in reaching higher tiles like 2048.
- Smoothness: Calculate the sum of absolute differences between adjacent tiles in rows and columns, because 
	smaller differences between adjacent tiles make the board easier to manage and merge.
- Monotonicity: Check if rows and columns are non-decreasing or non-increasing, for easier merging.