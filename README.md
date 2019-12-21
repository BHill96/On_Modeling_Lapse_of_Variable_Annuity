# On_Modeling_Lapse_of_Variable_Annuity
This repo manages the code for the paper On Modeling Lapse of Variable Annuities. It contains four files:
1. CARA.py
2. Driver.py
3. Expected Returns.py
4. HARA.py

CARA, Expected Returns, and HARA all contain functions used to find the lapse boundary with their respective utility functions outlined in the paper. Expected Returns and HARA use a function called LapseBoundary, while CARA plots a graph using the right-hand side and the left-hand side of it's lambda equation. It's intersection is the lapse boundary.

The Driver contains some examples of how to call the functions. It also plots graphs using the same parameter sets used in the paper. Please note for t1 the paper shows the possibility of multiple lapse regions, and the code reflects that by outputing multiple intervals for t1 if they exist. To plot the graph we use the lowest bound, but taking the mean would also be an acceptable strategy.
