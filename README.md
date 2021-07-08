UW Madison Spring 2021 Project - Heirarchical Agglomerative Clustering

Pokemon.csv contains stats for a large number of pokemon, and calculate_x_y sums their attack and defense.
This tuple is organized in a list of the first 20 pokemon in the example, instantiated by
	python3 pokemon_stats.py <base> <offset>
where base is where the range begins in the pokedex, up to 721 and
offset is the defined range after the base.

HAC will organize and build a tree to represent each
data point's relationship to another, and imshow is used to neatly plot
the data. The primary challenge in this assignment was to implement HAC
on our own without using scipy.linkage(), which would yield almost the
same results.
