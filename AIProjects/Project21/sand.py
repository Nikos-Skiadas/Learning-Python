from __future__ import annotations


import pandas as pd
import sklearn
import sklearn.linear_model


train_data = pd.DataFrame(
	dict(
		x = [
			28384,
			31008,
			42026,
			60236,
			42404,
		],
		y = [
			5.5,
			5.6,
			6.5,
			6.9,
			7.3,
		]
	),
	index = [
		"turkey",
		"hungary",
		"france",
		"united states",
		"new zealand",
	],
)
test_data = pd.DataFrame(
	dict(
		x = [
			48698,
			55938,
		],
		y = [
			7.3,
			7.6,
		]
	),
	index = [
		"australia",
		"denmark",
	],
)

regressor = sklearn.linear_model.LinearRegression()
regressor.fit(
	train_data[["x"]],
	train_data[["y"]],
)

print()
print(f"w_1 = {regressor.coef_}")
print(f"w_0 = {regressor.intercept_}")
print()

pred = regressor.predict(test_data[["x"]])

train_loss = regressor.score(
	train_data[["x"]],
	train_data[["y"]],
)
test_loss = regressor.score(
	test_data[["x"]],
	test_data[["y"]],
)
