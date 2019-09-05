import preprocessor as pr

d = pr.get_original_data("data/breast-cancer-wisconsin.data")
d = pr.remove_missing_rows(d)


