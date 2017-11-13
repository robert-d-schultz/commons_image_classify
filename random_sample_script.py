# random sample

# standard error = sqrt(p(1-p)/n)

# results:

# population of Commons PD logos
# % for error type #1 (above threshold) = 18%
# standard error = 0.05433231082
# 95% confidence interval = 18.00% ± 10.65%

# population of Wikipedia non-free logos
# % for error type #4 (below threshold) = 30%
# standard error = 0.06480740698
# 95% confidence interval = 30.00% ± 12.70%


import csv
import random as r

with open("./Commons_PD_text_logos.csv") as f:
    reader = csv.reader(f)
    chosen_rows = r.sample(list(reader),50)
    strs = ["https://commons.wikimedia.org/wiki/" + l[1] for l in chosen_rows]
    [print(st) for st in strs]
