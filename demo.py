# import numpy as np
#
# trigger = np.load("./trigger/trojan_trigger.npz")
#
# print(trigger["x"])
# # trigger_new = {'x': trigger["x"] * 2, 'y': 1}
# # np.save("./trigger/trojan_trigger", trigger_new)
str = """
90.96	99.91	87.58	8.97	86.46	2.12	89.66	5.26	87.89	1.48	86.42	3.73	86.88	3.33	85.83	0.18	87.55	3.28	84.46	3.06	86.33	11.84
"""
result = []
print(str.split("\n"))
# for re in str.split("\n"):
#     if len(re) > 0 and re[0] == 'a':
#         result.append(re[6:8] + "." + re[8:10])
#         result.append(re[17:19] + "." + re[19:21])
# length = len(result)
# for i in range(0, length, 1):
#     print(result[i], end="\t")

for re in str.split("\t"):
    if len(re) <= 0:
        continue
    result.append(re)
length = len(result)
for i in range(0, length, 2):
    print(result[i], end="\t")
    print(result[i + 1])
