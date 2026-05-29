
def approach5(res1, res2, res3, res4, grades, params):
    a = params['A']
    b = params['B']
    c = params['C']
    d = params['D']
    results = []
    for rs in res1:
        g1 = rs[1]

        g2 = 0
        for r in res2:
            if r[0] == rs[0]:
                g2 = r[1]
        g3 = 0
        for r in res3:
            if r[0] == rs[0]:
                g3 = r[1]

        g4 = 0
        for r in res4:
            if r[0] == rs[0]:
                g4 = r[1]
        #print(g1,g2,g3,g4)
        pg = a * g1 + b * g2 + c * g3 + d * g4
        results.append((rs[0], pg))

    return results