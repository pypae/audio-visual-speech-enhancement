

def shiftn(s, n):
    s1 = ''
    for c in s:
        if c.isalpha():
            s1 += chr(ord(c) + n)
        else:
            s1 += c
    return s1

def rarechars(s):
    d = {}
    for c in s:
        if c not in d.keys():
            d[c] = 1
        else:
            d[c] += 1

    return d

