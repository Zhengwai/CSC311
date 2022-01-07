def f1(v):
    return v**2/(1-v**2)**3


def f1n(v, r):
    sum = 0
    for i in range(r):
        sum += ((-1)**i+1)*(i*i+2*i)*(v**i)
        print(((-1)**i+1)*(i*i+2*i)*(v**i))
    return sum/16


def f1m(v, r):
    sum = 0
    for i in range(r):
        sum += (i+2)*(i + 1) * (v ** (2*i+2))
    return sum / 2

def f2(v):
    return 2/(v+1)**3

def f2n(v, r):
    sum = 0
    for i in range(r):
        sum += (((-1)**i)*(i+2)*(i+1))*(v**i)
    return sum

def f3(v):
    return 1/(v+1)**2

def f3n(v, r):
    sum = 0
    for i in range(r):
        sum += (((-1)**i)*(i+1))*(v**i)
    return sum

def f4(v):
    return 1/(v+1)

def f4n(v, r):
    sum = 0
    for i in range(r):
        sum += (-1)**i*v**i
    return sum

def f5(v):
    return 2/(v-1)**3

def f5n(v, r):
    sum = 0
    for i in range(r):
        sum += (i+2)*(i+1)*(v**i)
    return -1*sum

def f6(v):
    return 1/(v-1)**2

def f6n(v, r):
    sum = 0
    for i in range(r):
        sum += (i+1)*(v**i)
    return sum

def f7(v):
    return 1/(v-1)

def f7n(v, r):
    sum = 0
    for i in range(r):
        sum += (v**i)
    return -1*sum

def f8n(v, r):
    sum = 0
    for i in range(1,r):
        sum += (v**i)/(1-v**i)
    return sum


print(f1(0.7))
print(f1n(0.7, 3))
print(f1m(0.7, 1))