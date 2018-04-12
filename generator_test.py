def generator():
    i = 0
    while True:
    i += 1
    yield i

for item in generator():
    print(item)
    if item > 4:
    break
