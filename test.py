def position_map(pos):
    x, y = 13, 0
    count = 28
    while pos >= count:
        x -= 1
        y += 1
        pos -= count
        count -= 2
    return x, y + pos


# for i in range(200):
print(position_map(209))