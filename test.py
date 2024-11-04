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


def game_pos_map(x, y):
    count = 28
    res = 0
    if x >= 14:
        while x > 14:
            res += count
            count -= 2
            x -= 1
        res += y - (28 - count) // 2 + 213
    else:
        while x < 13:
            res += count
            x += 1
            count -= 2
        res += y - (28 - count) // 2
    return res


print(game_pos_map(27, 14))


import torch

# 示例输入张量
input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 计算softmax结果
softmax_result = torch.softmax(input_tensor, dim=0)

# 选取softmax结果大于0.1的元素及其索引
selected_indices = torch.nonzero(softmax_result > 0.1).squeeze().numpy()
selected_elements = softmax_result[softmax_result > 0.1]

print("Softmax结果:", softmax_result)
print("选取的元素:", selected_elements)
print("选取元素的索引:", selected_indices)