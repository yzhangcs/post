# -*- coding: utf-8 -*-


def pad(sequences, batch_first=False, padding_value=0):
    max_size = max(sequences, key=len).size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        prev_l = length
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor
