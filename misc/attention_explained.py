import torch
import math

d_k = 1

##########################
# Attention with masking #
##########################

# The starting point is a vector of tokens:
# [50000, 14, 56, 134, ...]
# Then, we use learnable embeddings to translate numbers into d_model-dimensional vectors:
# [[embedding1],
#   [1243, 12541, 4, 85 ,7979, ...]
#   [embedding3]]
# At this stage, we have a matrix where each row (in PyTorch, but columns in the paper) represents the embedding of a token. 
# Now, we can proceed to evaluate self-attention.

query = key = value = torch.tensor(
    [
        [2,2,2],    # token1
        [3,3,3],    # token2
        [0,0,0],    # token3
    ],
    dtype=torch.float
)
print('query', query)

# The attention mechanism is not learnable; instead, it constrains backpropagation values in a way that teaches all layers, 
# including the linears and FFNs before and after the attention, to follow the attention pattern.
# In this example, I've set the last embedding to a zero-vector for better visualization of the scores.

scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    # Q*(K^T)  /  sqrt(d_k)
print('scores', scores)

# tensor([[12., 18.,  0.],
#         [18., 27.,  0.],
#         [ 0.,  0.,  0.]])

# Now, we enter the territory of masks!
# The attention mechanism uses two types of masks:

# (1) The first one is a subsequent mask - this mask prevents the decoder from generating output based on the following input words.

def subsequent_mask(size):
    attn_shape = (1, size, size)
    # setting all diagonals above main+1 diagonal to be ones
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

masks = subsequent_mask(query.size(-1))
print('subsequent_mask', masks)

# tensor([[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]])

# It may seem strange that the True values are in the bottom-left corner, but they are applied to future tokens.
# It's important to note that in the paper's notation, embeddings are represented by columns (unlike rows in torch tensors), so keep in mind to transpose.
# Additionally, remember that the mask defines relations between words, so each word should have a relation to the previous one but not to the next.

print('masks.transpose', masks.transpose(-2, -1))

# tensor([[[ True,  True,  True],
#          [False,  True,  True],
#          [False, False,  True]]])

# The first position has no relation (only self-relation), the second knows about the previous ones, and the last knows about the entire sentence.

# (2) The second type of masks is padding masks.
# When processing sentences in batches, PyTorch tensors need elements of the same size to be processed efficiently (a batch is a high-dimensional tensor).
# Therefore, we align the lengths of our sentences. Assuming the max length in our batch is 3, and the example we're using has a length of 2, we add a 
# pad_token (with a embedding [0, 0, 0]) to our sentence. Now, let's create a binary mask for the sentence:

padding_mask = torch.tensor([[True, True, False]])


# The nature of the mask is obvious. Applying this mask to the scores aims to set -inf weights 
# for connections between pad tokens and real tokens, and vice versa for real tokens to pads.

def padding_masks(binary_masks):
    masks = torch.zeros(binary_masks.size()[0], binary_masks.size()[1], binary_masks.size()[1])
    masks_length = binary_masks.sum(dim=1)
    for i in range(binary_masks.size()[0]):
        masks[i, :masks_length[i], :masks_length[i]] = 1
    return masks.type(
        torch.bool
    )
print('padding_masks', padding_masks(padding_mask))

# tensor([[[ True,  True, False],
#          [ True,  True, False],
#          [False, False, False]]])

# That makes sense! Now, the last row ([False, False, False]) indicates that the padding token should not relate to any real token.
# The first two rows, 
# [True, True, False],
# [True, True, False],
# indicate that the real tokens should not relate to the pads either.

masks = masks * padding_masks(padding_mask) # * is an element-wise multiplication
print('combined masks', masks)

# tensor([[[ True, False, False],
#          [ True,  True, False],
#          [False, False, False]]])

scores = scores.masked_fill(masks == 0, float('-inf'))
print('scores', scores)

# tensor([[[12., -inf, -inf],
#          [18., 27., -inf],
#          [-inf, -inf, -inf]]])

p_attn = torch.softmax(scores, dim = -1)
print('p_attn', p_attn)

# tensor([[[1.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.2339e-04, 9.9988e-01, 0.0000e+00],
#          [       nan,        nan,        nan]]])

# Now we have scores of relations of every element in a sentence and we want to say every embedding in the sentence how it relates to the other elements of the sequence.

result = torch.matmul(p_attn, value)
print('result', result)

# tensor([[[2.0000, 2.0000, 2.0000],
#          [2.9999, 2.9999, 2.9999],
#          [   nan,    nan,    nan]]])
