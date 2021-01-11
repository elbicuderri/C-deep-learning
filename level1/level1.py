## Convolution

def Conv1d(sequence, kernel, padding=0, stride=1):
    output = sequence * kernel # * : convolution1d
    return output

Sequence = [3, 4, 5, 6, 1, 2, 3, 0, -1, 7]

Kernel_1 = [-1, 1]

Output_1 = Conv1d(Sequence, Kernel_1)

##======================================================================

def Conv2d(image, kernel, padding=0, stride=1):
    output = image * kernel # * : convolution2d
    return output

Image = [[2, 1, 3],
         [4, 2, 1],
         [0, 7, 2]]

Kernel_2 = [[1, -1],
          [-1, 0]]

Output_2 = Conv2d(Image, Kernel_2)

##======================================================================

def Conv2d_v2(image, kernel, padding=0, stride=1):
    output = image * kernel # * : convolution2d
    return output

Image_2 = [2, 1, 3, 4, 2, 1, 0, 7, 2]

Kernel_3 = [1, -1, -1, 0]

Output_3 = Conv2d_v2(Image_2, Kernel_3)





## implement padding, stride...

## maxpool, batch-normalization, dense, softmax, relu, conv-batchnorm-fusion