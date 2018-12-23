This is a `pytorch` implementation of [Associative Domain Adaptation](https://arxiv.org/abs/1708.00938).

## Results

| val :arrow_down:, train :arrow_right:  | just svhn | just mnist | svhn to mnist | mnist to svhn  |
|---|---|---|---|---|
| svhn | 93.9  | 60.3  | 95.3  | 85.6  |
| mnist |  59.3 |  99.4 |  98.3 | 99.6  |

## Notes:
1. I used a colored MNIST instead of the original MNIST.  
During training and evaluation I randomly colored backgrounds and digits.

## Training curves
![svhn to mnist](images/losses1.png)
![mnist to svhn](images/losses2.png)

## Requirements
1. pytorch 1.0
2. numpy, Pillow

## Other implementations
1. [stes/torch-associative](https://github.com/stes/torch-associative)
2. [haeusser/learning_by_association](https://github.com/haeusser/learning_by_association)
