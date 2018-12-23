This a `pytorch` implementation of [Associative Domain Adaptation](https://arxiv.org/abs/1708.00938).

## Results

| val dataset | just svhn | just mnist | svhn -> mnist | mnist -> svhn  |
|------|---|---|---|---|
|  svhn | 0.939  | 0.603  | 0.953  | 0.856  |
|   mnist   |  0.593 |  0.994 |  0.983 | 0.996  |

## Notes:
1.

## Requirements
1. pytorch 1.0
2. numpy, Pillow

## Other implementations
1. [stes/torch-associative](https://github.com/stes/torch-associative)
2. [haeusser/learning_by_association](https://github.com/haeusser/learning_by_association)
