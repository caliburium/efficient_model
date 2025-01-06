 - Imagenette requires Pytorch 2.2.1 or later
 - deeplake < 4.0

| Target Domain \ Target Class |     Supervised     | Unsupervised |
|:----------------------------:|:------------------:|:------------:|
|    Without Discriminator     |   MultiSource NN   |     CNN      |
|          Supervised          |   We'll gonna do   |     DANN     |
|         Unsupervised         | MMLD + TargetTrain |     MMLD     |



# Dataset Classes
|    | imagenette 224*224 | STL 96*96 | CIFAR10 32*32 | PACS     |
|----|--------------------|-----------|---------------|----------|
| 1  | Tench              | Airplane  | Airplane      | Dog      |
| 2  | English Springer   | Bird      | Birde         | Elephant |
| 3  | Cassette Player    | Car       | Car           | Giraffe  |
| 4  | Chain Saw          | Cat       | Cat           | Guitar   |
| 5  | Cassette Player    | Dear      | Dear          | Horse    |
| 6  | French Horn        | Dog       | Dog           | House    |
| 7  | Garbage Truck      | Horse     | Frog          | Person   |
| 8  | Gas Pump           | Monkey    | Horse         |          |
| 9  | Golf Ball          | Ship      | Ship          |          |
| 10 | Parachute          | Truck     | Truck         |          |
- PACS must be resized