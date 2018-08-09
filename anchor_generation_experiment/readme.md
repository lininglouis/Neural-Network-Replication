base anchor with raio 0.5, 1, 2, base_size

![base_anchors](/home/lining/Gluon_learn/Neural-Network-Replication/anchor_generation_experiment/base_anchors.png)

after shift the base anchor by 

```
    shifts = np.array([[  0, 0 , 0, 0], 
                       [500, 500 ,500 , 500],
                       [  0, 500 , 0, 500], 
                       [500, 0, 500, 0]])
```

We get the all anchors

![all_anchors](/home/lining/Gluon_learn/Neural-Network-Replication/anchor_generation_experiment/all_anchors.png)

