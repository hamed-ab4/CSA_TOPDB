## Code Dependencies and References

The codes in this repository were developed based on a network, whose original implementation is accessible at the following link: [https://github.com/daizuozhuo/batch-dropblock-network](https://github.com/daizuozhuo/batch-dropblock-network).




### Additional References:
We have also utilized parts of the code from the following repositories:  
- https://github.com/RQuispeC/top-dropblock
- https://github.com/chengy12/apnet 



The folder **“models”** includes Python codes with the names initiated with `network`, each corresponding to an evaluated network in our study:

- `network_se.py`: Proposed network with CAS module including SE 
- `network_eca.py`: Proposed network with CAS module including ECA 

> **⚠️ Note:** Before running the training, rename the desired network file to `networks.py`.



### Training Market1501 

To train the network on the Market1501 dataset, execute the following command:  
```bash  
python main_reid.py train --save_dir='/content/out' --max_epoch=420 --eval_step=30 --dataset=market1501 --test_batch=64 --train_batch=64 --optim=adam --adjust_lr
```

### Testing Market1501
 
To test the network on the **Market1501** dataset, execute the following command:  
```bash  
python main_reid.py train --save_dir='/content/out' --model_name=CSA_TOPDB --train_batch=64 --test_batch=64 --dataset=market1501 --pretrained_model='/content/out/model_best.pth.tar' --evaluate  ```
