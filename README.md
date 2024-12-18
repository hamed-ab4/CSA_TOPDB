## Code Dependencies and References

The codes in this repository were developed based on the **BDB network**, whose original implementation is available here:  
- (https://github.com/daizuozhuo/batch-dropblock-network

### Additional References:
We have also utilized parts of the code from the following repositories:  
- [Top DropBlock](https://github.com/RQuispeC/top-dropblock)  
- [APNet](https://github.com/chengy12/apnet)  




The folder “models”  includes python codes with the names initiated with network, each of which related to a evaluated network in our study:
•	network_se.py: proposed network with CAS module including SE (+triplet loss)
•	network_eca.py: proposed network with CAS module including ECA (+triplet loss)
> **⚠️ Note:** Before running the training, rename the desired network file to `networks.py`.


### Training Dukemtmc  
```bash  
python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=420 --eval_step=30 --dataset=Dukemtmc --test_batch=64 --train_batch=64 --optim=adam --adjust_lr





2-python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=420 --eval_step=30 --dataset=market1501 --test_batch=64 --train_batch=64 --optim=adam --adjust_lr


