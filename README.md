The codes available in this repository developed based on a BإDB network, whose code is available at: “https://github.com/daizuozhuo/batch-dropblock-network”
The folder “models”  includes python codes with the names initiated with network, each of which related to a evaluated network in our study:
•	network_se.py: proposed network with CAS module including SE (+triplet loss)
•	network_eca.py: proposed network with CAS module including ECA (+triplet loss)
> **⚠️ Note:** Before running the training, rename the desired network file to `networks.py`.


### Training Dukemtmc  
```bash  
python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=128 --train_batch=128 --optim=adam --adjust_lr





2-python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=420 --eval_step=30 --dataset=market1501 --test_batch=64 --train_batch=64 --optim=adam --adjust_lr


