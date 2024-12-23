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

### Customizing and Evaluating Network Components  

To better understand the impact of specific components on the network's performance, you can selectively disable certain features by removing them (commenting out the corresponding code). Below are some examples:  

- **Disabling Branches for Analysis**:  
In the network module, locate the section labeled # third branch. By removing the lines (commenting them out) in this section, you can evaluate the effect of excluding the regularization branch on the network's training and performance. Similarly, for analyzing the impact of the global branch, you can follow the same approach by identifying and removing (commenting out) the lines corresponding to the global branch.

- **Adjusting the Loss Function**:  
  The network uses a combination of Softmax loss and Triplet loss, defined in the `main_reid.py` module. You can remove (comment out) the lines corresponding to each loss function to analyze their individual contributions to the network's learning process.  

This approach allows for flexible experimentation, enabling you to adapt the network structure and loss functions to suit your research requirements.  

## Datasets Used
This project uses multiple publicly available datasets commonly used for person re-identification tasks. Below is a list of the datasets and their access information

### Market1501 and DukeMTMC
These datasets can be downloaded from Kaggle.

### CUHK03-l and CUHK03-d
These datasets are available through the following Google Drive link:

- [Download CUHK03-l and CUHK03-d](https://drive.google.com/file/d/1pBCIAGSZ81pgvqjC-lUHtl0OYV1icgkz/view)


### Training Market1501 

To train the network on the Market1501 dataset, execute the following command:  
```bash  
python main_reid.py train --save_dir='/content/out' --max_epoch=420 --eval_step=30 --dataset=market1501 --test_batch=64 --train_batch=64 --optim=adam --adjust_lr
```

### Testing Market1501
 
To test the network on the **Market1501** dataset, execute the following command:  
```bash  
python main_reid.py train --save_dir='/content/out' --model_name=CSA_TOPDB --train_batch=64 --test_batch=64 --dataset=market1501 --pretrained_model='/content/out/model_best.pth.tar' --evaluate
```
