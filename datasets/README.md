In this directory we provide the meta-info of the datasets used in this paper.

## Vimeo

The official page of Vimeo dataset is [here](http://toflow.csail.mit.edu/). It has 64,612 training samples and 7,824 test samples. The complete dataset can be downloaded [here](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

In this paper we use Vimeo as the training set, the keys for lmdb files can be found at [Vimeo7_train_keys.pkl](./meta_info/Vimeo7_train_keys.pkl).

The test set of Vimeo is split into 3 subsets based on the movement: fast, medium and slow. The indices of videos for each subset can be found at [slow_testset.txt](/meta_info/slow_testset.txt), [medium_testset.txt](/meta_info/medium_testset.txt), and [fast_testset.txt](/meta_info/fast_testset.txt). Note that we remove videos with totally black frames to avoid NaN results during evaluation.


## Vid4

Vid4 is a 4-clip test set that has 171 frames in total. It can be downloaded [here](https://drive.google.com/drive/folders/10-gUO6zBeOpWEamrWKCtSkkUFukB9W5m).

In this paper, we use this dataset to measure the performance and runtime of different methods. 