# DND-Diko-WWWR

We present the Deep Nutrient Deficiency - Dikopshof - Winter Wheat and Winter Rye (DND-Diko-WWWR) dataset, which consists of 1,800 RGB images of winter wheat (WW2020) and 1,800 RGB images of winter rye (WR2021). The images were captured by a camera mounted on a UAV at three time points at the long-term fertilizer experiment (LTFE) Dikopshof near Bonn, Germany. The images were annotated with seven types of fertilizer treatments. The dataset is used for image classification. 

![Examples](https://raw.githubusercontent.com/image-hosting/ImageHosting/main/img/202306200458172.png)
*Example images*

## Download
The dataset can be downloaded from [PhenoRoam](https://phenoroam.phenorob.de/geonetwork/srv/eng/catalog.search#/metadata/2a2964ca-1120-4a07-9155-d910b399e97c).

The dataset directory should have this basic structure: 
```
DND-Diko-WWWR/
    ├── WW2020/
        ├── images/
            ├── 20200422_0.jpg
            └── ...
        ├── trainval.txt
        ├── test.txt
        └── labels_trainval.yml
    └── WR2021/
        ...
```
## Dataset Overview
- **UAV-Based RGB Images**: The proposed DND-Diko-WWWR dataset consists of 1,800 RGB images of winter wheat (WW2020) and 1,800 RGB images of winter rye (WR2021). 
- **Annotation**: The images were annotated with seven nutrient treatments:  unfertilized, _PKCa, N_KCa, NP_Ca, NPK_, NPKCa, NPKCa+m+s, where “_” stands for the omission of the corresponding nutrient (N: nitrogen, P: phosphorous, K: potassium, Ca: lime) and +m+s stands for additional application of mineral fertilizer and farmyard manure.
- **File Name**: The file was named as Date_RandomNumber.jpg, e.g. 20200506_0.jpg, where Date is the sampling date and RandomNumber $r \in[0, 1800)$. 
- **#Train:#Val:#Test**: 1332:468 (74%:26%). 

## Guide for the Challenge at CVPPA@ICCV'23

As part of the [8th Workshop on Computer Vision in Plant Phenotyping and Agriculture (CVPPA)](https://cvppa2023.github.io/) at the IEEE/CVF International Conference of Computer Vision (ICCV) 2023, we are organizing a [challenge](https://codalab.lisn.upsaclay.fr/competitions/13833) that aims at providing a solution to recognize nutrient deficiencies in winter wheat winter rye using UAV-based RGB images. 

In this challenge, one has to provide a zipped file of two text file named "predictions_WW2020.txt" and "predictions_WR2021.txt" with no headers. Each line in the text file should state the image filename, followed by the prediction index (0-6). The file should contain a prediction for each of the test images, corresponding to 468 lines of text. A sample submission file is shown below:

submission.zip
- predictions_WW2020.txt
```
20200506_0.jpg 5
20200506_11.jpg 0
...
20200422_1797.jpg 6
```
- predictions_WR2021.txt

An example of submission file can be download [here](https://github.com/jh-yi/DND-Diko-WWWR/tree/main/codalab/res_test). 

Teams will compete to provide the highest top-1 accuracy on the subset of winter wheat (WW2020) and winter rye (WR2021), respectively. The ranking will be based on the average accuracy of WW2020 and WR2021.

The best-performing solutions will be contacted by the workshop organizers after the end date of the challenge and we will have a part in the workshop to announce the winners of this challenge. Besides a certificate, the authors are invited to provide an overview of their technical approach.

**Note**
- It is allowed to use publicly available pre-trained models for weight initialization and transfer learning purposes.
- However, it is NOT permitted to use other data than the provided dataset for this competition.
- It is the participant's responsibility to divide the training set into proper training and validation splits if it is needed

### Training and Prediction

Please specify your data_root (path to the downloaded dataset) at `./configs/config.py`. We provide a baseline model based on ResNet-50 and Swin Transformer V2. You can now use the script `trainval.sh` to train the baseline model, and `test.sh'` to perform prediction on the test set and to generate the .txt file for submission to Codalab.

### Evaluation
The top-1 accuracy is used. We provide an evaluation script to test your model over the validation set (a subset of the provided trainval set generated by yourself).
Note that this script cannot be used to evaluate models over the test set, as we do not provide labels for the test set. It is good practice to ensure your predictions work with this script, as the same script is used on the evaluation server. 

Run `python ./codalab/evaluate.py` with the arguments as described in the script to evaulate the predictions. This creates a *scores.txt* file containing the metrics used for the challenge leaderboard.

### Submissions
The evaluation server is hosted using CodaLab. Submitting to the challenge requires a CodaLab account. 
Please find the evaluation server [here](https://codalab.lisn.upsaclay.fr/competitions/13833).
To participate in the challenge, you need to upload a .zip file of the .text files generated by the script provided before. 

### FAQ
Q: Can we use any other data than the data provdided?
A: No.

Q: Is data augmentation allowed?
A: Yes, as long as the augmentations are only applied to the provided data.

Q: Can we use two different models for WW2020 and WR2021?
A: No, the models should have the same architecture and hyper-parametrs, but you can finetune both models separately. 

## Cite


## License
This dataset follows Creative Commons Attribution Non Commercial Share Alike 4.0 Internation License.