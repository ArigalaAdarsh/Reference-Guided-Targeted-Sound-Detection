# SHARED REPRESENTATION LEARNING FOR REFERENCE-GUIDED TARGETED SOUND DETECTION (ICASSP 2026 ACCEPTED)

This repository provides the **official implementation** of our unified encoder model for **Target Sound Detection (TSD)**, submitted to **ICASSP 2026**.  
It includes training and evaluation for the **URBAN-SED** and **URBAN*8K** datasets.  
Note: Pretrained model weights will be released later to reproduce the results.
## Datasets
- **UrbanSound8K**: [Download here](https://urbansounddataset.weebly.com/urbansound8k.html)  
- **URBAN-SED**: [Download here](https://zenodo.org/records/1324404)  
- Keep these in the datasets/URBAN-SED
- datasets/ UrbanSound8K

#  Download Pre-trained Weights

1. Download the pre-trained weights from [Google Drive](#).  
2. Place the downloaded weights into the following directory:

   ```bash
   experiments/model_output/
3. In the Evolution Notebook (inside the notebooks/ folder), update the path to the weights and run evaluation to reproduce our results.
 
## Installation

```bash
git clone https://github.com/your-username/Reference-Guided-TSD-Unified-Encoder.git
cd Reference-Guided-TSD-Unified-Encoder
```


Install python >= 3.8

### 1. Install Requirements
Upgrade pip and install dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

2. Load the ConvNeXt checkpoint for the audio encoder (trained on the AudioSet2M). Download `convnext_tiny_471mAP.pth` from [Zenodo](https://zenodo.org/records/8020843) and place it in the `convnext` folder.  

## Preparing the Dataset
To make the dataset ready, run:

```bash
python data/extract_feature.py
```

Inside the script, change the parameters accordingly:

```python
what_type = 'train'  # options: 'train', 'val', 'test'
```
and update the respective dataset path locations.

To experiment with the Strong Plus dataset setup, prepare the dataset by running:

```bash
python data/extract_convnext_feature.py
```

Inside the script, change the parameters accordingly:

```python
what_type = 'train'  # options: 'train', 'val', 'test'
```
and update the respective dataset path locations.



## Training
To train the model, run:

```bash
bash bash/tsd.sh
```

## Evaluation
Check the `notebooks/` folder for evaluation scripts.  
For example, you can run:

```bash
Evaluating_ConvNeXt_multiplication.ipynb
```

---
