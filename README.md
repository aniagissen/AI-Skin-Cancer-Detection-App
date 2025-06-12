# AI Skin Cancer Detection App

A Streamlit-based app prototype that diagnoses seven types of skin lesions using deep learning, with visual explanations (Grad-CAM) and tailored AI skincare advice via LLaVA-7B.

---

## Dataset

This project was built using the [HAM10000 dataset](https://www.kaggle.com/datasets/surajghuwalewala/ham10000-segmentation-and-classification), which includes thousands of dermatoscopic images across the following classes:

- Actinic keratoses
- Basal cell carcinoma
- Benign keratosis-like lesions
- Dermatofibroma
- Melanocytic nevi
- Melanoma
- Vascular lesions

---

## Model + File Downloads

Please download the pre-trained model file (`skin_cancer_model.pth`) and place it inside the project folder:

[Download Model File](https://artslondon-my.sharepoint.com/:u:/g/personal/a_gissen0620241_arts_ac_uk/EadLfF8fS0BJkro3ty4T3aYBzGP2Dx6dX6H72498-p9Ht?w=e-2kBlKW)

If you want to test the model on sample images, use the following folder of example moles (already labelled by diagnosis):

[Download Example Mole Images](https://artslondon-my.sharepoint.com/:f:/g/personal/a_gissen0620241_arts_ac_uk/EovQpXFdRyxKu5sbjZPvIR8BAS8IV2QoQsV41lUcpkHMsg?e=nA8a8h)

---

## Setup Instructions

In your terminal, run the following:

```
# Create and activate the environment
conda create --name SkinCancerAI python=3.12
conda activate SkinCancerAI

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
cd /path/to/AI-Skin-Cancer-Detection-App/Streamlit-App
streamlit run Run-App.py
