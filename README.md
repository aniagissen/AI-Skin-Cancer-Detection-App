# AI-Skin-Cancer-Detection-App
Code for a streamlit app to diagnose skin lesions into seven types of skin cancer. 

This was built using the HAM10000 dataset from the model - https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification for a project using AI for Media.

Please download the model file path - https://artslondon-my.sharepoint.com/:u:/g/personal/a_gissen0620241_arts_ac_uk/EaIdLF8fS0BJkro3ty4T3aYBzGP2Dx6dX6H72498-p9Htw?e=2kBlKW

I have also added a folder of moles, named by what they should be diagnosed as, if you want to use those as examples rather than having to search for some. https://artslondon-my.sharepoint.com/:f:/g/personal/a_gissen0620241_arts_ac_uk/EovQpXFdRyxKu5sbjZPvlR8BAS8lV2oQsV41lUcplkHMsg?e=aN8a8h


Then run this code in your terminal:

```
conda create --name SkinCancerAI python=3.12

conda activate SkinCancerAI

pip install -r requirements.txt

cd /path/to/my/project/folder

streamlit run RunApp.py
```

