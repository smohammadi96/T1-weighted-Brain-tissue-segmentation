# Brain tissue segmentation
CycleFormer: Brain tissue segmentation in the presence of Multiple sclerosis lesions and Intensity Non-Uniformity artifact

## Download test dataset

Test dataset: [Drive](https://drive.google.com/file/d/1_Q0-VO_8xajg4ZtQYclq6t4fr0lANE8_/view?usp=sharing)

Labels (WM, GM, CSF, MS): [Drive](https://drive.google.com/file/d/1iyJalNDdhZaPUkYuV_KphLW8NpzvgNDz/view?usp=sharing)

T1_A2B [Drive](https://drive.google.com/file/d/16TVJTplWBkUty_VjE3F4-x2UuqodVfeP/view?usp=sharing)

**Note: T1_A2B data consists of MRI images without noise and INU which generated by our model. By downloading this data, it doesnt need to run generator first. You can skip generator step**

## Download pre-trained weights
T1-weighted segmentation models (Transformers): [Drive 1](https://drive.google.com/file/d/1sFtfAtIuaqd0XlQW225m0EDj9tYLDKTY/view?usp=sharing), [Drive 2](https://drive.google.com/file/d/1PiVRVKw2mQM3BL4HvEYiOW15N5MnZT9e/view?usp=sharing)

T1-weighted segmentation models (Attention-unet): [Drive](https://drive.google.com/file/d/18LsqKVm-cs8V2aR01HEvnXo-bebo0Akq/view?usp=sharing)

T1-weighted-generator: [Drive](https://drive.google.com/file/d/1tPFXAiXkm1hq0j4_wHP7U2mNr0s8ETsy/view?usp=sharing)

PVTv2b3: [pvt_v2_b3](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth)

## How to run?
Step 0:
```bash
cd segemntor
python -m venv myenv
source activate myenv
pip install -r requirements.txt
```

Step 1 (optional):
```bash
python test.py
```

Step 2:
```bash
python predict_segmentation.py
```
