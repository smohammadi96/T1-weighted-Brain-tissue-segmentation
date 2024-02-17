# Brain tissue segmentation

### Download test dataset

Test dataset: [Drive]()

Labels (WM, GM, CSF, MS): [Drive]()

T1_A2B [Drive]()

**Note: T1_A2B data consists of MRI images without noise and INU which generated by our model. By downloading this data, it doesnt need to run generator first. You can skip generator step**

### Download pre-trained weights
T1-weighted segmentation models (Transformers): [Drive 1](https://drive.google.com/file/d/1sFtfAtIuaqd0XlQW225m0EDj9tYLDKTY/view?usp=sharing), [Drive 2]()

T1-weighted segmentation models (Attention-unet): [Drive]()

T1-weighted-generator: [Drive]()

PVTv2b3: [pvt_v2_b3](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth)

### How to run?
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
