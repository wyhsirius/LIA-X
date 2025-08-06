# LIA-X: Interpretable Latent Portrait Animator
Yaohui Wang, Di Yang, Xinyuan Chen, François Brémond, Yu Qiao, Antitza Dantcheva
### [Project Page](https://wyhsirius.github.io/LIA-X-project/) | [Paper]()

<img src="teaser.gif" width="1000">

## Setup

Prepare the environment and download [model]() from huggingface to `./models`. 

```bash
git clone https://github.com/wyhsirius/LIA-X
cd LIA-X
conda env create -f environment.yml
conda activate liax
```

## Gradio Interface 
We strongly recommend to use either our [online]() or local gradio interface.

For image animation and image editing, run

```bash
python app_animtion.py
```

For video editing, run

```bash
python app_vid_edit.py
```

## Inference
You can use `inference.py` to run the demo. Use `--mode` flag to choose the setting from `image animation`, `video editing`, `image editing` and `linear interpolation` to run. The `--cfg` flag indicates the path of corresponding configuration files. Try to play with `motion_id` and `motion_value` in configuration file to obtain different results. The following are some of the examples.

**1. Image Animation**
```bash
python inference.py --mode animation --cfg 'config/animation/animation1.yaml'
```
<img src="assets/animation1.gif">

**2. Video Editing**
```bash
python inference.py --mode vid_edit --cfg 'config/vid_edit/demo1.yaml'
python inference.py --mode vid_edit --cfg 'config/vid_edit/demo2.yaml'
```
<img src="assets/vid_edit1.gif" height="180">     <img src="assets/vid_edit2.gif" height="180">


**3. Image Editing**
```bash
python inference.py --mode img_edit --cfg 'config/img_edit/demo1.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo2.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo3.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo4.yaml'
```
<img src="assets/img_edit1.png" height="180"> <img src="assets/img_edit2.png" height="180"> <img src="assets/img_edit3.png" height="180"> <img src="assets/img_edit4.png" height="180">

**4. Linear Interpolation**
```bash
python inference.py --mode interpolation --cfg 'config/interpolation/demo1.yaml'
python inference.py --mode interpolation --cfg 'config/interpolation/demo2.yaml'
python inference.py --mode interpolation --cfg 'config/interpolation/demo5.yaml'
python inference.py --mode interpolation --cfg 'config/interpolation/demo6.yaml'
```
<img src="assets/interpolation1.gif" height="180"> <img src="assets/interpolation2.gif" height="180"> <img src="assets/interpolation5.gif" height="180"> <img src="assets/interpolation6.gif" height="180">

**5. Animating your own data**

1. Put source image in `./data/source` and driving video in `./data/driving`.
2. Modify `./config/default.yaml` and set correct `source_path` and `driving_path`.
3. Play with `motion_value` to obtain the best results. By default (`motion_value=0`), the source image will not be edited.
```bash
python inference.py --mode animation --cfg 'config/animation/default.yaml'
```

