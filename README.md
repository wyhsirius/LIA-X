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
We strongly recommend to play the model interactively. We provide both [online]() and local gradio interface.

For image animation and image editing, run

```bash
python app_animtion.py
```

For video editing, run

```bash
python app_vid_edit.py
```

## Inference
We provide configurations in `./config` for **image animation**, **video editing**, **image editing** and **linear interpolation**. Try playing with `motion_id` and `motion_value` in configuration file to obtain different results.

- Image Animation

```bash
python inference.py --mode animation --cfg 'config/animation/animation1.yaml'
```
<img src="assets/animation1.gif">

- Video Editing

```bash
python inference.py --mode vid_edit --cfg 'config/vid_edit/demo1.yaml'
python inference.py --mode vid_edit --cfg 'config/vid_edit/demo1.yaml'
```
<img src="assets/vid_edit1.gif" height="180"> <img src="assets/vid_edit2.gif" height="180">


- Image Editing

```bash
python inference.py --mode img_edit --cfg 'config/img_edit/demo1.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo2.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo3.yaml'
python inference.py --mode img_edit --cfg 'config/img_edit/demo4.yaml'
```

- Interpolation

```bash
python inference.py --mode interpolation --cfg 'config/interpolation/demo1.yaml'
```

If you would like to try other data, you could put source images in `./data/source`, driving videos in `./data/driving` and run
```bash
python inference.py --mode animation --cfg 'config/animation/default.yaml'
```
In this way, the source image will not be edited, you could try to set different values in `motion_value` to obtain better results.


