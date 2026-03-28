# Tennis-Shot-Pose-Estimation-using-LSTM
this project is about blablabla
## Dataset Construction
### Download data
to get the dataset, you can download any videos on youtube using a converter.
## Tennis shot Annotation
to make your annotation, you can use the `extract_nomirror.py` file
 ```bash
 python extract_nomirror.py --video yourtrainingvideo.mp4
```
example of annotation:

![Image](https://github.com/user-attachments/assets/e3436f46-e549-4023-a9b8-25f81857fb20)

press on your keyboard to annotate the following class:

1. F for Forehand
2. B for Backhand
3. S for Serve
4. D for Undo
5. SPACE for pause
6. [ / ] to move forward/backward
7. Q for exit

This will output a CSV filen named `tennis_dataset.csv` containing something like this:

```bash

```
