# bikeFitChecker
From a video of a cyclist pedaling on a home trainer. It will place key point on joint and connect them. With this geomerty, it will extract the key angle of the body position.

## Usages

The video which will be used for analysis should be in a format .mp4 and in the directory `video`.

The output video will be saved in the directory `output`.

To make it work, please create two direcotries `video` and `output` or change the code to match your case.

Exemple directory structure : 
```text
./
├── video/
│   └── input.mp4
└── output/
    └── result.mp4
```

## Get Started

To install python dependencies :

````
pip install -r requirement.txt
````

Other dependencies :
- [ffmpeg](https://www.ffmpeg.org) (an easy way to do it is by using [chocolatey](https://chocolatey.org))

To run the app
````
streamlit run ./app.py
````
