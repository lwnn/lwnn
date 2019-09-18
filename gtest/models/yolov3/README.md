
from https://pjreddie.com/darknet/yolo/

apply below code to darknet/examples/detector.c to dump input raw

```
static void save_input(image sized) {
	static int id=0;
	char name[128];
	snprintf(name, sizeof(name), "tmp/input%d_.raw", id++);
	FILE* fp = fopen(name, "wb");
	fwrite(sized.data, sizeof(float), sized.h*sized.w*sized.c, fp);
	fclose(fp);
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
...
        float *X = sized.data;
        save_input(sized);
        time=what_time_is_it_now();
...
}
```

Then transpose the dumped input raw from CHW to lwnn HWC format:

```python
import numpy as np
import glob
for inp in glob.glob('tmp/input*_.raw'):
    img = np.fromfile(inp, np.float32)
    img = img.reshape(3,608,608)
    img = img.transpose((1,2,0))
    img.tofile('%s.raw'%(inp[:-5]))
```

```sh
python darknet2lwnn.py -i cfg/yolov3.cfg -w yolov3.weights -o yolov3
```