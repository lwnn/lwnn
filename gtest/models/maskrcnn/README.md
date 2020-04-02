
https://github.com/matterport/Mask_RCNN

generate the lwnn model by applying below patch to model.py and run the demo.ipynb

```sh
from keras2lwnn import *
@@ -1839,7 +1839,7 @@ class MaskRCNN():

         # Inputs
         input_image = KL.Input(
-            shape=[None, None, 3], name="input_image")
+            shape=[h, w, 3], name="input_image")
@@ -1873,7 +1873,7 @@ class MaskRCNN():
                     name="input_gt_masks", dtype=bool)
         elif mode == "inference":
             # Anchors in normalized coordinates
-            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
+            input_anchors = KL.Input(shape=[261888, 4], name="input_anchors")
@@ -2474,6 +2474,8 @@ class MaskRCNN():
             log("image_metas", image_metas)
             log("anchors", anchors)
         # Run object detection
+        keras2lwnn(self.keras_model, 'maskrcnn', use_keras2lwnn=True, rpnconfig=self.config)
```

