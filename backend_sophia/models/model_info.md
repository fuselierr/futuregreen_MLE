(Feb 27)
- `benchmark_model.keras`

(Mar 13)
- `resnet50_model.keras`
- `mobilenetv3_model.keras`
- `nasnetmobile_model.keras`

(Mar 16)
- `MobileNetV3Model_imgsize_(224, 224).keras`
- `NASNetMobileModel_imgsize_(224, 224).keras`
  - these are the two candidate models for the app version
  - the glass category is currently not so accurate, since there are fewer images in the dataset after multi-object filtering

(Mar 19)
- `FINAL_VER_MobileNetV3Model_datasetvers1_imgsize_(224, 224)_mindim_(50).keras`
- `FINAL_VER_MobileNetV3Model_datasetvers2_imgsize_(224, 224)_mindim_(50).keras`
  - the chosen model for the mobile app is MobileNetv3, and removes images if either their length or width is smaller than 50 pixels
- some notes:
  - there are two versions of the model
    - one was trained with dataset_vers1 (D1), and the other with dataset_vers2 (D2)
  - D1 contains many more images than D2, but they contain the entire image (i.e. their bounding box was not calculated, so the images are not cropped)
  - D2 contains the cropped images (i.e. the output after multi-object detection), but contains much fewer images than D1 due to the multi-object filtering
    - particularly, glass only has ~600 images, so the model accuracy on glass is not so good
  - so, you should test both models out to see which one performs better with user images

(Apr 5)
- `dataset_vers3_resnet50.keras`