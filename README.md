Basic program for training a pedestrian detector.

Input data directories:
- "resources/train/": images for training new models
- "resources/test_public/": images for testing trained models

1) Specify the CURRENT_STAGE value in Main.java to create a directory for a new training iteration in "resources/out/your_stage_name/"
   Your new CURRENT_STAGE value should not match existing directory names in "resources/out/"
2) Run Main.java to run the full training cycle
   "model.yml" will be updated upon completion
4) Check the results in "resources/out/your_stage_name/img"
