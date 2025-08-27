_base_ = ["../s3dis/openseg-pt-v1-0-msp.py"]

# recognizer settings
recognizer = dict(type="MaxProbability", method="max_logits")