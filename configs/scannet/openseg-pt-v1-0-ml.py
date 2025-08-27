_base_ = ["../scannet/openseg-pt-v1-0-msp.py"]

# recognizer settings
recognizer = dict(type="MaxProbability", method="max_logits")