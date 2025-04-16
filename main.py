import onnx

model = onnx.load("assets/migan_pipeline_v2.onnx")
for output in model.graph.output:
    print(output.name)