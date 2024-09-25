import onnx

# Load the model
model = onnx.load('code/models/model.onnx')

# Print model details
print(onnx.helper.printable_graph(model.graph))
