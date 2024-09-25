from maraboupy import Marabou

# Load your model (assuming it's in ONNX format)
network = Marabou.read_onnx('path/to/your/model.onnx')

# Define the input perturbation range
epsilon = 0.1  # Adjust this value as needed

# Assume input variable indices and output class index (adjust according to your model)
input_var_index = range(4096)  # Replace with actual input variable indices
output_class_red = 0  # Replace with the actual index for 'red'
output_class_not_red = 1  # Replace with the actual index for 'not red'

# Create constraints for perturbation
constraints = []
for i in input_var_index:
    constraints.append((i, '>=', original_value[i] - epsilon))
    constraints.append((i, '<=', original_value[i] + epsilon))

# Create output constraints to ensure classification remains 'red'
output_constraints = []
output_constraints.append((output_class_red, '>=', threshold))  # Ensure 'red' output remains high
output_constraints.append((output_class_not_red, '<=', threshold))  # Ensure 'not red' output is low

# Add the constraints to the network
network.addConstraints(constraints)
network.addOutputConstraints(output_constraints)

# Run the verification
result = network.verify()

if result.is_satisfiable():
    print("There exists a perturbation that changes the classification.")
else:
    print("The model is robust to perturbations within the specified range.")

