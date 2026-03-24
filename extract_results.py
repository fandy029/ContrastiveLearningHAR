
import nbformat
import json

# Read the notebook
with open('SimCLR_MotionSense_result.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Extract outputs from code cells
results = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and hasattr(cell, 'outputs') and cell.outputs:
        for output in cell.outputs:
            if output.output_type == 'stream' and hasattr(output, 'text'):
                for line in output.text.split('\n'):
                    if any(keyword in line for keyword in ['Epoch', 'accuracy', 'loss', 'Testing', 'Validation', 'Training', 'shape']):
                        results.append(line.strip())
            elif output.output_type == 'execute_result' and hasattr(output, 'data'):
                if 'text/plain' in output.data:
                    text = output.data['text/plain']
                    if any(keyword in text for keyword in ['accuracy', 'loss', 'Testing', 'Validation', 'shape']):
                        results.append(text.strip())

print("Extracted results:")
print("=" * 60)
for line in results:
    print(line)
print("=" * 60)
print(f"\nTotal cells executed: {len([c for c in nb.cells if c.cell_type == 'code' and hasattr(c, 'execution_count') and c.execution_count is not None])}")

# Check the final cell output
last_cell = nb.cells[-1]
print(f"\nLast cell ({len(nb.cells)}) type: {last_cell.cell_type}")
