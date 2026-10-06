from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

# Define increment operator circuit
inc_c = QuantumCircuit(2, name="Increment operator")
inc_c.cx(0, 1)
inc_c.x(0)

# Convert the circuit to a controlled gate
c_inc_gate = inc_c.to_gate().control(1)
    
# Define decrement circuit
dec_c = QuantumCircuit(2, name="Decrement operator")
dec_c.x(0)
dec_c.cx(0, 1)

# Convert decrement circuit to a controlled gate
c_dec_gate = dec_c.to_gate().control(1)

# Define circuit for one step of DTQW
def dtqw_step(qc, position, coin):
    qc.h(coin)
    qc.x(coin)
    qc.append(c_dec_gate, [coin, position[0], position[1]])
    qc.x(coin)
    qc.append(c_inc_gate, [coin, position[0], position[1]])
    qc.x(coin)


# Initialise registers for walk qubits
nodes = QuantumRegister(2, "position")
coin = QuantumRegister(1, "coin")
classical = ClassicalRegister(2, "measurement")

# Initialise quantum circuit
qc = QuantumCircuit(coin, nodes, classical)
qc.initialize([1/np.sqrt, 1/np.sqrt(2)], coin)
qc.initialize([1, 0, 0, 0], nodes)

# Set number of steps of walk
n_steps = 10

# Create circuit for n_steps of DTQW
for _ in range(n_steps):
    dtqw_step(qc, nodes, coin)

# Measure position qubits
qc.measure(nodes, classical)

# Simulate walk
simulator = AerSimulator()
qct = transpile(qc, simulator)
result = simulator.run(qct, shots=1000).result()
counts = result.get_counts()

print("\nResults:", counts, "\n"*3)

# Visualise measurement outcomes
plot_histogram(counts)

