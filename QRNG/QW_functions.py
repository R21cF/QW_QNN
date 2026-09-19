"""
Define functions that are used in generation of random numbers via a discrete-time QW 
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
from n_cycle_gates import shift_gate


# Define a funcion to implement a Hadamard coin toss
def h_coin_toss():
    n = 1
    coin = QuantumRegister(n, "coin qubits")
    results = ClassicalRegister(n, "measurement")
    qc = QuantumCircuit(coin, results)
    
    for i in range(n):
        qc.h(coin[i])
    
    qc.measure(coin, results)

    simulator = AerSimulator()
    qct = transpile(qc, simulator)
    outcome = simulator.run(qct, shots=1).result().get_counts()

    return list(outcome.keys())[0]


# This function generates a random number based on the outcome of a HQW
def qw_rng(n=5, no_of_steps=50):

    # Set starting position of walk
    initial_value = ((2 ** n) // 2) - int(h_coin_toss())

    # Initialize a register with n qubits
    qubits = QuantumRegister(n, "nodes")

    # Initialize a 1-bit coin register for the QW
    coin = QuantumRegister(1, "coin")

    classical = ClassicalRegister(n, "cbits")

    qc = QuantumCircuit(coin, qubits, classical)

    qc.initialize([1/np.sqrt(2), 1j/np.sqrt(2)], coin)

    # Initialize position qubits to the initial_value state
    qc.initialize(initial_value, qubits)

    # Create circuit for n_steps of walk
    for i in range(no_of_steps):
        qc.h(coin)
        qc.append(shift_gate(qubits, function=0), [qubit for reg in qc.qregs for qubit in reg])
        qc.append(shift_gate(qubits, function=1), [qubit for reg in qc.qregs for qubit in reg])
        qc.barrier()

    qc.measure(qubits, classical)

    # Implement classical simulation of the above QW
    simulator = AerSimulator()
    qct = transpile(qc, simulator)
    counts = simulator.run(qct, shots=1).result().get_counts()

    # Get a random number (in binary) from the quantum-walk-circuit
    b_num = list(counts.keys())[0]

    # Return binary string of random number
    return b_num
