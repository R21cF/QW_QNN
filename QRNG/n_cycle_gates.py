from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Define general shift gate for a CQW
def shift_gate(qubits, function=1):
    n = len(qubits)

    if function == 0:
        qc = QuantumCircuit(qubits, name='Left-step operator')
        for i in range(n):
            if i < n - 1:
                    qc.mcx(qubits[:n - i - 1], n - i - 1, ctrl_state=0)
        qc.x(qubits[0])
        return qc.to_gate().control(1, ctrl_state=0)

    qc = QuantumCircuit(qubits, name='Right-step operator')
    for i in range(n):
        if i < n - 1:
                qc.mcx(qubits[:n - i - 1], n - i - 1)
    qc.x(qubits[0])
    return qc.to_gate().control(1)
