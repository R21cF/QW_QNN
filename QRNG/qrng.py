from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

q = QuantumRegister(1)
c = ClassicalRegister(1)

qc = QuantumCircuit(q, c)

qc.h(q[0])

qc.draw('mpl')