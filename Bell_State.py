import numpy as np
import json
import time
import cirq
import os
import pickle


def main():
    # Main Function Parameters
    function = 'read'  # Can either be set to 'write' or 'read'
    project_name = 'bell_state'
    no_counts = 1000
    sim_mode = 'cirq'
    two_qubit_gate = 'Sycamore'
    processor = 'NA'

    if function == 'write':
        # Construct Bell State
        bell_circuit = cirq.Circuit()
        qubits = [cirq.LineQubit(i) for i in range(0, 2)]
        bell_circuit.append(cirq.H(qubits[0]))
        bell_circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        print(bell_circuit)

        # Perform measurements in X, Y, and Z bases
        results_dict = {}
        measurement_circuit = cirq.Circuit()
        measurement_circuit.append(cirq.measure(*qubits, key='x'))
        # First X basis
        rot_from_x = cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5)
        x_rot_circuit = cirq.Circuit()
        x_rot_circuit.append(rot_from_x(q) for q in qubits)
        result = run_circuit(bell_circuit + x_rot_circuit + measurement_circuit,
                             no_counts, sim_mode, two_qubit_gate, processor)
        results_dict.update({"All_X": result.histogram(key='x')})
        # Now Y basis
        rot_from_y = cirq.PhasedXPowGate(phase_exponent=0.0, exponent=0.5)
        y_rot_circuit = cirq.Circuit()
        y_rot_circuit.append(rot_from_y(q) for q in qubits)
        result = run_circuit(bell_circuit + y_rot_circuit + measurement_circuit,
                             no_counts, sim_mode, two_qubit_gate, processor)
        results_dict.update({"All_Y": result.histogram(key='x')})
        # Finally Z basis
        result = run_circuit(bell_circuit + measurement_circuit, no_counts,
                             sim_mode, two_qubit_gate, processor)
        results_dict.update({"All_Z": result.histogram(key='x')})

        # Now save all measurements in the form of a dictionary of counts dictionaries
        root_dir = os.getcwd()
        top_dir = project_name
        dir_path = os.path.join(root_dir, top_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(dir_path + '/bell_measurements.json', 'w') as f:
            json.dump(results_dict, f)

    elif function == 'read':
        # Read in previously saved measurements
        root_dir = os.getcwd()
        top_dir = project_name
        dir_path = os.path.join(root_dir, top_dir)
        with open(dir_path + '/bell_measurements.json', 'r') as f:
            results_dict = json.load(f)

        # Now I have my saved dictionary loaded and can do further data processing. For example...
        z_counts = results_dict["All_Z"]
        for key, value in z_counts.items():
            print("Basis state in decimal:", key, "Probability:", float(value)/float(no_counts))
    return


def run_circuit(circuit: cirq.Circuit, rep: int, sim_mode: str, two_qubit_gate: str, processor: str
                ) -> cirq.TrialResult:
    # This function is a wrapper utility for running a circuit on classical backend or quantum hardware
    if two_qubit_gate == "CZ":
        cirq.google.ConvertToXmonGates().optimize_circuit(circuit)  # Use for CH
        gate_set = cirq.google.XMON
    elif two_qubit_gate == "Sycamore":
        cirq.google.ConvertToSycamoreGates().optimize_circuit(circuit)  # Use for C\sqrt{H}
        gate_set = cirq.google.SYC_GATESET
    elif two_qubit_gate == "root_iSWAP":
        cirq.google.ConvertToSqrtIswapGates().optimize_circuit(circuit)  # Use for EAP tests
        gate_set = cirq.google.SQRT_ISWAP_GATESET
    else:
        raise ValueError("Not a valid gate set.")

    # print('\nOriginal circuit\n', circuit)
    cirq.EjectZ().optimize_circuit(circuit)
    # print('\nConverted circuit\n', circuit)

    if sim_mode == 'engine':
        # Create an Engine object.  This uses the project id of your
        # Google cloud project.
        project_id = ''  # FILL THIS STRING WITH YOUR PROJECT ID
        engine = cirq.google.Engine(project_id=project_id)

        print("Uploading program and scheduling job on Quantum Engine...\n")

        results = engine.run(
            program=circuit,
            repetitions=rep,
            processor_ids=[processor],
            gate_set=gate_set)

        # print("Scheduled. View the job at: https://console.cloud.google.com/quantum/"
        #       f"programs/{results.program_id}/jobs/{results.job_id}"
        #       f"/overview?project={project_id}")

    elif sim_mode == 'cirq' and processor == 'NA':
        results = cirq.Simulator().run(
            circuit, repetitions=rep)
    else:
        raise ValueError('No such simulation mode')
    return results


if __name__ == "__main__":
    main()
