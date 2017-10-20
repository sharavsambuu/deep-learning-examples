# simple rnn represented as a for loop

def rnn(cell, input_list, initial_state):
    state = initial_state
    outputs = []
    for i, input in enumerate(input_list):
        output, state = cell(input, state)
        outputs.append(output)
    return (outputs, state)

