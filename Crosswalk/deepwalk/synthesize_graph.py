from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


DATA = '../data'

def generate_2_groups_dataset(args):
    n = args.nodes
    n_red = int(n * args.Pred)

    edges = []
    for i in range(1, n_red + 1):
        for j in range(i + 1, n_red + 1):
            if np.random.rand() < args.Phom:
                edges.append((i, j))
                edges.append((j, i))
    for i in range(n_red + 1, n + 1):
        for j in range(i + 1, n + 1):
            if np.random.rand() < args.Phom:
                edges.append((i, j))
                edges.append((j, i))
    for i in range(1, n_red + 1):
        for j in range(n_red + 1, n + 1):
            if np.random.rand() < args.Phet:
                edges.append((i, j))
                edges.append((j, i))

    filename = f'{DATA}/synth2/synthetic_n' + str(n) + '_Pred' + str(args.Pred) + \
               '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)

    with open(filename + '.attr', 'w') as f:
        for i in range(1, n_red + 1):
            f.write(str(i) + ' 1' + '\n')
        for i in range(n_red + 1, n + 1):
            f.write(str(i) + ' 0' + '\n')

    with open(filename + '.links', 'w') as f:
        for e in edges:
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')

def generate_n_group_dataset(args):
    n = args.nodes

    num_classes = len(args.class_probs)
    class_sizes = [int(n * float(p)) for p in args.class_probs]
    class_delimiters = [sum(class_sizes[:i]) + 1 for i in range(1, num_classes+1)]

    class_labels = list(range(num_classes))

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if i in class_delimiters or j in class_delimiters:
                if np.random.rand() < args.Phet:
                    edges.append((i+1, j+1))
                    edges.append((j+1, i+1))
            else:
                if np.random.rand() < args.Phom:
                    edges.append((i+1, j+1))
                    edges.append((j+1, i+1))

    filename = f'{DATA}/synth{num_classes}/synthetic_n' + str(n) + '_Probs[' + '_'.join(map(str, args.class_probs)) + ']' + \
               '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)

    with open(filename + '.attr', 'w') as f:
        for i in range(n):
            f.write(str(i+1) + ' ' +
                    str(class_labels[next((x[0] for x in enumerate(class_delimiters) if x[1] > i), num_classes - 1)]) + '\n')

    with open(filename + '.links', 'w') as f:
        for e in edges:
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')


def main(args):
    if len(args.class_probs) > 2:
        generate_n_group_dataset(args)
    elif len(args.class_probs) == 2:
        generate_2_groups_dataset(args)
    else:
        print(f'Cannot generate dataset with {args.num_groups} groups.')


if __name__ == '__main__':
    parser = ArgumentParser("Synthesize Graph",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--nodes', type=int, help='Number nodes')
    parser.add_argument('--Pred', type=float, help='Probability of being red for each node')
    parser.add_argument('--Phom', type=float, help='Probability of within group connections')
    parser.add_argument('--Phet', type=float, help='Probability of cross group connections')

    # Extra
    parser.add_argument('--class-probs', nargs='+', help='Probability of being in each class')
    args = parser.parse_args()

    main(args)

    # python synthesize_graph.py --nodes 500 --Phom 0.025 --Phet 0.001 --class-probs 0.33 0.33 0.33
    # python synthesize_graph.py --nodes 1000 --Phom 0.025 --Phet 0.001 --class-probs 0.33 0.33 0.33
    # python synthesize_graph.py --nodes 2000 --Phom 0.025 --Phet 0.001 --class-probs 0.33 0.33 0.33

    # python synthesize_graph.py --nodes 500 --Phom 0.025 --Phet 0.001 --class-probs 0.2 0.2 0.2 0.2 0.2
    # python synthesize_graph.py --nodes 1000 --Phom 0.025 --Phet 0.001 --class-probs 0.2 0.2 0.2 0.2 0.2
    # python synthesize_graph.py --nodes 2000 --Phom 0.025 --Phet 0.001 --class-probs 0.2 0.2 0.2 0.2 0.2

    # python synthesize_graph.py --nodes 500 --Phom 0.025 --Phet 0.001 --class-probs 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
    # python synthesize_graph.py --nodes 2000 --Phom 0.025 --Phet 0.001 --class-probs 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
    # python synthesize_graph.py --nodes 5000 --Phom 0.025 --Phet 0.001 --class-probs 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

