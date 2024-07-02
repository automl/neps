from neps_examples import all_main_examples
from pathlib import Path

def print_examples():
    print("The following examples are available")
    for folder, examples in all_main_examples.items():
        print()
        for example in examples:
            print(f'python -m neps_examples.{folder}.{example}')

def print_specific_example(example):
    neps_examples_dir = Path(__file__).parent
    print(neps_examples_dir)
    example_file = neps_examples_dir / f"{example.replace('.', '/')}.py"
    print(example_file.read_text())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", default=None, help="Example name to print in form of 'basic_usage.hyperparameters'")
    args = parser.parse_args()

    if args.print:
        print_specific_example(args.print)
    else:
        print_examples()
