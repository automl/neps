# run_pipeline.py ----------------------------------------------------------
import argparse, time, json, torch, torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import neps
from neps.plot.tensorboard_eval import tblogger

def hpo_wrapper(neps_args: argparse.Namespace):
    def run(conf: argparse.Namespace = None):
        """A simple pipeline that trains a model on MNIST and evaluates it."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        start_eval = time.time()
        # toy 1‑layer model
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = (torch.optim.SGD if neps_args.optimizer == "sgd" else torch.optim.Adam)(
            model.parameters(), lr=neps_args.learning_rate
        )

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/tmp",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=conf.train_batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("/tmp", train=False, transform=transforms.ToTensor()),
            batch_size=conf.test_batch_size,
        )

        writer = tblogger.ConfigWriter(write_summary_incumbent=True, 
                                        root_directory=neps_args.root_directory,
                                        pipeline_directory=neps_args.pipeline_directory,
                                        previous_pipeline_directory=neps_args.previous_pipeline_directory)

        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            writer.add_scalar("train/loss", loss.item(), i)
            if i % 10 == 0:
                print(f"Pipeline ID {neps_args.pipeline_id} - Iteration {i}: Loss = {loss.item()}")
            loss.backward()
            opt.step()

        time.sleep(2)  # simulate some time for training

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        return dict(objective_to_minimize=1-acc, cost=time.time() - start_eval)

    return run


def main():
    try:
        hp_parser = argparse.ArgumentParser()
        #load hyperparameters
        hp_parser.add_argument("--learning-rate", type=float, required=True)
        hp_parser.add_argument("--optimizer", choices=["sgd", "adam"], required=True)
        
        #load neps pipeline info
        hp_parser.add_argument("--pipeline-id", type=int, required=True)
        hp_parser.add_argument("--root-directory", type=str, required=True)
        hp_parser.add_argument("--pipeline-directory", type=str, required=True)
        hp_parser.add_argument("--previous-pipeline-directory", type=str, required=True)
        
        hp_args, remaining = hp_parser.parse_known_args()

        # potentially other args needed for your pipeline and you have defined in the logic
        run_parser = argparse.ArgumentParser()
        run_parser.add_argument("--train_batch_size", type=int, default=128)
        run_parser.add_argument("--test_batch_size", type=int, default=128)

        args = run_parser.parse_args(remaining)

        user_result = hpo_wrapper(neps_args=hp_args)(conf=args)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        user_result = dict(objective_to_minimize=None, cost=None, exception= e)
    neps.save_pipeline_results(
        user_result=user_result,
        pipeline_id=hp_args.pipeline_id,
        root_directory=Path(hp_args.root_directory),
    )  

if __name__ == "__main__":
    main()
