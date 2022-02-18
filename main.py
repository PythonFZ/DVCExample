from src.evaluate import Evaluate
from src.featurization import Featurize
from src.prepare import Prepare
from src.train import Train

if __name__ == "__main__":
    Prepare().write_graph(run=True)
    Featurize().write_graph(run=True)
    Train().write_graph(run=True)
    Evaluate().write_graph(run=True)
