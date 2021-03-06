import myparser
import commons
import logging
import datasets_ws
from os.path import join
from datetime import datetime
import torch
import multiprocessing
import network
import test

"""
SCELTA DATASET:
"pitts30k"
"st_lucia"
"""
ds = "pitts30k"

if __name__ == '__main__':

    args = myparser.parse_arguments()

    start_time = datetime.now()
    args.output_folder = join("runs", ds, args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    best_model_path = args.best_model_path
    commons.setup_logging(args.output_folder)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    # Creation of Datasets
    logging.debug(f"Loading dataset {ds} from folder {args.datasets_folder}")

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, ds, "test")
    logging.info(f"Test set: {test_ds}")

    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    # Test best model on test set
    # Path da rivedere
    best_model_state_dict = torch.load(best_model_path)["model_state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, model)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")


