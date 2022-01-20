
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use")
    # Other parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="Val/test threshold in meters")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="Train threshold in meters")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument("--augment_data", type=int, default=0, help="Augment training data with transforms:"
                                                                    "0: No transforms "
                                                                    "1: Trivial Transforms "
                                                                    "2: Random Croppings "
                                                                    "3: Random Croppings and jitter "
                                                                    "4: Autotransform with Imagenet policy "
                                                                    "5: Random realistic jitter")
    parser.add_argument('--colab', default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resize", type=int, default=None, nargs='+', help="Resize images to dimension specified")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, required=True, help="Path with datasets")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")
    parser.add_argument("--best_model_path", type=str, default="../best_models/best_model.pth", help="Path to model used for testing")

    # Network variants parameters
    parser.add_argument("--netvlad_clusters", type=int, help="use vlad layer with specified number"
                                                             " of clusters if not specified avgPooling is used instead")
    parser.add_argument("--gem_power", type=float, help="use gem layer with initial p value")
    parser.add_argument("--attention", type=int, default=0, help="use CRAM attention layer between backbone and aggregation layer")
    parser.add_argument("--loss", type=str, default="triplet ranking", choises=["triplet_ranking", "sare_ind", "sare_joint"])

    args = parser.parse_args()
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    return args

