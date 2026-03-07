import argparse
import torch
import numpy as np

from huggingface_hub import hf_hub_download

from cleanrl_utils.evals import MODELS
from cleanrl_utils.port_gameboy_worlds import get_curiosity_module


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="the random seed for reproducibility")
    parser.add_argument("--exp-name", type=str, default="dqn_atari",
        help="the name of this experiment (e.g., ppo, dqn_atari)")
    parser.add_argument("--save-name", type=str, default="",
        help="the name to use when saving the model (e.g., dqn_atari_model). If not specified, will not be used")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--hf-entity", type=str, default="cleanrl",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--hf-repository", type=str, default="",
        help="the huggingface repo (e.g., cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1)")
    parser.add_argument("--model_path", type=str, default="",
                        help="the path to the saved model (e.g., ./dqn_atari_model.pth)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--eval-episodes", type=int, default=3,
        help="the number of evaluation episodes")
    parser.add_argument("--curiosity_module", type=str, default="embedbuffer",
                    help="the type of curiosity module to use.")
    parser.add_argument("--observation_embedder", type=str, default="random_patch",
                    help="the type of observation embedder to use for the curiosity module.")
    parser.add_argument("--embedder_load_path", type=str, default=None,
                    help="path to load the observation embedder's weights from. Only applicable if the observation embedder supports loading.")
    parser.add_argument("--similarity_metric", type=str, default="cosine",
                    help="the similarity metric to use for the EmbedBuffer curiosity module.")
    parser.add_argument("--buffer_save_path", type=str, default=None,
                    help="path to save the curiosity module's buffer")
    parser.add_argument("--buffer_load_path", type=str, default=None,
                    help="path to load the curiosity module's buffer from")
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    module = get_curiosity_module(args)
    args.curiosity_module = module
    Model, make_env, evaluate = MODELS[args.exp_name]()
    if args.model_path:
        model_path = args.model_path
    else:
        if not args.hf_repository:
            args.hf_repository = (
                f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            )
        print(f"loading saved models from {args.hf_repository}...")
        model_path = hf_hub_download(
            repo_id=args.hf_repository, filename=f"{args.exp_name}.cleanrl_model"
        )
    run_name = (
        f"eval/{args.env_id}/{args.exp_name}/" + f"{args.save_name}/"
        if args.save_name
        else ""
    )
    evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=args.eval_episodes,
        run_name=run_name,
        Model=Model,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        capture_video=True,
        args=args,
    )
