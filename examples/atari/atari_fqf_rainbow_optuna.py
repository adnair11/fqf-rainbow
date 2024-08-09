import argparse
import datetime
import os
import pprint
import sys

import numpy as np
import torch
from atari_network import DQN
from atari_wrapper import make_atari_env

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import FQFPolicy,FQF_RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction, FullQuantileFunctionRainbow
import optuna
import logging
import pickle


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--algo-name", type=str, default="RainbowFQF-tuning")
    parser.add_argument("--seed", type=int, default=3128)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fraction-lr", type=float, default=2.5e-9)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-fractions", type=int, default=32)
    parser.add_argument("--num-cosines", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=10.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[512])
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    #rainbow elements
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    # parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)


    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument("--per", type=bool, default=False)
    return parser.parse_args()


def test_fqf(alpha,beta,noisy_std,args: argparse.Namespace = get_args()) -> None:
    env, train_envs, test_envs = make_atari_env(
        args.task,
        args.seed,
        args.training_num,
        args.test_num,
        scale=args.scale_obs,
        frame_stack=args.frames_stack,
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # args.noisy_std
    # args.alpha
    # args.beta

    # args.alpha = trial.suggest_float('gamma', 0.95, 0.99)
    args.alpha = alpha
    args.beta = beta
    args.noisy_std = noisy_std

    print("alpha and beta and noisy_std",alpha,beta,noisy_std)
    # define model
    feature_net = DQN(*args.state_shape, args.action_shape, args.device, features_only=True)
    preprocess_net_output_dim = feature_net.output_dim  # Ensure this is correctly set
    print(preprocess_net_output_dim)
    net = FullQuantileFunctionRainbow(
        preprocess_net=feature_net,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        num_cosines=args.num_cosines,
        preprocess_net_output_dim=preprocess_net_output_dim,
        device=args.device,
        noisy_std = args.noisy_std,
        is_noisy=not args.no_noisy,  # Set to True to use noisy layers
        is_dueling = not args.no_dueling,  # Set to True to use noisy layers
    ).to(args.device)
    print(net)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    fraction_net = FractionProposalNetwork(args.num_fractions, net.input_dim)
    fraction_optim = torch.optim.RMSprop(fraction_net.parameters(), lr=args.fraction_lr)
    # define policy
    policy: FQF_RainbowPolicy = FQF_RainbowPolicy(
        model=net,
        optim=optim,
        fraction_model=fraction_net,
        fraction_optim=fraction_optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        num_fractions=args.num_fractions,
        ent_coef=args.ent_coef,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        is_noisy=not args.no_noisy
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer: VectorReplayBuffer | PrioritizedVectorReplayBuffer
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
        )
    else:
        print("Using PER")
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm,
        )
        print("PER as buffer")
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # args.algo_name = "fqf_per_noisy"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 21
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
                # print("beta updated - anneal")
            else:
                beta = args.beta_final
                # print("beta updated - final")
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch() -> None:
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            # buffer = VectorReplayBuffer(
            #     args.buffer_size,
            #     buffer_num=len(test_envs),
            #     ignore_obs_next=True,
            #     save_only_last_obs=True,
            #     stack_num=args.frames_stack,
            # )
            buffer = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
                alpha=args.alpha,
                beta=args.beta,
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()

    result.pprint_asdict()
    # watch()
    return result.best_reward


# def objective(tra)

def objective(trial):
    alpha = trial.suggest_float(name='alpha',low=0.4,high=0.7)
    beta = trial.suggest_float(name='beta',low=0.4,high=1.0)
    noisy_std = trial.suggest_float(name='noisy_std',low=0.1,high=0.6)
    final = test_fqf(alpha,beta,noisy_std,get_args())
    return final

if __name__ == "__main__":

    directory = "optuna_logs"
    filename = "SpaceInvaders-sampler.pkl"
    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    # torch.manual_seed(0)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "FQF-Rainbow-SpaceInvaders-tuning" # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    

    # # to restore study uncomment this block
    # restored_sampler = pickle.load(open(file_path, "rb"))
    # study = optuna.create_study(
    #     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
    # )

    # for new study uncomment this block
    study = optuna.create_study(direction='maximize',storage=storage_name,study_name=study_name,load_if_exists=True)
    study.optimize(objective, n_trials=2)

    with open(file_path, "wb") as fout:
        pickle.dump(study.sampler, fout)  

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=5)

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    df = study.trials_dataframe()
    print(df)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    # test_fqf(get_args())
