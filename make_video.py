"""
Render the best saved LISR learner in HalfCheetah and write a video.

By default it picks the most recent checkpoint in ./lisr_logs (best_learner_gen*.pt),
which — because checkpoints are only written when a new overall best is found — is
also the best-performing learner. Rendering uses gymnasium's MuJoCo HalfCheetah-v4
(envpool, the training env, is headless and cannot render); observation and action
layouts match, so the trained policy transfers directly.

Usage:
    python make_video.py
    python make_video.py --checkpoint lisr_logs/best_learner_gen84.pt --episodes 3
    python make_video.py --out videos/run.mp4
"""

import argparse
import glob
import os
import re
import warnings

import imageio.v2 as imageio
import numpy as np
import torch

import gymnasium as gym

import config as cfg
from networks import GaussianPolicy

warnings.filterwarnings("ignore")


def _latest_checkpoint(log_dir: str) -> str:
    paths = glob.glob(os.path.join(log_dir, "best_learner_gen*.pt"))
    if not paths:
        raise FileNotFoundError(f"No best_learner_gen*.pt checkpoints found in {log_dir!r}")
    # "last one" == highest generation number == best learner so far.
    return max(paths, key=lambda p: int(re.search(r"gen(\d+)", p).group(1)))


def _load_actor(checkpoint_path: str, device: str) -> tuple[GaussianPolicy, str]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor = GaussianPolicy(cfg.OBS_DIM, cfg.ACTION_DIM, cfg.HIDDEN_SIZE).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt.get("symbolic_tree", "<unknown>")


def main():
    parser = argparse.ArgumentParser(description="Record a video of the best LISR learner.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a .pt checkpoint. Default: latest in ./lisr_logs.")
    parser.add_argument("--log-dir", default="./lisr_logs", help="Where to look for checkpoints.")
    parser.add_argument("--out", default=None,
                        help="Output video path. Default: videos/<checkpoint_name>.mp4.")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to record back-to-back.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = args.checkpoint or _latest_checkpoint(args.log_dir)
    actor, tree = _load_actor(checkpoint, device)
    print(f"Loaded checkpoint : {checkpoint}")
    print(f"Symbolic reward   : {tree}")

    out_path = args.out
    if out_path is None:
        os.makedirs("videos", exist_ok=True)
        name = os.path.splitext(os.path.basename(checkpoint))[0]
        out_path = os.path.join("videos", f"{name}.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
    fps = int(env.metadata.get("render_fps", 30))

    frames, returns = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        while True:
            obs_t = torch.from_numpy(obs.astype("float32")).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor.act(obs_t, deterministic=True).squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            frames.append(env.render())
            if terminated or truncated:
                break
        returns.append(ep_return)
        print(f"  episode {ep + 1}/{args.episodes}: return = {ep_return:.1f}")

    env.close()

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"\nMean return       : {np.mean(returns):.1f}")
    print(f"Frames            : {len(frames)} @ {fps} fps")
    print(f"Video written     : {out_path}")


if __name__ == "__main__":
    main()
