from walk import *

def simulate_walk(walk_simulation: Walk) -> SimulationTrajectory:
    trajectory_walk = walk_simulation.execute()
    # Print summary
    print(f"\nTrajectory summary:")
    print(f"  Duration: {trajectory_walk.timestamps[-1]:.2f}s")
    print(f"  Steps: {len(trajectory_walk)}")
    print(f"  Base position range: {trajectory_walk.base_positions.min(axis=0)} to {trajectory_walk.base_positions.max(axis=0)}")
    return trajectory_walk

def visualize_trajectory(walk_sim: Walk, trajectory: SimulationTrajectory) -> None:
    # Visualize the trajectory using Rerun
    print("\nVisualizing trajectory in Rerun...")
    trajectory.to_rerun(walk_sim._model, spawn=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="H1 Humanoid Robot Walk Simulation with MuJoCo")
    parser.add_argument("--policy_path", type=str, default=POLICY_PATH, help="Path to the trained PyTorch policy (.pt file)")
    parser.add_argument(
        "--n_substeps",
        type=int,
        default=4,
        help="Number of MuJoCo substeps to run between each policy inference (default: 4)"
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.5,
        help="Scaling factor for policy actions (default: 0.5)"
    )
    parser.add_argument(
        "--vel_scale_x",
        type=float,
        default=0.5,
        help="Scaling factor for forward velocity command (default: 0.5)"
    )
    parser.add_argument(
        "--vel_scale_y",
        type=float,
        default=0.5,
        help="Scaling factor for lateral velocity command (default: 0.5)"
    )
    parser.add_argument(
        "--vel_scale_rot",
        type=float,
        default=1.0,
        help="Scaling factor for rotational velocity command (default: 1.0)"
    )
    parser.add_argument(
        "--time_walk",
        type=float,
        default=10.0,
        help="Duration of the walking simulation (default: 10.0s)"
    )
    parser.add_argument(
        "--trajectory_save_path",
        type=str,
        default="h1_walk_trajectory.npz",
        help="Path to save the recorded trajectory (default: h1_walk_trajectory.npz)"
    )

    args = parser.parse_args()
    config = SimulationConfig(
        policy_path=args.policy_path,
        n_substeps=args.n_substeps,
        action_scale=args.action_scale,
        vel_scale_x=args.vel_scale_x,
        vel_scale_y=args.vel_scale_y,
        vel_scale_rot=args.vel_scale_rot,
        time_walk=args.time_walk,
        trajectory_save_path=args.trajectory_save_path,
    )

    walk_simulation = Walk(config)
    trajectory_walk = simulate_walk(walk_simulation)

    visualize_trajectory(walk_simulation, trajectory_walk)
        

    
    


if __name__ == "__main__":
    main()
