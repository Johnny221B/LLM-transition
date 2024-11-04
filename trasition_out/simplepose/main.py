import argparse
import time
from processors.dp_pose_resnet_solver import DPProcessor

def main():
    parser = argparse.ArgumentParser(description="Run DDP training with specified config.")
    parser.add_argument('--config_path', type=str, help='Path to the configuration file.')
    args = parser.parse_args()

    ddp_processor = DPProcessor(cfg_path=args.config_path)
    ddp_processor.run()

if __name__ == '__main__':
    main()

