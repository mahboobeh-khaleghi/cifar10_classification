import nets
import dataloaders
import runners
import utils

def main(args):
    # read config 
    config = utils.read_config(args.config_path)
    
    # Training the network
    eval_report = runners.eval(
        config = config
    )
    
    # Saving training report
    eval_report.to_csv(config.eval_report_path)
    

if __name__ == "__main__":
    args = utils.get_args()
    main(args)