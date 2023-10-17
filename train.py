import runners
import utils

def main(args):
    # read config 
    config = utils.read_config(args.config_path)
    
    # Training the network
    model, train_report = runners.train(
        config = config
    )
    
    # Saving training report
    train_report.to_csv(config.train_report_path)
    

if __name__ == "__main__":
    args = utils.get_args()
    main(args)