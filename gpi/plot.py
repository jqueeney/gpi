"""Entry point for visualizing results."""
import numpy as np

from gpi.common.plot_utils import create_plotparser, open_and_aggregate
from gpi.common.plot_utils import plot_compare

def main():
    """Parses inputs, creates and saves plot."""
    parser = create_plotparser()
    args = parser.parse_args()

    x = np.arange(0,args.timesteps+1,args.interval)

    on_data = open_and_aggregate(
        args.import_path,args.on_file,x,args.window,args.metric)
    gpi_data = open_and_aggregate(
        args.import_path,args.gpi_file,x,args.window,args.metric)

    plot_compare(on_data,gpi_data,
        x,args.se_val,args.save_path,args.save_name)


if __name__=='__main__':
    main()