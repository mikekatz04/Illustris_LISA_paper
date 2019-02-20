#!/bin/bash
#    run with `sbatch run.sh`

#SBATCH --mail-user=lzkelley@northwestern.edu
#SBATCH -n 1
###SBATCH -p hernquist
###SBATCH -p itc_cluster
#SBATCH -p hernquist,itc_cluster
###SBATCH --mem-per-cpu=8192
#SBATCH --mem-per-cpu=40000
#SBATCH --time=20:00:00
#SBATCH -o data_out.%j
#SBATCH -e data_err.%j
#SBATCH -J data
####SBATCH --mail-type=END

#~/hostgen.sh

# python -m mbhmergers -e
# source activate ill

# python extraction_main.py --find_bhs
# python extraction_main.py --sub_part_ids
# python extraction_main.py --sublink_extraction
# python extraction_main.py --download_needed
# python extraction_main.py --density_vel_disp -r
python extraction_main.py --create_final_data -r

# python extraction_main.py -a


# rsync -a --progress fs3_extraction-files/* regal_extraction-files/
