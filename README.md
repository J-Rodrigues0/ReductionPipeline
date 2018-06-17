# CHEOPS MISSION REDUCTION PIPELINE

## Important notices

This version of the pipeline was tested on python 3.5 without errors.

The main() function of the reduction.py script applies the pipeline
steps by calling function from the reduction_inc.py file and should
thus be placed on the same folder.

The canny_edge_detection.py script houses the target area computation
functions and should also be in the same folder.

This script saves the data.txt file with the flux and background estimation
data to the output path selected by the user.

## Use Instructions

1. Run the reduction.py script
2. Select SubArray file
3. Select FlatField file
4. Select output folder
5. Choose if timing calculations should be done
6. Wait for the results.

## Final notes

If the user wishes too, it is possible to comment lines 146-152 and
uncomment lines 157-162 in order to disregard the tkinter user interface
and use the default values for the variables. This assumes the SubArray
and FlatField files are in the same directory and have the names in the
fits_path and flat_path variables, respectively.