# NFL-Big-Data-Bowl

23JAN2019

### Purpose

This repository contains my code and write-up for the NFL Big Data Bowl 2019 sponsored by Next Gen Stats.

Per the contest rules I am not allowed to share the data provided for the project, but the workflow has
comments throughout to explain how I went about the arrangment of data and analysis.

There was a lot of low-level event data. Instead of merging all of that data with the higher level tables
I chose to plan my work using the higher level tables. I then worked my way through each lower level event data
file according to that plan.

### Update 24JUN19

Added my explanation for how I interpreted the 'first_contact' tag, and why I made the assumption that it is a reference to the ball carrier. Curiously enough with this data set the play description field/column in the plays.csv file indicated who the ball carrier on each play was but it used an abbreviation. The sensor data identifies players with their full names. With a little cross referencing of the player profile data and some disambiguation work I was able to identify and extract the data for ball carriers on each rushing play.

### Update 27JAN19

Recreated report PDF (nfl_big_data_bowl_Brad_Nott_v3.pdf) using R markdown to correct the low resolution plots. Markdown file incorporates Python and R code. CSV data loaded in markdown file were derived from the raw data using master.py

R markdown file is now the primary source for plotting code (not master.py)

### Update 25JAN19

Added code at line 522 in master.py that recreates the previously constructed running back ranking to include the new variables generated by iterating over the tracking file data. This way you can not only see how the running backs rank according to the previously calculated adjusted success rate metric, but you can also see their individual performance in other areas that might have contributed to their success rate.

For some reason I was not able to eliminate all QB rushes from consideration so the output of this code includes some quarterbacks with rushes and corresponding calcualted data, and some with rushes and no calculated data.

### Note 1:

I orignially planned to examine rushing player data and wide receiver data. However times constraints led me to structure my research question to focus on rushing plays only. You will notice in my Python script that I initially performed subsets to retain pass plays and rushing plays. However, prior to iterating throug all of the tracking data I created a dictionary of games (keys) and play numbers (values) for ONLY non-QB non-fumble rushing plays. You could certainly extend the ideas that I considered to other player positions, but I limited the scope of my analysis to rushing plays only in order to comply with the deadline.

### Note 2:

In the Python script functions and code appear as the analysis requires instead of placing all functions at the beginning or in a separate file to be imported. If anything is confusing just ask. If you have access to the same data and want to use my code, run it a few lines at a time like you might a Jupyter Notebook.


Compute time (single thread) to loop through all low-level event data files(loop at code line 426), subset, and apply
necessary functions was ~3 min.

Total Data size:
- ~2 GB

Data Structure:
- collection of 94 tables extracted from a relational database

Approach overview:
- Subset primary dataframe to collect desired criteria
- Iterate over 91 data files cotaining low-level player tracking (event) data
- Use appropriate criteria to create lookup table for current data file
- Subset and group each data file and apply appropriate functions

Computer:
- Desktop with Intel core i7-8700k and 32 GB RAM
