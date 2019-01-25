# NFL-Big-Data-Bowl

23JAN2019

This repository contains my code and write-up for the NFL Big Data Bowl 2019 sponsored by Next Gen Stats.

Per the contest rules I am not allowed to share the data provided for the project, but the workflow has
comments throughout to explain how I went about the arrangment of data and analysis.

There was a lot of low-level event data. Instead of merging all of that data with the higher level tables
I chose to plan my work using the higher level tables. I then worked my way throug each lowwer level event data
file according to that plan.

### Note:

I orignially planned to examine rushing player data and wide receiver data. However times constraints led me to structure my research question to focus on rushing plays only. You will notice in my Python script that I initially performed subsets to retain pass plays and rushing plays. However, prior to iterating throug all of the tracking data I created a dictionary of games (keys) and play numbers (values) for ONLY non-QB non-fumble rushing plays. You could certainly extend the ideas that I considered to other player positions, but I limited the scope of my analysis to rushing plays only in order to comply with the deadline.

Compute time (single thread) to loop through all low-level event data files, subset, and apply
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
