# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:50:58 2019

@author: Brad Nott

NFL Big Data Bowl

"""

import os
import re
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir('C:/Users/Brad/Desktop/NFL Data/Big-Data-Bowl-master/Big-Data-Bowl-master/Data')

df = pd.read_csv('plays.csv')

# Exclude penalty and special teams plays
df = df[(df['isPenalty'] == False) & (df['SpecialTeamsPlayType'].isnull())]

# Exclude interceptions, sack plays, and QB runs
# (include complete pass plays, incomplete pass plays, and run plays)
df = df[df['PassResult'].isin(['C', 'I', np.nan])]

# sometimes two-point conversion attempts have a NA for SpecialTeamsPlayType
# and a False for isSTPlay
# Only way to catch them is to exclude down == 0 plays
df[(df['down'] == 0) & (df['SpecialTeamsPlayType'].isnull()) & (df['isSTPlay'] == False)]['playDescription']
df = df[df['down'] != 0]

# Exclude fumble plays (not case sensitive)
df = df[~df['playDescription'].str.contains("fumble", case = False)]



# Determine receiver and ball carrier

def get_rusher(description):
    """
    Purpose: collect rusher name on a given play
             
    Input: play description from plays.csv
    Output: rusher name
    """
    try:
        return list(re.findall(r'\w+[\.]+\w+', description))[0]
    except IndexError:
        # Found a description with no player names
        return np.nan
    
def get_receiver(description):
    """
    Purpose: collect receiver name on a given play
             
    Input: play description from plays.csv
    Output: receiver name
    """
    try:
        return list(re.findall(r'\w+[\.]+\w+', description))[1]
    except IndexError:
        # Found a description with no player names
        return np.nan

# Create rushers and receivers columns
rushers = df['playDescription'].apply(get_rusher).where(df['PassResult'].isnull())

receivers = df['playDescription'].apply(get_receiver).where(df['PassResult'].isin(['C', 'I']))

# Quick look
rushers.value_counts()[0:25]
receivers.value_counts()[0:25]

# Add columns
df.insert(loc = 25, column = 'Receiver', value = receivers)
df.insert(loc = 26, column = 'Rusher', value = rushers)

# Create a yards gain vs. needed column to help determine run play success
gained_needed = (round(df['PlayResult']/df['yardsToGo'], 2)).where(df['PassResult'].isnull())

# Add column
df.insert(loc = 27, column = 'GainNeed', value = gained_needed)

# Run play success decision rules require knowing which team is home
# and which is visitor
# Collect home and visitor abbreviations in a dictionary
#  gameId: (home, visitor)
games = pd.read_csv('games.csv')

home_visitor = {}

for i in range(games.shape[0]):
    gameId = games.iloc[i]['gameId']
    home = games.iloc[i]['homeTeamAbbr']
    visitor = games.iloc[i]['visitorTeamAbbr']
    home_visitor[gameId] = (home, visitor)
    
# Combine running back play success rules into a function
def rb_success(row, games):
    """
    Purpose: calcualte if a run play was a success based on decision rules
    
    Decision rules:
        - A play counts as a "Success" if it gains 40% of yards to go on first
          down, 60% of yards to go on second down, and 100% of yards to go on
          third or fourth down.
          
        - If the team is behind by more than a touchdown in the fourth quarter,
          the benchmarks switch to 50%/65%/100%.
          
        - If the team is ahead by any amount in the fourth quarter, the
          benchmarks switch to 30%/50%/100%.
    
    inputs:
        - An entire row from plays.csv with GainNeed column
        - a dict of gameId: (home, visitor)
        * Assumes down can take a value of 1,2,3 or 4 (not 0)
        
    output:
        - True if criteria met for success on run play
        - False if criteria not met
        - nan if not a running play
        
    Note: This function will not win any beauty contests
    """
    gameId = row['gameId']
    quarter = row['quarter']
    down = row['down']
    poss_team = row['possessionTeam']
    gain_need = row['GainNeed']
    
    if pd.isna(gain_need):
        # Not a run play
        return np.nan
    
    #print()
    #print(gain_need)
    
    if poss_team == games[gameId][0]:
        diff = row['HomeScoreBeforePlay'] - row['VisitorScoreBeforePlay']
    elif poss_team == games[gameId][1]:
        diff = row['VisitorScoreBeforePlay'] - row['HomeScoreBeforePlay']
    else:
        print('Error: posession team not found')
        
    ahead_by = 0
    down_by = 0
    
    if diff < 0:
        down_by = abs(diff)
    elif diff > 0:
        ahead_by = diff
    else:
        ahead_by = 0
    
    
    if (quarter == 4) and (down_by >= 6):
        if down == 1:
            return gain_need >= 0.5
        elif down == 2:
            return gain_need >= 0.65
        else:
            return gain_need >= 1.0

    elif (quarter == 4) and (ahead_by > 0):
        if down == 1:
            return gain_need >= 0.3
        elif down == 2:
            return gain_need >= 0.5
        else:
            return gain_need >= 1.0

        
    elif down == 1:
        return gain_need >= 0.4
    elif down == 2:
        return gain_need >= 0.6
    else:
        return gain_need >= 1.0
    
    #return np.nan

df = df.reset_index(drop = True)

run_success = pd.Series(np.nan, index = range(df.shape[0]))

for i in range(df.shape[0]):
    run_success[i] = rb_success(df.iloc[i], home_visitor)

# Add column
df.insert(loc = 28, column = 'RunSuccess', value = run_success)

# check to see if any touchdown runs are labeled as failures
df[df['playDescription'].str.contains('touchdown', case = False)][['quarter', 'down', 'GainNeed', 'RunSuccess', 'playDescription']]

success_rate = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()/x.shape[0], 2))

rushes = df.groupby('Rusher')['RunSuccess'].count()

successful = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()))

rb_success_rate = pd.DataFrame({'Rushes': rushes, 'Successful': successful, 'SuccessRate':success_rate})
rb_success_rate.sort_values('SuccessRate', inplace = True, ascending = False)
rb_success_rate.to_csv('success_rate.csv')


def ci_lower_bounds(successful_runs, n, confidence):
    """
    successful_runs: total runs multiplied by success rate
    n: total runs
    confidence: desired confidence level (e.g., 0.95)
    """
    if n == 0:
        return 0

    z = stats.norm.ppf(1-(1-0.95)/2)
    
    p_hat = 1.0*successful_runs/n
    
    return (p_hat + z*z/(2*n) - z*sqrt((p_hat*(1 - p_hat) + z*z/(4*n))/n))/(1 + z*z/n)



# sorted by confidence in success rate

rb_success_rate['Conf_Adj_Rank'] = rb_success_rate.apply(lambda x: round(ci_lower_bounds(x['Successful'], x['Rushes'], 0.95), 4), axis = 1)

rb_success_rate.sort_values('Conf_Adj_Rank', inplace = True, ascending = False)
rb_success_rate.to_csv('success_rate_adj.csv')



# Gather run plays
#Group subset of run plays by game ID, collect associated plays
df_rush = df[df['RunSuccess'].isin([True, False])]
rushes_by_gameId = df_rush[['playId', 'gameId', 'RunSuccess', 'Rusher', 'PlayResult']].groupby('gameId')


# Build a rush plays dictionary to associate games with a list of plays
# format:
#       key: gameId
#       value: [(plays.csv_index, playId, RunSuccess, Rusher, ...]
rush_plays = {}

for name, group in rushes_by_gameId:

    rush_plays[name] = list(zip(list(group.index),
              list(group['playId']),
              list(group['RunSuccess']),
              list(group['PlayResult']),
              list(group['Rusher'])))
    
def tot_dist_time(play):
    """
    Total distance player traveled on the play
    Total time to go that distance
    
    Use for computing efficiency and average speed columns
    
    Input: pandas groupby object; grouped by playId
    Output: distance, time
        - yards gained will come from the PlayResult column in the game_rush
          dataframe
    """
    
    stop_index = play['event'].last_valid_index()
    stop_event = play['event'][stop_index]
    
    if stop_event == 'qb_kneel':
        return np.nan
    
    dist = 0
    time = 0
    moving = False

    
    for i in range(play.shape[0]):
        #try:
            #print(play['event'][i])
        #except KeyError:
            #print("failed on )
        
        # We do not know which man 'man_in_motion' refers to
        # Use 'ball_snap' as start of distance
        if play['event'].iloc[i] == 'ball_snap' or play['event'].iloc[i] == 'snap_direct':
            moving = True
            
        # final non NA value marks the end of the run
        # it will be 'tackle', 'out_of_bounds', 'touchdown', etc...
        elif play['event'].iloc[i] == stop_event:
            return round(dist, 2), round(time/10, 2)
        
        if moving:
            dist += play['dis'].iloc[i]
            time += 1
        #elif not moving and dist > 0:
            # Run has finished
            
            
def yards_after_contact(play):
    
    
    stop_index = play['event'].last_valid_index()
    stop_event = play['event'][stop_index]
    
    if stop_event == 'qb_kneel':
        return np.nan
    
    dist = 0
    moving = False

    for i in range(play.shape[0]):
        
        # We do not know which man 'man_in_motion' refers to
        # Use 'ball_snap' as start of distance
        if play['event'].iloc[i] == 'first_contact':
            moving = True
            
        # final non NA value marks the end of the run
        # it will be 'tackle', 'out_of_bounds', 'touchdown', etc...
        elif play['event'].iloc[i] == stop_event:
            return round(dist, 2)
        
        if moving:
            dist += play['dis'].iloc[i]
            
def max_speed(play):
    
    stop_index = play['event'].last_valid_index()
    stop_event = play['event'][stop_index]
    
    if stop_event == 'qb_kneel':
        return np.nan
    
    speeds = []
    moving = False

    
    for i in range(play.shape[0]):
       
        if play['event'].iloc[i] == 'ball_snap' or play['event'].iloc[i] == 'snap_direct':
            moving = True
            
        # final non NA value marks the end of the run
        # it will be 'tackle', 'out_of_bounds', 'touchdown', etc...
        elif play['event'].iloc[i] == stop_event:
            
            try:
                return max(speeds)
            except:
                # Missing ball snap tag in tracking data
                return np.nan
                
        if moving:
            speeds.append(play['s'].iloc[i])


def contact_speed(play):
    """
    Output: player speed at time of contact
    """

    for i in range(play.shape[0]):
       
        if play['event'].iloc[i] == 'first_contact' or play['event'].iloc[i] == 'tackle':
            return play['s'].iloc[i]
        
    return 'no contact'
            
            
            
            

    



# Iteration procedure

num_plays = df.shape[0]

# Get all file names from directory
all_files = os.listdir()

# Collect tracking data filenames
game_filenames = []

for file in all_files:
    if file.startswith('tracking'):
        game_filenames.append(file)
        
# Create empty new columns:
# - tot_dist
tot_dist_col = pd.Series(np.nan, index = range(num_plays))
tot_time_col = pd.Series(np.nan, index = range(num_plays))
yds_aft_ct_col = pd.Series(np.nan, index = range(num_plays))
max_spd_col = pd.Series(np.nan, index = range(num_plays))
ct_spd_col = pd.Series(np.nan, index = range(num_plays))



# Loop over all tracking files (steps below)
# 1. load next file
# 2. subset
# 3. group by playId
# 4. apply appropriate function
# 5. fill new empty column with values at appropriate index







total_rows = 0
files_read = 0
missing = 0

for filename in game_filenames:
    
    files_read += 1
    
    print("{} files to go".format(len(game_filenames) - files_read))
    
    game = pd.read_csv(filename)
    game['split_name'] = game['displayName'].apply(lambda x: x.split()[0][0].lower() + ' ' + x.split()[1].lower() if len(x.split()) > 1 else x)  

    game_id = int(re.findall(r'\d+', filename)[0])
    print('current gameId: {}'.format(game_id))

    # Subset to keep desired rows associated with this game
    look_up = pd.DataFrame(rush_plays[game_id], columns = ['orig.index', 'playId', 'RunSuccess', 'PlayResult', 'Rusher'])
    look_up['split_name'] = look_up['Rusher'].apply(lambda x: x.split('.')[0][0].lower() + ' ' + x.split('.')[1].lower())
    
    game_rush = pd.merge(look_up, game)
    
    # group by playId and check that only one unique name exists per group
    # detect of two players with similar names were on field during same play
    
    
    rows = game_rush.shape[0]
    total_rows += rows

    grouped_by_playId = game_rush.groupby('playId')
    
    
    
    
    
    #if not tag_check(grouped_by_playId):
        #print('Unbalanced tags in game {}'.format(game_id))
    
    # Collect playId (index) and time to throw for all plays in one game
    d_t = {}
    yds_aft_ct = {}
    mx_spd = {}
    ct_spd = {}
    for name, group in grouped_by_playId:
        d_t[group['playId'].iloc[0]] = tot_dist_time(group)
        yds_aft_ct[group['playId'].iloc[0]] = yards_after_contact(group)
        mx_spd[group['playId'].iloc[0]] = max_speed(group)
        ct_spd[group['playId'].iloc[0]] = contact_speed(group)
    

    for play in rush_plays[game_id]:

        index = play[0]
        play_num = play[1]
        
        #print("index: ", index)
        #print("play num: ", play_num)
        #print("d_t: ", d_t[play_num])
        #print(d_t[play_num][1])
       
    
        try:
            tot_dist_col[index] = d_t[play_num][0]
            tot_time_col[index] = d_t[play_num][1]
            yds_aft_ct_col[index] = yds_aft_ct[play_num]
            max_spd_col[index] = mx_spd[play_num]
            ct_spd_col[index] = ct_spd[play_num]
        except TypeError:
            # QB kneel play; nan values for rush_player
            tot_dist_col[index] = np.nan
            tot_time_col[index] = np.nan
            yds_aft_ct_col[index] = np.nan
            max_spd_col[index] = np.nan
            ct_spd_col[index] = np.nan
        except KeyError:
            missing += 1
            print('Play num {} has been removed from tracking data'.format(play_num))
            continue
        

# Add column
df.insert(loc = 29, column = 'TotalDist', value = tot_dist_col)
df.insert(loc = 30, column = 'TotalTime', value = tot_time_col)
avg_speed = round(df['TotalDist']/df['TotalTime'], 2)

efficiency = round(df['TotalDist']/df['PlayResult'], 2)

df.insert(loc = 31, column = 'AvgSpeed', value = avg_speed)
df.insert(loc = 32, column = 'MaxSpeed', value = max_spd_col)
df.insert(loc = 33, column = 'ContactSpeed', value = ct_spd_col)
df.insert(loc = 34, column = 'YardsAfterContact', value = yds_aft_ct_col)
df.insert(loc = 35, column = 'EFF', value = efficiency)

df = df.replace(np.inf, np.nan)
df = df.replace('no contact', np.nan)

# Save updated version of plays.csv with new columns of calculated values
df.to_csv('plays_v2.csv')


# Recalculate Running Back rankings using new variables
# See how the other variables stack up when sorted by adjusted success rate

success_rate_v2 = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()/x.shape[0], 2))

rushes = df.groupby('Rusher')['RunSuccess'].count()
successful = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()))
avg_eff = round(df.groupby('Rusher')['EFF'].mean(), 2)
avg_max_spd = round(df.groupby('Rusher')['MaxSpeed'].mean(), 2)
avg_ct_spd = round(df.groupby('Rusher')['ContactSpeed'].mean(), 2)
avg_yds_aft_ct = round(df.groupby('Rusher')['YardsAfterContact'].mean(), 2)

rb_success_rate_v2 = pd.DataFrame({'Rushes': rushes,
                                   'Successful': successful,
                                   'AvgEFF': avg_eff, 
                                   'AvgMaxSpeed': avg_max_spd,
                                   'AvgCtSpeed': avg_ct_spd,
                                   'AvgYardsAfterContact': avg_yds_aft_ct,
                                   'SuccessRate':success_rate})

# sorted by confidence in success rate

rb_success_rate_v2['Conf_Adj_Rank'] = rb_success_rate.apply(lambda x: round(ci_lower_bounds(x['Successful'], x['Rushes'], 0.95), 4), axis = 1)

rb_success_rate_v2.sort_values('Conf_Adj_Rank', inplace = True, ascending = False)
rb_success_rate_v2.to_csv('success_rate_adj_v2.csv')


import seaborn as sns
import matplotlib.pyplot as plt

# Reset default params
sns.set()
sns.set(rc={'figure.figsize':(11.7,8.27)})

# Set context to `"paper"`
sns.set_context("poster")

sns.swarmplot(x="quarter", y="ContactSpeed", data=df[~df['ContactSpeed'].isin(['no contact'])])
plt.show()

sns.set_context("poster")
ax = sns.violinplot(x = "YardsAfterContact", data=df)
ax.set_title("Yards After Contact (rushing plays)")
#grid.set_xticklabels(rotation=30)
#ax = sns.violinplot(x = "YardsAfterContact", data=df)
# Set the `xlim`
#ax.set(xlim=(0, 100))
ax.figure.savefig("yds_ct_dist.png")
ax = sns.regplot(x="ContactSpeed", y="YardsAfterContact", data=df[~df['ContactSpeed'].isin(['no contact'])], fit_reg = False)
ax.set(xlabel='Speed at Moment of Contact (yds/s)', ylabel='Yards Gained After Contact')
ax.figure.savefig("spd_yds_gain.png")
plt.show()

ax.set_context("poster")
ax = sns.regplot(x="MaxSpeed", y="YardsAfterContact", data=df[~df['MaxSpeed'].isin(['no contact'])], fit_reg = False)
ax.set(xlabel='Max Speed (yds/s)', ylabel='Yards Gained After Contact')
ax.figure.savefig("max_spd_yds_gain2.png")
plt.show()

ax.set_context("poster")
ax = sns.regplot(x="AvgSpeed", y="RunSuccess", data=df[df['RunSuccess'] == True].isin(['no contact'])], fit_reg = False)
ax.set(xlabel='Max Speed (yds/s)', ylabel='Yards Gained After Contact')
ax.figure.savefig("max_spd_yds_gain2.png")
plt.show()


from scipy.stats.stats import pearsonr

x = df['ContactSpeed']
y = df['YardsAfterContact']

def r2(x,y):
    nas = nas = np.logical_or(np.isnan(x), np.isnan(y))
    return pearsonr(x[~nas], y[~nas])[0]
sns.set_context("poster")
h = sns.jointplot(x, y, kind="reg", stat_func=r2)
h.set_axis_labels(xlabel='Contact Speed (yds/s)', ylabel='Yards Gained After Contact')
plt.show()
h.savefig("spd_yds_gain_corr.png")


x = df['EFF']
y = df['YardsAfterContact']

def r2(x,y):
    nas = nas = np.logical_or(np.isnan(x), np.isnan(y))
    return pearsonr(x[~nas], y[~nas])[0]
sns.set_context("poster")
h = sns.jointplot(x, y, kind="reg", stat_func=r2)
h.set_axis_labels(xlabel='Rushing Efficiency', ylabel='Yards Gained After Contact')
plt.show()
h.savefig("spd_eff_corr.png")

sns.set_context("poster")
ax = sns.violinplot(x="RunSuccess", y='ContactSpeed', data=df, palette="muted")
ax.figure.savefig("run_spd.png")

sns.set_context("poster")
ax = sns.violinplot(x="RunSuccess", y='YardsAfterContact', data=df, palette="muted")
ax.figure.savefig("run_suc_yds.png")

sns.set_context("poster")
ax = sns.violinplot(x="RunSuccess", y='EFF', data=df, palette="muted")
ax.figure.savefig("run_spd.png")

ax = sns.regplot(x="AvgSpeed", y="EFF", data=df, fit_reg = False)
ax.set(xlabel='Average Speed Throughout A Play (yds/s)', ylabel='Efficiency')

df['MaxSpeed'].describe()
df[df['RunSuccess'] == False]['ContactSpeed'].describe()

df[df['RunSuccess'] == False]['YardsAfterContact'].describe()


success_rate = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()/x.shape[0], 2))

max_speed = df.groupby('Rusher')['MaxSpeed'].mean()
avg_spd = df.groupby('Rusher')['AvgSpeed'].mean()

successful = df.groupby('Rusher')['RunSuccess'].apply(lambda x: round((x*1).sum()))

rb_success_rate = pd.DataFrame({'Rushes': rushes, 'Successful': successful, 'MaxSpeed': max_speed, 'AvgSpeed': avg_spd, 'SuccessRate':success_rate})
rb_success_rate.sort_values('SuccessRate', inplace = True, ascending = False)


eff = df.groupby('Rusher')['EFF'].mean()
