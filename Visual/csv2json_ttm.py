import json
import pandas as pd
import scipy.io as sio
import os
import argparse

def csv2json(args): 
        
    records=[]
    df = pd.read_csv(args.input,header=None)
    
    lam_df = pd.read_csv(args.input, names=['video_id', 'frame_id', 'label', 'score']) 

    filter_predictions = lam_df.copy()
    for video in filter_predictions.video_id.unique():
        a=filter_predictions[(filter_predictions['video_id'] == video)].copy()
        a['score_med'] = a.loc[:, 'score'].max()
        a = a.dropna()
        aind = a.index
        if filter_predictions.loc[aind,'score'].equals(a.loc[:,'score_med']):
            continue
        else:
            filter_predictions.loc[aind,'score'] = a.loc[:,'score_med']
        
        
    results = []
    for i in range(len(filter_predictions)):
        sid = filter_predictions.iloc[i].video_id
        frame_id = int(filter_predictions.iloc[i].frame_id)
        score = float(filter_predictions.iloc[i].score)
        results.append({
            "video_id": sid,
            "frame_id": frame_id,
            "label": 1,
            "score": score
            })

    output = {
            "version": "1.0",
            "challenge": "ego4d_talking_to_me",
            "results": results
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent = 4)

  
if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser(description='Ego4d Social Benchmark')
    argparser.add_argument('--input', type=str, help='Input_pred.csv file')
    argparser.add_argument('--output', type=str, help='Output_ttm.json file')
    args = argparser.parse_args()
    csv2json(args)
