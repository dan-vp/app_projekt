
import pandas as pd
import pickle
import pandas as pd
from scipy.signal import resample
import numpy as np
import fastdtw.fastdtw

def preprocessing(df):
    col = []
    
    # Entferne Zeilen mit NaN-Werten
    df.dropna(axis = 0, inplace = True)
    

    # Entferne redundante Bezeichnungen aus den Spaltennamen
    for x in df.columns:
        col.append(x.split(" ")[0])

    df.columns = col
    
    # Entferne alle Schritte mit StepID = 2, da diese für das Projekt nicht relevant sind
    df.drop(df[df["CuStepNo"] == 2].index, inplace = True)
    
    # Extrahiere die Zeitangaben bzw. formatiere die Zeitstempel in das richtige Format
    df.timestamp = pd.to_datetime(df.timestamp)


    df["day"] = df.timestamp.dt.day
    df["hour"] = df.timestamp.dt.hour
    df["second"] = df.timestamp.dt.second
    
    df = step_batch_df(df) # erstellt die  beiden Dateien, die im nächsten Schritt gelesen werden
    
    # Entferne einen Zeileneintrag, bei dem die DeviationID 0 ist
    # Entferne den letzten unvollständigen Batch aus dem Datensatz
    df_steps = pd.read_pickle("SmA-Four-Tank-Info-Steps.pkl")
    df_batches = pd.read_pickle("SmA-Four-Tank-Info-Batches.pkl")
    df = df.drop(df[df.timestamp >= '2018-10-31 14:29:32'].index)
    df = df.drop(df[df.DeviationID == 0].index)
    
    return df

def step_batch_df(df):
    # determine start and end of steps
    df['dstep_p']=df['CuStepNo'].diff()
    df['dstep_n']=df['CuStepNo'].diff(-1)

    vsteps = [1,7,8,3]

    # select rows with a step change
    dfsen=df[(df['dstep_n']!=0)]
    dfsep=df[(df['dstep_p']!=0)]
    dfse=pd.concat([dfsen,dfsep])
    dfse=dfse.sort_values(by=['timestamp'])

    # create new dataframe where we store extracted information
    dfinfo_steps=pd.DataFrame(columns=['step_length','start','end','stepn'])

    # iterative approach
    pstep=-1
    c=0
    for n in range(dfse.shape[0]):
        # get row
        r=dfse.iloc[n]
        if pstep==r['CuStepNo']:
            # determine step length
            stepl=r['timestamp']-dfse.iloc[n-1]['timestamp']
            # update dataframe
            dfinfo_steps.loc[c]=(stepl,dfse.iloc[n-1]['timestamp'],r['timestamp'],r['CuStepNo'])
            c=c+1
        else:
            pstep=r['CuStepNo']
    print('Max step_length: {}'.format(dfinfo_steps['step_length'].max()))
    print('Min step_length: {}'.format(dfinfo_steps['step_length'].min()))
    print('#steps: {}'.format(dfinfo_steps.shape[0]))

    # now determine whether the batch is complete
    batchn=1
    batchi=-1
    dfinfo_steps["batchn"]=0
    dfinfo_steps["is_complete"]=False
    dfinfo_batches=pd.DataFrame(columns=['batch_length','start','end','steps','batchn','is_complete'])
    n=0
    b=0
    while True:
        if n+len(vsteps)>dfinfo_steps.shape[0]:
            # complete info at incomplete, last batch
            steps=[]
            for v in range(dfinfo_steps.shape[0]-n):
                dfinfo_steps.at[n+v,'batchn']=batchi
                dfinfo_steps.at[n+v,'is_complete']=False
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+v,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                   dfinfo_steps.at[n+v,'end'],steps,batchi,False]
            b=b+1
            break
        # check if all steps of a batch are present and in correct order
        isCorrect=True
        for v in range(len(vsteps)):
            isCorrect=dfinfo_steps.loc[n+v,'stepn']==vsteps[v]
            if not isCorrect:
                break
        if isCorrect:
            steps=[]
            for v in range(len(vsteps)):
                dfinfo_steps.at[n+v,'batchn']=batchn
                dfinfo_steps.at[n+v,'is_complete']=True
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+v,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                   dfinfo_steps.at[n+v,'end'],steps,batchn,True]
            n=n+len(vsteps)
            batchn=batchn+1
            b=b+1
        else:
            steps=[]
            for vc in range(v):
                dfinfo_steps.at[n+vc,'batchn']=batchi
                dfinfo_steps.at[n+vc,'is_complete']=False
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+vc,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                   dfinfo_steps.at[n+vc,'end'],steps,batchi,False]
            batchi=batchi-1
            n=n+vc
            b=b+1
            
    # save dfinfo_steps to file
    dfinfo_steps.to_pickle('SmA-Four-Tank-Info-Steps.pkl')
    dfinfo_batches.to_pickle('SmA-Four-Tank-Info-Batches.pkl')
    
    df_steps = pd.read_pickle("SmA-Four-Tank-Info-Steps.pkl")
    df_batches = pd.read_pickle("SmA-Four-Tank-Info-Batches.pkl")
    
    # Füge Batchnummern und Schrittlängen zum Dataframe hinzu
    a = df.merge(df_steps, how = "left", left_on = "timestamp", right_on = "start")
    a[["stepn"]] = a[["stepn"]].fillna(method = "ffill")
    a[["batchn"]] = a[["batchn"]].fillna(method = "ffill")
    a["step_length"] = a["step_length"].dt.total_seconds()
    a[["step_length"]] = a[["step_length"]].fillna(0)
    # redundante oder unnötige Spalten entfernen
    a.drop(columns = ["start", "end", "is_complete", "stepn"], inplace = True)

    useless_columns = []

    for x in a.columns[1:]:
        if a[x].describe()[2] == 0.0 and a[x].describe()[3] == a[x].describe()[-1]:
            print(f"Irrelevant column: {x} with {a[x].unique()} values")
            useless_columns.append(x)

    a.drop(columns = useless_columns, inplace = True)
    a.drop(["PIC14007_SP"], axis = 1, inplace = True) # hat auch unbedeutende Werteänderungen/ -abweichungen

    # Erstelle neue Spalte timestamp_difference, die den Zeitunterschied seit dem vorherigen Datenpunkt in Sekunden anzeigt.

    a["timestamp_difference"] = a.timestamp.diff().dt.total_seconds()

    # Setze den Wert der zuvor erstellten Spalte bei jedem neuen Batch zu Beginn auf 0

    a.loc[a.timestamp_difference > 80, "timestamp_difference"] = 0
    a.loc[0, "timestamp_difference"] = 0

    # Erstelle neue Spalte batch_duration, die pro Batch die gesamt verstrichene Zeit seit Batchbeginn anzeigt.


    a["batch_duration"] = 0

    for x in range(1, a.batchn.nunique() + 1):
        # Pro Batchnummer wird die Batchzeit berechnet (sonst bekommt man am Ende die Gesamtdauer aller Batches)
        a.loc[a.batchn == x, "batch_duration"] = a.loc[a.batchn == x, :].timestamp_difference.cumsum()

    for x in range(1, a.batchn.nunique() + 1):
        for i in a[a.batchn == x].CuStepNo.unique():
            a.loc[(a.batchn == x) & (a.CuStepNo == i), "step_length"] = a.loc[(a.batchn == x) & (a.CuStepNo == i), :].timestamp_difference.cumsum()

            
    return a


def dtw(df):
    a_dtw = df.drop(["timestamp",
                    "day", "hour", "second", "dstep_p", "dstep_n",
                   "timestamp_difference"], axis = 1)


    # dtw_df soll die neuen Zeitreihen beinhalten
    dtw_df = pd.DataFrame(columns=a_dtw.columns)

    dtw_series_list = []

    g = []




    # Gehe über jedes Feature
    for col in a_dtw.columns:
        print(f"DTW auf Spalte: {col}")

        dtw_series = []

        for x in a_dtw.CuStepNo.unique():
            # Datensatz mit nur diesem Schritt
            step_df = a_dtw[a_dtw.CuStepNo == x]

            # Schrittdauer für angepasste neue Zeitserien ermitteln
            median_length = int(a_dtw[a_dtw.CuStepNo == x].step_length.median())

            # pro Batch: resample die Zeitserie auf die vorgegebene Länge median_length (etwa Halbierung der Zeitserie)
            # danach: DTW zwischen dieser Serie und der ursprünglichen Zeitserie, um die originalen Werte
            # möglichst akkurat zu rekonstruieren

            for b in a_dtw.batchn.unique():

                time_series = step_df[step_df.batchn == b][col].to_numpy()


                # Schnellere Laufzeit für die jeweiligen 5 Features. Genaues Resampling wird hier nicht benötigt
                if col.lower() not in ["deviationid", "batchn", "custepno", "step_length", "batch_duration"]:
                    new_time_series = resample(time_series, median_length)

                else:
                    new_time_series = np.ones(median_length)

                # DTW, um alte Zeitreihe an die neue Größe anzupassen (Berechnen der Distanzmatrizen)
                distance, path = fastdtw.fastdtw(time_series, new_time_series)

                # Rekonstruieren der neuen Zeitserie mit den Werten aus der alten Serie
                for i, j in path:
                    new_time_series[j] = time_series[i]

                # Werte an dtw_series ranhängen
                dtw_series.append(new_time_series)

        # enthält alle bisher aufgenommenen Daten, könnte für einen effizienteren Algorithmus angepasst werden
        dtw_series_list.append(dtw_series)

    # Für jedes Feature und jeden Schritt werden die Werte aus dtw_series zu dtw_df hinzugefügt, and add them to the new DataFrame
    for i, col in enumerate(a_dtw.columns):
        dtw_df[col] = pd.concat([pd.Series(batch) for batch in dtw_series_list[i]])

    # Zeitunterschied zwischen den interpolierten Zeilen

    dtw_df["timestamp_difference"] = 1

    # Erstelle die Werte für step_length wie im originalen DataFrame (bevor das DTW angewendet wurde)

    for x in range(1, dtw_df.batchn.nunique() + 1):
        for i in dtw_df[dtw_df.batchn == x].CuStepNo.unique():
            dtw_df.loc[(dtw_df.batchn == x) & (dtw_df.CuStepNo == i), "step_length"] = dtw_df.loc[(dtw_df.batchn == x) & (dtw_df.CuStepNo == i), :].timestamp_difference.cumsum()

    # Spalte kann wieder entfernt werden, da nicht mehr benötigt
    dtw_df.drop("timestamp_difference", axis = 1, inplace = True)

    dtw_df.to_pickle("df_app_ml_ready.pkl")
    
    return dtw_df



def pipeline(df):
        return dtw(preprocessing(df))

