"""Reproducible temporal validation and freight-rate prediction. Run: python train.py."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
CAT = ['pickup', 'delivery', 'equipment']
NUM = ['distance', 'weight', 'weight_missing', 'dow_sin', 'dow_cos', 'year_sin', 'year_cos', 'day_index']

def features(df, signals=False):
    x = df.copy()
    dt = pd.to_datetime(x['date'], errors='raise')
    x['weight'] = pd.to_numeric(x['weight'], errors='coerce').mask(lambda s: s <= 0)
    x['weight_missing'] = x['weight'].isna().astype(int)
    x['dow_sin'] = np.sin(2*np.pi*dt.dt.dayofweek/7)
    x['dow_cos'] = np.cos(2*np.pi*dt.dt.dayofweek/7)
    x['year_sin'] = np.sin(2*np.pi*(dt.dt.dayofyear-1)/365.25)
    x['year_cos'] = np.cos(2*np.pi*(dt.dt.dayofyear-1)/365.25)
    x['day_index'] = (dt-pd.Timestamp('2025-01-01')).dt.days
    cols = CAT+NUM+(['market_index', 'quote_signal'] if signals else [])
    return x[cols]

def fit_model(df, leaves, signals=False):
    x = features(df, signals)
    enc = ColumnTransformer([('categories', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), CAT)], remainder='passthrough')
    z = enc.fit_transform(x)
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=180, learning_rate=.07,
        max_leaf_nodes=leaves, min_samples_leaf=45, l2_regularization=3,
        categorical_features=[0,1,2], early_stopping=False, random_state=42)
    # Predict dollars/mile; weighting by miles makes absolute loss dollar-oriented.
    model.fit(z, df.posted_rate/df.distance, sample_weight=df.distance)
    return {'encoder':enc, 'model':model, 'signals':signals}

def predict(bundle, df):
    z=bundle['encoder'].transform(features(df, bundle['signals']))
    return np.maximum(bundle['model'].predict(z)*df.distance.to_numpy(), .01)

def metrics(y,p):
    return {'MAE':float(mean_absolute_error(y,p)), 'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'median_absolute_error':float(np.median(np.abs(y-p))), 'bias':float(np.mean(p-y))}

def audit(df):
    return {'rows':len(df), 'date_min':df.date.min(), 'date_max':df.date.max(),
        'missing':df.isna().sum().astype(int).to_dict(), 'invalid_weight':int((df.weight<=0).sum()),
        'duplicate_ids':int(df.load_id.duplicated().sum()),
        'duplicate_rows_without_id':int(df.drop(columns='load_id').duplicated().sum()),
        'nonpositive_distance':int((df.distance<=0).sum())}

def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    dev=pd.read_csv(ROOT/'data/train_test.csv'); final=pd.read_csv(ROOT/'data/validation.csv')
    template=pd.read_csv(ROOT/'data/validation_predictions_template.csv')
    dec=pd.read_csv(ROOT/'data/december_chart_inputs.csv')
    assert dev.load_id.is_unique and final.load_id.is_unique
    assert (dev.distance>0).all() and (dev.posted_rate>0).all() and (final.distance>0).all()
    a={'development':audit(dev),'final':audit(final),
       'unseen_pickup':sorted(set(final.pickup)-set(dev.pickup)),
       'unseen_delivery':sorted(set(final.delivery)-set(dev.delivery)),
       'rate_per_mile_quantiles':(dev.posted_rate/dev.distance).quantile([0,.01,.5,.95,.99,1]).to_dict()}
    (out/'data_audit.json').write_text(json.dumps(a,indent=2))
    rows=[]
    for month in ['2025-08','2025-09']:
        tr=dev[dev.date<month+'-01']; va=dev[dev.date.str.startswith(month)]
        for leaves in [15,31]:
            b=fit_model(tr,leaves);p=predict(b,va)
            row={'period':month,'model':f'core_{leaves}','train_rows':len(tr),'test_rows':len(va),**metrics(va.posted_rate,p)}
            rows.append(row);print(row,flush=True)
        base=float(np.median(tr.posted_rate/tr.distance))
        rows.append({'period':month,'model':'median_rate_baseline','train_rows':len(tr),'test_rows':len(va),**metrics(va.posted_rate,va.distance*base)})
    scores=pd.DataFrame(rows);winner=scores[scores.model.str.startswith('core')].groupby('model').MAE.mean().idxmin()
    leaves=int(winner.split('_')[1])
    # Auxiliary signals have no provenance and are absent from December inputs.
    # Evaluate a sensitivity model, but do not make these signals a production dependency.
    tr=dev[dev.date<'2025-09-01'];va=dev[dev.date.str.startswith('2025-09')]
    sb=fit_model(tr,leaves,True)
    rows.append({'period':'2025-09','model':'signals_sensitivity','train_rows':len(tr),'test_rows':len(va),**metrics(va.posted_rate,predict(sb,va))})
    tr=dev[dev.date<'2025-10-01'];test=dev[dev.date.str.startswith('2025-10')]
    b=fit_model(tr,leaves);p=predict(b,test)
    rows.append({'period':'2025-10','model':winner,'train_rows':len(tr),'test_rows':len(test),**metrics(test.posted_rate,p)})
    baseline=float(np.median(tr.posted_rate/tr.distance))
    rows.append({'period':'2025-10','model':'median_rate_baseline','train_rows':len(tr),'test_rows':len(test),**metrics(test.posted_rate,test.distance*baseline)})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    errors=test[['load_id','equipment','distance','posted_rate']].copy();errors['predicted_rate']=p
    errors['absolute_error']=abs(errors.posted_rate-p);errors.to_csv(out/'october_predictions.csv',index=False)
    group=errors.groupby('equipment').agg(rows=('load_id','size'),MAE=('absolute_error','mean'));group.to_csv(out/'october_by_equipment.csv')
    # Simulate unseen cities using the frozen October model and missing category codes.
    cold=test.copy();cold['pickup']='__unseen__';cold['delivery']='__unseen__'
    cold_metrics=metrics(test.posted_rate,predict(b,cold))
    b=fit_model(dev,leaves);joblib.dump(b,out/'model.joblib')
    preds=pd.Series(predict(b,final),index=final.load_id)
    assert set(template.load_id)==set(preds.index)
    template['predicted_rate']=template.load_id.map(preds).round(2)
    template.to_csv(ROOT/'validation_predictions.csv',index=False)
    dec['predicted_rate']=np.round(predict(b,dec),2)
    dec.to_csv(ROOT/'data/december_chart_inputs.csv',index=False)
    dec.to_csv(out/'december_predictions.csv',index=False)
    summary={'selected_model':winner,'selection':'Mean August and September MAE; October untouched until selection.',
        'october_cold_city_stress':cold_metrics,'final_predictions':len(template),
        'december_min':float(dec.predicted_rate.min()),'december_max':float(dec.predicted_rate.max()),
        'signals_policy':'Excluded from final model: unknown availability/provenance and absent in fixed chart inputs.',
        'features':CAT+NUM}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
