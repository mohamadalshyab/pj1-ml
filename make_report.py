"""Build the submission report from actual saved metrics and scorer output."""
from pathlib import Path
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

P=Path(__file__).resolve().parent
M=pd.read_csv(P/'results/metrics.csv');A=json.loads((P/'results/data_audit.json').read_text())
S=json.loads((P/'results/summary.json').read_text())
styles=getSampleStyleSheet()
fontroot=Path('/usr/share/fonts/truetype/dejavu')
if (fontroot/'DejaVuSans.ttf').exists():
    pdfmetrics.registerFont(TTFont('DejaVu',str(fontroot/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold',str(fontroot/'DejaVuSans-Bold.ttf')))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=15,spaceAfter=10,textColor=colors.HexColor('#253747')))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6))
styles['Title'].textColor=colors.HexColor('#064A56');styles['Heading1'].textColor=colors.HexColor('#064A56')
if (fontroot/'DejaVuSans.ttf').exists():
    for name in ['BodyCustom','SmallCustom']:styles[name].fontName='DejaVu'
    for name in ['Title','Heading1','Heading2']:styles[name].fontName='DejaVuBold'
story=[]
def p(text,style='BodyCustom'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'Heading1')
def table(rows,widths):
    t=Table([[Paragraph(str(c),styles['SmallCustom']) for c in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E5F0F1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#064A56')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DDE3E8'))]))
    story.append(t);story.append(Spacer(1,12))
def page():story.append(PageBreak())

p('Freight Rate Prediction','Title')
p('Machine Learning Engineer assessment | Mohammed Nail Al-Sheyab','Heading2')
p('Technical report | 5 October 2026')
h('1. Objective and outcome')
p('Estimate total shipment prices for all 12,000 unlabeled loads and produce the required fixed December scenario chart. The solution uses one core-feature model for both prediction tasks and a chronological evaluation that respects the future-dated final inputs.')
table([['October holdout measure','Selected model','Baseline'],['MAE (USD)','114.35','253.85'],['RMSE (USD)','649.73','695.05'],['Median absolute error (USD)','36.80','132.82'],['Mean bias: predicted minus actual (USD)','-46.49','+33.75']],[250,110,110])
p('The selected model reduces October MAE by 54.95% relative to the median-rate-per-mile baseline. This is an internal holdout comparison. The final November/December labels are unavailable; no hidden evaluation score is claimed.')
h('Data and deliverables')
table([['Input / output','Scope'],['Labeled development data','48,000 loads; 1 January-31 October 2025'],['Final input data','12,000 loads; 1 November-31 December 2025'],['Final prediction file','Exactly load_id,predicted_rate; 12,000 positive predictions'],['December scenario','31 daily predictions; Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb']],[185,285])
p('The provided scorer successfully validated both output files and generated scorer_results/candidate_december.png. Its checks establish output validity, not predictive accuracy.')
page()
h('2. Data quality and feature policy')
table([['Check','Development','Final inputs','Treatment'],['Missing weight','300','165','Preserve as NaN'],['Nonpositive weight','292','145','Convert to NaN'],['Missing market_index','374','249','Not required by final model'],['Duplicate load IDs','0','0','Validate unique alignment'],['Duplicate rows excluding ID','0','0','No deduplication needed'],['Nonpositive distance','0','0','Required positive for rate target']],[154,75,75,166])
p('A missing-weight indicator is added. HistGradientBoosting handles missing numeric values natively. Categorical encoders are fitted only on each training window; unseen categories map to NaN. The final inputs contain eight new cities: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk and San Diego.')
p('Development prices per mile range from 0.33 to 14.13, with a median of 2.15 and a 99th percentile of 3.18. These are unusual observations, not proven errors. No target rows are removed or winsorized. Absolute loss reduces sensitivity during fitting; evaluation retains every October row.')
h('Features available for both prediction tasks')
p('Pickup city, delivery city, equipment, distance, cleaned weight, a missing-weight flag, weekday sine/cosine, day-of-year sine/cosine and elapsed day index. load_id is excluded from training and retained only for output alignment. Coordinate columns are excluded so the same input contract works for the fixed December scenario.')
p('market_index and quote_signal have undocumented provenance and are absent from December inputs. A September sensitivity experiment with both signals produced MAE 114.37, versus 108.97 for the core model. This single experiment does not show that these fields are universally unhelpful or prove leakage; it supports the simpler final feature policy within this assessment.')
h('Geographic fallback')
p('Unseen cities are routed through the estimator\'s missing-category behavior. As a stress test, both city fields were masked in every October row, using the frozen October model. MAE increased to 130.76. This is only a simulated robustness check and is not a measured score for genuinely new cities.')
page()
h('3. Training and temporal validation')
table([['Purpose','Training period / rows','Evaluation period / rows'],['Model selection, fold 1','January-July / 33,718','August / 4,759'],['Model selection, fold 2','January-August / 38,477','September / 4,670'],['Locked holdout','January-September / 43,147','October / 4,853'],['Final refit','January-October / 48,000','November-December / 12,000 unlabeled']],[125,172,173])
p('Two tree capacities were compared using the mean of August and September MAE. The 31-leaf variant averaged 99.03 USD, compared with 99.31 for 15 leaves. The small difference should not be overinterpreted. October was evaluated after configuration selection, and no configuration changes were made in response to its score.')
rows=[['Period','Model','MAE','RMSE']]
for _,r in M.iterrows():rows.append([r.period,r.model,f'{r.MAE:.2f}',f'{r.RMSE:.2f}'])
table(rows,[80,206,92,92])
p('The final estimator uses 180 boosting iterations, learning rate 0.07, 31 leaves, minimum 45 samples per leaf, L2 regularization 3 and random seed 42. Internal early stopping is disabled to avoid an unplanned random validation split. This is a bounded comparison, not exhaustive hyperparameter optimization.')
p('Training target = posted_rate / distance. Sample weight = distance. Therefore distance x |predicted rate per mile - actual rate per mile| equals the absolute error of the total predicted price. At inference the rate is multiplied by distance; a 0.01 USD lower bound guarantees positive output. The baseline uses the unweighted median training rate per mile multiplied by distance.')
page()
h('4. Fixed December prediction chart')
story.append(Image(str(P/'scorer_results/candidate_december.png'),width=470,height=270,kind='proportional'))
p(f'The provided score.py generated this chart from completed December inputs. Predicted rates range from {S["december_min"]:.2f} to {S["december_max"]:.2f} USD. Route, equipment, weight and distance are fixed; only date changes. No values were hand-shaped to produce this curve.')
h('Interpretation and limitations')
p('Date-derived features produce the variation. Only January-October of one year is labeled, so December behavior and annual seasonality are uncertain. Trees do not extrapolate a time trend beyond their observed split ranges, and no explicit holiday effect is modeled. The curve is a conditional model output, not evidence of learned December market behavior.')
p('October RMSE substantially exceeds MAE, indicating a heavy tail of large errors. Negative mean bias indicates underprediction on average. Robust absolute loss is appropriate for MAE, but may sacrifice performance if the hidden evaluation emphasizes squared error. The employer\'s final metric is unspecified.')
p('Further work would verify signal availability and lineage, inspect high-error shipments with domain experts, obtain additional years, validate on truly held-out geographic groups and assess prediction intervals. No target labels or aggregate target statistics from the final dataset were used.')
h('Reproducibility and assistance')
p('Run python train.py, then run the scorer command in README.md. Requirements are pinned; results/metrics.csv and data_audit.json provide machine-readable evidence. Final predictions are mapped by load_id to template order. The saved model and source inputs are included in the local review package.')
p('Implementation and documentation were prepared with AI assistance. Candidate review and an independently understood walkthrough are required before submission. The repository and Loom link are delivery steps outside this local package.')
p('Sources: supplied assessment PDF, supplied CSV files, README and full score.py provided in chat. Estimator reference: scikit-learn HistGradientBoostingRegressor official documentation, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html','SmallCustom')

def footer(c,doc):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#667788'))
    c.drawString(42,25,'Mohammed Nail Al-Sheyab | Freight Rate Prediction');c.drawRightString(A4[0]-42,25,str(doc.page))
SimpleDocTemplate(str(P/'Freight_Rate_Report.pdf'),title='Freight Rate Prediction',author='Mohammed Nail Al-Sheyab',pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
