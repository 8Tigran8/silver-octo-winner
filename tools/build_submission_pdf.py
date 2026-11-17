"""Build the selected submission report from frozen evaluation evidence."""
import argparse, json
from pathlib import Path
from html import escape
from reportlab.pdfgen import canvas as canvas_module
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle

ROOT=Path(__file__).resolve().parents[1]
NAVY=colors.HexColor('#132B3C');TEAL=colors.HexColor('#087F80');INK=colors.HexColor('#263D4B');GRAY=colors.HexColor('#607580');PALE=colors.HexColor('#EDF5F4');LINE=colors.HexColor('#D8E4E5');GOLD=colors.HexColor('#AA692F')
FONT='/System/Library/Fonts/Supplemental/'
for name,file,fallback in [('Body','Arial.ttf','Helvetica'),('Body-Bold','Arial Bold.ttf','Helvetica-Bold'),('Body-Italic','Arial Italic.ttf','Helvetica-Oblique')]:
 if Path(FONT+file).exists():pdfmetrics.registerFont(TTFont(name,FONT+file))
 else:pdfmetrics.registerFont(pdfmetrics.Font(name,fallback,'WinAnsiEncoding'))
ST={
 'title':ParagraphStyle('title',fontName='Body-Bold',fontSize=29,leading=33,textColor=NAVY,spaceAfter=15),
 'subtitle':ParagraphStyle('subtitle',fontName='Body',fontSize=12,leading=17,textColor=GRAY,spaceAfter=16),
 'h1':ParagraphStyle('h1',fontName='Body-Bold',fontSize=22,leading=27,textColor=NAVY,spaceAfter=14),
 'h2':ParagraphStyle('h2',fontName='Body-Bold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=13,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='Body',fontSize=9.6,leading=14.2,textColor=INK,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.1,leading=11.5,textColor=GRAY,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.5,leading=11.7,textColor=INK),
 'head':ParagraphStyle('head',fontName='Body-Bold',fontSize=8.1,leading=11,textColor=colors.white),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=7.6,leading=11.2,textColor=NAVY,backColor=PALE,borderPadding=11,spaceAfter=12),
}

class ReportCanvas(canvas_module.Canvas):
 def __init__(self,*a,**kw):super().__init__(*a,**kw);self.setCreator('Ruses Cules technical report');self.setAuthor('Ruses Cules')

def frame(c,doc):
 c.saveState();w,h=A4
 c.setFillColor(TEAL);c.rect(42,h-35,24,3,fill=1,stroke=0)
 c.setFont('Body-Bold',8);c.setFillColor(GRAY);c.drawString(75,h-35,'RUSES CULES  /  TECHNICAL REPORT')
 c.setFont('Body',8);c.drawRightString(w-42,h-35,'17 NOVEMBER 2026')
 c.setStrokeColor(LINE);c.line(42,37,w-42,37);c.setFont('Body',7.5);c.setFillColor(GRAY)
 c.drawString(42,23,'More Than a Hack 2027  |  Team possession');c.drawRightString(w-42,23,str(doc.page));c.restoreState()

def diagram():
 d=Drawing(510,142)
 boxes=[(0,69,140,52,'TACTICAL TRACKING',['Player + ball geometry','Period-local features']),
        (184,94,140,43,'BALL OBSERVED',['Boosted + temporal, 50/50']),
        (184,18,140,58,'BALL MISSING',['D: 75% specialist + 25% activity','H/A: ownership specialist']),
        (365,61,145,63,'TEMPORAL DECODER',['Duration + transition costs','H / A / D / X'])]
 for x,y,w,h,title,lines in boxes:
  d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=PALE,strokeColor=LINE,strokeWidth=.8));d.add(String(x+10,y+h-15,title,fontName='Body-Bold',fontSize=7.7,fillColor=TEAL))
  for i,line in enumerate(lines):d.add(String(x+10,y+h-30-i*12,line,fontName='Body',fontSize=7.1,fillColor=INK))
 for pts in [(140,95,162,95),(162,95,162,116),(162,116,184,116),(162,95,162,47),(162,47,184,47),(324,116,345,116),(345,116,345,93),(345,93,365,93),(324,47,345,47),(345,47,345,80),(345,80,365,80)]:d.add(Line(*pts,strokeColor=GRAY,strokeWidth=.8))
 return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'out/submission-final/Ruses Cules.pdf');a=ap.parse_args()
 choice=json.loads((ROOT/'docs/experiments/sensor-relevance/selection_override.json').read_text());assert choice['selected']=='uniform_temporal_focus'
 audit=json.loads((ROOT/'docs/experiments/final-opened-comparison/decision.json').read_text());sensors=json.loads((ROOT/'docs/experiments/sensor-relevance/results.json').read_text())
 rows={r['scenario']:r for r in audit['comparisons'] if r['cohort']=='all18' and r['reference']=='canonical'}
 story=[];md=['# Ruses Cules: team possession from tactical tracking',''];W=A4[0]-84
 def p(text,style='body'):
  story.append(Paragraph(escape(text).replace('\n','<br/>'),ST[style]));md.extend([text,''])
 def h(text):p(text,'h2');md[-2]='## '+text
 def page(k,title):
  if k>1:story.append(PageBreak())
  p(f'{k:02d} / '+title,'h1');md[-2]='## '+title
 def tab(data,widths):
  processed=[[Paragraph(escape(str(x)),ST['head' if i==0 else 'cell']) for x in row] for i,row in enumerate(data)]
  t=Table(processed,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.6,LINE)]));story.extend([t,Spacer(1,9)])
  md.append('| '+' | '.join(map(str,data[0]))+' |');md.append('| '+' | '.join('---' for _ in data[0])+' |')
  md.extend('| '+' | '.join(map(str,row))+' |' for row in data[1:]);md.append('')

 p('Team possession\nfrom tactical tracking','title')
 p('Missing-ball specialists with duration-constrained temporal decoding','subtitle')
 tab([['SELECTED MODEL','OUTPUT','AUDIT SPELL F1'],['uniform_temporal_focus','199,319 frames / H A D X','0.6746 on 18 PFF games']],[205,156,150])
 h('What we built')
 p('We estimate which team owns the ball and when play is dead from tactical ATD XML/TXT. We combine geometry-based predictors with temporal specialists for missing-ball intervals, then decode complete possession segments. The method runs offline on CPU and uses future context.')
 story.append(diagram());md.extend(['Pipeline: ATD geometry -> observed/missing-ball predictors -> duration decoder -> H/A/D/X CSV.',''])
 h('Observed ball and missing ball')
 p('When the ball is observed, we retain an equal mixture of the inherited boosted predictors and the weighted temporal network for dead/live and conditional H/A probabilities. During gaps, dead/live uses 75% of a temporal specialist and 25% of a player-activity temporal network. A separate ownership specialist supplies conditional H/A.')
 p('Geometry networks use 108 standardized features plus finite masks at 5 Hz, width 48 and seven residual dilation blocks: 101.8 seconds of context. Activity uses 64 player-only features plus masks, width 32 and six blocks: about 50.6 seconds. We average direct and team-swapped inference.')
 h('Input contract')
 p('We validate XML teams, track slots and contiguous ATD frame IDs, convert coordinates to metres and reset interpolation, derivatives and neural context at each period. A detection needs two finite coordinates. Missingness remains explicit. Video, homography and Smart Tagging do not enter inference. We exclude target possession labels and human-produced events from fitting and selection.')
 p('H = home possession; A = away possession; D = dead ball. X marks excluded intervals from published period metadata. ATD one-based IDs map to zero-based output rows. The CSV header is frame,label.','small')

 page(2,'Training data and possession rules')
 p('We fit on 65 distinct external matches. The inherited nine-game and 19-game boosted registries are overlapping subsets of this union, not additional matches. Development and audit matches remain outside fitting and normalization.')
 tab([['SOURCE','GAMES','REFERENCE FRAMES','REFERENCE'],['IDSSE / DFL','7','1,002,644','Provider BallStatus / BallPossession'],['Metrica samples 1 and 2','2','286,162','Intervals from sample events'],['SkillCorner A-League','20','2,955,525','Automatic H/A phases; complementary D proxy'],['PFF World Cup 2022','36','5,636,115','Pinned SDK event-state reconstruction']],[124,60,106,221])
 p('Frame counts precede augmentation. Geometry specialists use 139 natural/degraded scenarios from the same 65 games; the activity network uses 72 natural/blind sequences from the 36 PFF training games. For PFF predictor geometry we use raw visible detections, excluding event-informed/estimated ball positions, labels and identifiers. References have different annotation semantics and include proxies.')
 h('Frozen training configuration')
 p('Both geometry specialists continue the parent network for 4,000 steps with seed 20261117 and its training-only normalizer. Dead/live uses uniform training windows; H/A training focuses additional windows on ownership errors during gaps. The activity network trains for 4,000 steps with seed 20261118 and training-period normalization. Its inputs exclude ball coordinates, gap duration, identities, timestamps and other model outputs. Metadata and weight hashes specify the exact configurations.')
 h('Possession and restart decisions')
 p('The decoder has H, A, dead-after-H and dead-after-A states; both dead states emit D. Minimum live duration is 0.4 s, minimum D duration 1 s, H/A transition cost 4.5 and other transition costs 9, with stride 5. Restart snapping is disabled. Published period and kickoff metadata determine excluded blocks and pre-restart D.')
 p('A loose ball or ball flight does not automatically stop play. A touch need not change the owning team. H-D-H contains two possessions; contested live possession has no separate output class. Temporal decoding means changed gap probabilities can also affect neighboring observed-ball labels.')
 h('Source access and attribution')
 p('DFL uses CC BY 4.0; SkillCorner uses MIT with attribution. Metrica supplies sample tracking/events. We obtained original PFF access through its provider; a standard permissive license for that release has not been established, and the mirror\'s GPL label is not applied to it. Source-access qualifications remain in docs/PFF_SOURCE.md. We exclude raw organizer and external data from the repository.','small')

 page(3,'Measured performance')
 p('We score maximal H/A possession spells with the official metric, averaging F1 over temporal IoU thresholds 0.3, 0.4, 0.5, 0.6 and 0.7, then average across matches. Frame-level macro F1 and per-class F1 provide additional regression checks.')
 tab([['10 DEVELOPMENT GAMES','PREVIOUS','SELECTED','WINS'],['Original tracking','0.6800','0.7017','10 / 10'],['Original blind stress, seed 7','0.6673','0.6863','9 / 10'],['Blind stress, seed 23 XOR match ID','0.6672','0.6870','8 / 10'],['Blind stress, seed 101 XOR match ID','0.6630','0.6884','10 / 10']],[269,80,82,80])
 p('These ten matches were repeatedly used for development. Four scenarios still represent ten matches. The original +0.02 effect-size gate failed in two stress cases (+0.0190 and +0.0198); those failures remain in the experiment records.','small')
 tab([['18 PREVIOUSLY OPENED AUDIT GAMES','PREVIOUS','SELECTED','GAIN'],*[ [label,f"{rows[key]['old_spell']:.4f}",f"{rows[key]['new_spell']:.4f}",f"{rows[key]['gain']:+.4f}"] for key,label in [('natural_detected','Original tracking'),('label_blind_seed7','Original blind stress'),('vv_hz10_noise0p5','VV: 10 Hz / 0.5 m jitter'),('vv_hz15_noise0p5','VV: 15 Hz / 0.5 m jitter'),('vv_hz10_noise2','VV: 10 Hz / 2 m jitter'),('vv_hz15_noise2','VV: 15 Hz / 2 m jitter')]]],[269,80,82,80])
 p('The selected model wins 18/18 original-tracking games and 17/18 original-blind games. The descriptive paired whole-match bootstrap 95% interval for the original-tracking gain is +0.0270 to +0.0441 (10,000 resamples). Reuse, common teams and one provider limit that interval\'s interpretation.')
 h('Short spells and ball gaps')
 tab([['18 GAMES, ORIGINAL TRACKING','PREVIOUS','SELECTED'],['Recall of possessions shorter than 3 s, IoU 0.5','14.23%','23.77%'],['Wrong team / missing-ball reference-live frames','11.12%','9.94%'],['False D / missing-ball reference-live frames','18.88%','17.70%'],['False LIVE / missing-ball reference-dead frames','2.61%','2.87%']],[319,96,96])
 p('The 18 games did not enter fitting, but were previously opened and now inform release selection. They are not a fresh prospective holdout. Canonical and SDK-shadow references reconstruct the same events. Full match IDs, failures and diagnostics: docs/experiments/final-opened-comparison/REPORT.md.','small')

 page(4,'Why we selected this version')
 p('We first rolled back under a conservative rule requiring the new model to avoid regressions on all four VV stress presets. After a sensor-relevance review, we explicitly overrode that release choice. The original gate still FAILS for this model; we did not rewrite it as a passed experiment.')
 h('What the additional stress changes')
 p('Each preset combines 10/15 Hz observation thinning, independent Gaussian x/y jitter of 0.5/2 m for players and ball, and two Poisson ball-gap starts per minute with 2-6 s gaps. It preserves existing gaps and leaves non-observation grid rows empty, without holding or interpolation. The presets have not been calibrated to the organizer\'s tracking error distribution.')
 tab([['INPUT DIAGNOSTIC','SUPPLIED TRACKING','STATIONARY VV PROBE'],['Rows with all players missing','0%','40-60%'],['Median player speed','1.36 m/s from coordinates','8.43 / 12.56 m/s at 0.5 m jitter'],['Median apparent speed at 2 m jitter','Not applied to target','33.72 / 50.26 m/s'],['Mean players per supplied row','20.49','22 only on retained rows']],[183,145,183])
 p('These diagnostics use automatic tactical geometry and synthetic stationary tracks only, with no target possession labels. They show that the presets create severe combined corruption unlike the supplied smooth player input. Smoothness is not proof of correct positions or possession labels. No model weights, decoder parameters or predictions changed during the review.')
 h('The trade-off')
 p('We favor the consistent original/original-blind gains and improved short-spell/gap decisions over a veto based on these uncalibrated failures. The older model remains preferable under a minimax objective over the four extreme presets. The selected model has stronger measured results on ordinary PFF tracking. We have no empirical probability model that proves which one maximizes hidden-match score.')
 p('This is a post-result engineering choice and introduces additional selection bias. It is documented in docs/experiments/sensor-relevance/selection_override.json. The prior comparison results and rollback package remain available.')
 h('Limits that still matter')
 p('The target has long real ball gaps: median 10.72 s and 95th percentile 50.80 s within the published playing blocks. Most short possessions remain unresolved; the selected model matched 0/97 subsecond spells on the ten natural development games. Independent La Liga evaluation is absent, source labels are proxies, and all validation quality comes from one provider/tournament. We cannot infer hidden accuracy or a finalist probability.')

 page(5,'Reproduction and handoff')
 p('Use Python 3.12 and the main branch. The frozen CPU weights require only tactical/BAR-ATH_tactical_atd.xml and .txt from the organizer data bundle. Training downloads and research caches are unnecessary.')
 commands="""git clone --branch main --single-branch \\
  https://github.com/8Tigran8/silver-octo-winner.git ruses-cules
cd ruses-cules
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_final_candidate.py \\
  --bundle assets/final-candidate/bundle.json \\
  --data-dir /absolute/path/to/data \\
  --output 'Ruses Cules.csv'
.venv/bin/python tools/final_submission_gate.py \\
  --csv 'Ruses Cules.csv'"""
 story.append(Preformatted(commands,ST['code']));md.extend(['```sh',commands,'```',''])
 h('Exact output')
 p('The generator refuses an existing output. The CSV has frame,label and 199,319 contiguous rows. Expected SHA-256:')
 digest='897db60c18fefe362acee239665136d47a8b2c87a54eb8eae67231b0dcdf9e33';story.append(Preformatted(digest[:32]+'\n'+digest[32:],ST['code']));md.extend([digest,''])
 h('Verification evidence')
 p('Four clean-source CPU generations (macOS ARM twice, offline Linux ARM and emulated Linux AMD64) produced identical bytes and passed the official validator. These runs used one physical host. Our teammate separately reported two matching runs for this candidate on a Linux ARM server, with the same hash and validator PASS; that external evidence is a teammate report. The 176-test regression suite passes. These checks reproduce inference, not weight training or hidden-match accuracy.')
 p('The manifest binds inference source, model weights, organizer inputs and the output hash. Generation source: 98e931ccc2eea3e81bcb8aa4b04b90c49bc64e4e. Later documentation and decision revisions retain the same inference/model bytes. Old weighted_short_gap remains available under releases/weighted-short-gap/.')
 h('Sources and full records')
 p('DFL: huggingface.co/datasets/pysport/idsse-data (Bassek et al., 2025). Metrica: github.com/metrica-sports/sample-data. SkillCorner: github.com/SkillCorner/opendata. PFF: blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset. Exact source files, versions, SDK pins and training registries are recorded in the repository; source-access qualifications are in docs/PFF_SOURCE.md.','small')
 p('Official protocol: github.com/metrica-sports/more-than-a-hack-2027. Frozen comparison: docs/experiments/final-opened-comparison/. Sensor review and explicit choice: docs/experiments/sensor-relevance/. Portable/research parity and clean-source receipts: docs/releases/uniform-temporal-focus/.','small')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 doc=SimpleDocTemplate(str(a.output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=62,bottomMargin=52,title='Ruses Cules - Team possession from tactical tracking',author='Ruses Cules',pageCompression=1)
 doc.creator='Ruses Cules technical report'
 doc.build(story,onFirstPage=frame,onLaterPages=frame,canvasmaker=ReportCanvas)
 a.output.with_suffix('.md').write_text('\n'.join(md));print(a.output)
if __name__=='__main__':main()
