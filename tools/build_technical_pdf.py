"""Create the four-page technical submission from completed aggregate receipts."""
import html
import json
from pathlib import Path
import shutil
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon

ROOT=Path(__file__).resolve().parents[1]
NAVY=colors.HexColor('#173547');TEAL=colors.HexColor('#157d69');PALE=colors.HexColor('#edf4f4');GRAY=colors.HexColor('#536371')
STYLES=getSampleStyleSheet()
STYLES.add(ParagraphStyle(name='TitleTeam',fontName='Helvetica-Bold',fontSize=25,leading=28,textColor=NAVY,spaceAfter=12))
STYLES.add(ParagraphStyle(name='SubtitleTeam',fontName='Helvetica',fontSize=12,leading=16,textColor=GRAY,spaceAfter=15))
STYLES.add(ParagraphStyle(name='SectionTeam',fontName='Helvetica-Bold',fontSize=15,leading=19,textColor=NAVY,spaceBefore=9,spaceAfter=9,keepWithNext=True))
STYLES.add(ParagraphStyle(name='BodyTeam',fontName='Helvetica',fontSize=10.3,leading=14,spaceAfter=8))
STYLES.add(ParagraphStyle(name='SmallTeam',fontName='Helvetica',fontSize=8.6,leading=11.4,textColor=GRAY,spaceAfter=6))
STYLES.add(ParagraphStyle(name='CellTeam',fontName='Helvetica',fontSize=8.7,leading=11.2))
STYLES.add(ParagraphStyle(name='HeadTeam',fontName='Helvetica-Bold',fontSize=8.6,leading=11,textColor=colors.white))
STYLES.add(ParagraphStyle(name='CodeTeam',fontName='Courier',fontSize=8.2,leading=11,backColor=PALE,borderPadding=9,spaceAfter=10))


def P(text,style='BodyTeam'):return Paragraph(text,STYLES[style])


def table(rows,widths):
    cells=[[P(str(value),'HeadTeam' if i==0 else 'CellTeam') for value in row] for i,row in enumerate(rows)]
    result=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c6d2d6'))]))
    return result


def architecture():
    d=Drawing(510,145)
    def box(x,y,w,h,lines,fill=PALE):
        d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=fill,strokeColor=colors.HexColor('#a7bcc3'),strokeWidth=.7))
        for i,line in enumerate(lines):d.add(String(x+w/2,y+h/2+(len(lines)-1)*6-i*12,line,fontName='Helvetica',fontSize=8.9,textAnchor='middle',fillColor=NAVY))
    def arrow(x1,y1,x2,y2):
        d.add(Line(x1,y1,x2,y2,strokeColor=GRAY,strokeWidth=1))
        d.add(Polygon([x2,y2,x2-5,y2+3,x2-5,y2-3],fillColor=GRAY,strokeColor=GRAY))
    box(0,46,100,50,['Automatic tracking','XML + TXT']);box(119,46,110,50,['87 geometry features','+ 21 contact features'])
    box(248,91,115,39,['Frozen v3 boosted path','9 / 19 public games'])
    box(248,16,115,49,['102-second TCN','108 features + masks','65 public games'],colors.HexColor('#d8eee8'))
    box(389,46,119,50,['50/50 mixture','Duration decoder','H / A / D / X'])
    arrow(100,71,119,71);arrow(229,71,248,110);arrow(229,71,248,41);arrow(363,110,389,80);arrow(363,41,389,61)
    return d


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#cfdbdf'));canvas.line(42,37,A4[0]-42,37)
    canvas.setFont('Helvetica',8);canvas.setFillColor(GRAY);canvas.drawString(42,24,'Ruses Cules | More Than a Hack 2027');canvas.drawRightString(A4[0]-42,24,f'Page {doc.page}');canvas.restoreState()


def main():
    base=ROOT/'docs/experiments/pff-mixed';final=json.loads((base/'final_results.json').read_text());decision=final['decision']
    if not decision['status'].startswith('PASS:'):raise ValueError('This submission document requires completed final quality gates')
    proof=json.loads((ROOT/'out/candidates/pff-mixed/verification.json').read_text());selection=json.loads((base/'selection.json').read_text());old=json.loads((base/'old_results.json').read_text());dev=json.loads((base/'pff_dev_results.json').read_text());gaps=json.loads((base/'train_cv_diagnostic.json').read_text());diagnostic=json.loads((base/'spell_diagnostics.json').read_text())['summary']['natural_detected/boundary102']
    if selection['candidate']!='boundary102' or proof['candidate']!='boundary102':raise ValueError('Document template/candidate mismatch')
    natural=decision['comparisons']['canonical/natural_detected'];blind=decision['comparisons']['canonical/label_blind_seed7'];h=old['decisions']['boundary102']['historical'];paired=old['decisions']['boundary102']['paired'];d=dev['decisions']['boundary102']['comparisons']['canonical/natural_detected']
    folder=ROOT/'out/review-ready';folder.mkdir(exist_ok=True);pdf=folder/'Ruses Cules.pdf';csv=folder/'Ruses Cules.csv';shutil.copy2(ROOT/'out/candidates/pff-mixed/Ruses Cules.csv',csv)
    story=[]
    def add(text,style='BodyTeam'):story.append(P(text,style))
    add('Ruses Cules','TitleTeam');add('Possession from noisy tracking<br/>Validated method: boundary102','SubtitleTeam')
    add('1. Task and method','SectionTeam')
    add('We label every video frame as home possession (H), away possession (A), dead ball (D), or outside the published playing periods (X). The output contains exactly 199,319 ordered rows, with header <font name="Courier">frame,label</font>. The method uses the supplied automatic tactical XML/TXT; no manual correction of the provided game is made.')
    add('Missing ball observations make nearest-player rules unstable. We combine an existing boosted geometry model with a longer temporal network, which uses observed contacts, team shape and motion around an occlusion. An auxiliary change loss encourages useful possession boundaries. This produces a reviewable timeline for match analysis from already available tracking; it does not require a new video detector.')
    story.append(architecture());story.append(Spacer(1,5))
    add('The neural branch receives 108 geometry/contact features plus 108 finite-value masks at 5 Hz. Seven residual blocks, with two 3-tap convolutions each and dilations 1/2/4/8/16/32/64, give a 509-step (101.8-second) receptive field. Width is 48, dropout 0.15. Two logits predict dead probability and home possession conditional on live play. Direct and team-swapped predictions enforce home/away symmetry.')
    add('The inherited v3 path uses the original nine-game boosted/stacked classifiers and a nineteen-game team head. Both probability channels are averaged 50/50 with the new neural branch. A cost-9 duration Viterbi decoder retains the existing one-second minimum live/dead intervals. Four internal states remember dead-after-home versus dead-after-away; both output D. Restart snapping is disabled.')
    add('Possession and input rules','SectionTeam')
    add('Controlled possession is inferred from geometry and context. Loose balls and challenges use learned contact/context probabilities and the transition penalty; a nearest touch does not automatically force a turnover. A stoppage splits possessions even when the same team restarts. Track slots are fragments, not stable identities. Video, homography and Smart Tagging are unused. Published period/kickoff alignment is retained; no target ground truth trains, selects or calibrates this method.')

    story.append(PageBreak());add('2. Training data and isolation','SectionTeam')
    story.append(table([
        ['Public source','Games','H/A/D frames','Reference and source terms'],
        ['IDSSE / DFL','7','1,002,644','Provider BallStatus / BallPossession. CC BY 4.0.'],
        ['Metrica samples 1 / 2','2','286,162','Provider event intervals converted by code. Event-derived proxy; acknowledge Metrica.'],
        ['SkillCorner A-League','20','2,955,525','Automatic H/A phases; complement inside metadata periods is a D proxy. MIT; credit SkillCorner / PySport.'],
        ['Original PFF World Cup release','36','5,636,115','Embedded PFF event annotations converted with the primary SDK state machine. Original provider access; no raw redistribution.']
    ],[110,46,79,276]));story.append(Spacer(1,9))
    add('The neural full fit uses 65 distinct games and 9,880,446 unaugmented reference-period frames. Frame counts are not independent samples. Provider annotation workflows were not independently audited. The inherited boosted and team components keep their separate nine- and nineteen-game registries.')
    add('PFF features use only raw VISIBLE ball/player detections. ESTIMATED tracks, event-informed smoothed ball positions, event/possession fields and identities never enter features. Reference reconstruction is separate: OUT/END means dead; kickoff/OTB means live; home_ball supplies ownership. Canonical labels reset at independent periods and mask conflicting duplicates. A fixed SDK last-row shadow is checked separately. These are provider-event proxies, not corrected frame-by-frame competition truth.')
    add('Training uses seed 20261113, 6,000 AdamW updates, learning rate 0.001, weight decay 0.001, batch 16 and 256 supervised 5-Hz steps per crop. Provider groups are sampled equally (DFL / Metrica / SkillCorner / PFF), then games and scenarios uniformly. Team swaps are random. SkillCorner D-proxy loss has weight 0.25. The auxiliary boundary BCE has weight 0.15 and positive weight 8; unknown/context-only pairs are masked.')
    add('Features are standardized using training observations only, with a 0.01 standard-deviation floor, clipping to [-8, 8] and explicit missing masks. Every complete scored fold is excluded from all direct and nested learned components and the normalizer. Actual checkpoint normalizers are compared exactly with training-only recomputation. No augmentation sibling crosses a game split.')
    add('The original PFF split was declared before access: 36 training, 10 development, 12 final and 6 excluded country bridges. The final groups A/H national teams are absent from PFF training/development. One candidate and source/model/reader/scenario hashes were sealed before any final contents. No model switching or post-test tuning followed. All 12 final games and both sensor cases were retained.')
    add('Validation-only sources also include Metrica sample 3, one Groningen/SciSports WCSF sample, and nine European SkillCorner games (conditional H/A only). All older cohorts are reused research; only the sealed PFF final is the prospective final estimate.', 'SmallTeam')
    add('Sources: <a href="https://doi.org/10.1038/s41597-025-04505-y" color="#157d69">IDSSE / DFL paper</a>; <a href="https://github.com/metrica-sports/sample-data" color="#157d69">Metrica sample-data</a>; <a href="https://github.com/SkillCorner/opendata" color="#157d69">SkillCorner Open Data</a>; <a href="https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset" color="#157d69">PFF original release</a>. Versions, file hashes and SDK pin are in the repository experiment records.', 'SmallTeam')

    story.append(PageBreak());add('3. Quality evidence','SectionTeam')
    add('The unmodified official metric matches maximal H/A possession spells at temporal IoU thresholds 0.3/0.4/0.5/0.6/0.7 and averages the F1 scores. MacroF1 and H/A/D F1 guard frame quality. Scores below are paired on the same games; absolute values across different cohorts are not comparable train/test gaps.')
    story.append(table([
        ['Evaluation','V4 spellF1','New','Gain','Game wins'],
        ['29 reused development games',f"{h['baseline_composite']:.4f}",f"{h['candidate_composite']:.4f}",f"{h['gain']:+.4f}",f"{h['game_wins']}/29"],
        ['Same complete-fold exclusions',f"{paired['baseline_composite']:.4f}",f"{paired['candidate_composite']:.4f}",f"{paired['gain']:+.4f}",f"{paired['game_wins']}/29"],
        ['PFF development - natural',f"{d['baseline']:.4f}",f"{d['candidate']:.4f}",f"{d['gain']:+.4f}",f"{d['wins']}/10"],
        ['Sealed PFF final - natural',f"{natural['baseline']:.4f}",f"{natural['candidate']:.4f}",f"{natural['gain']:+.4f}",f"{natural['wins']}/12"],
        ['Sealed PFF final - sensor stress',f"{blind['baseline']:.4f}",f"{blind['candidate']:.4f}",f"{blind['gain']:+.4f}",f"{blind['wins']}/12"]
    ],[219,73,69,75,75]));story.append(Spacer(1,10))
    add(f"All predeclared gates pass. Final group gains: A {decision['group_gains']['A']:+.4f}, H {decision['group_gains']['H']:+.4f}. Final natural macroF1 gain is {natural['macro_gain']:+.4f}; H/A/D F1 gains are {natural['class_f1_gains']['H']:+.4f} / {natural['class_f1_gains']['A']:+.4f} / {natural['class_f1_gains']['D']:+.4f}. The descriptive paired whole-game 95% gain interval is [{decision['descriptive_paired_game_95ci'][0]:.4f}, {decision['descriptive_paired_game_95ci'][1]:.4f}]. Shared national teams and only two groups limit independence assumptions.")
    story.append(Image(str(base/'final_game_gains.png'),width=511,height=209));story.append(Spacer(1,6))
    add('The data-expanded 25-second control also passes development, but the combined longer-context/boundary model scores higher under the predeclared selection rule. The two changes are not separately ablated. Reused external losses remain: noisy Metrica3 spellF1 -0.0090 and European conditional H/A macroF1 -0.0037, both within the fixed 0.01 guard.', 'SmallTeam')

    story.append(PageBreak());add('4. Reproduce and interpret','SectionTeam')
    add('Use Python 3.12 and the pinned requirements. Frozen public-trained weights are included; generating the organizer file requires only the organizer tactical XML/TXT. It does not download external matches, retrain, read video or score target ground truth. The data directory must contain tactical/BAR-ATH_tactical_atd.xml and .txt.')
    commands="""python3.12 -m venv .venv
.venv/bin/pip install -r requirements-temporal.txt
.venv/bin/python tools/fetch_reference.py
.venv/bin/python tools/generate_mixed_candidate.py \\
  --bundle assets/pff-mixed-candidate/bundle.json \\
  --data-dir /absolute/path/to/data \\
  --output 'Ruses Cules.csv'
.venv/bin/python ref/official/evaluation/validate_submission.py \\
  'Ruses Cules.csv'"""
    story.append(Preformatted(commands,STYLES['CodeTeam']))
    add(f"Two clean committed-source CPU generations passed the official validator and produced identical CSV bytes in {proof['runs'][0]['seconds']:.3f}s and {proof['runs'][1]['seconds']:.3f}s. Observed peak child RSS was {proof['peak_child_RSS_mb']/1024:.2f} GiB. Organizer/model input hashes remained unchanged. This repeats frozen-weight inference, not training. Source: {proof['source_commit'][:12]}; the same inference core remained frozen through final validation.")
    digest=proof['runs'][0]['csv_sha256'];add('CSV SHA-256:<br/><font name="Courier" size="8.7">'+digest[:32]+'<br/>'+digest[32:]+'</font>')
    add('Model checks verify the bundle, checkpoint, metadata and inherited asset hashes before loading. The new checkpoint uses weights-only deserialization. Optional raw-data fitting and all whole-fold provenance are documented. The source repository contains neither organizer match assets nor raw PFF tracking, references or frame predictions.')
    add('Overfit and remaining limits','SectionTeam')
    rows=[['Standalone neural branch','Full-fit spellF1','Whole-fold-out','Gap']]
    for provider,r in gaps['summary'].items():rows.append([provider+f" ({r['games']} games)",f"{r['full_fit_spellF1']:.4f}",f"{r['wholefold_out_spellF1']:.4f}",f"{r['optimism_gap']:+.4f}"])
    story.append(table(rows,[219,99,108,85]));story.append(Spacer(1,8))
    add('The 0.151 Metrica gap indicates a material source-specific overfit risk on only two games. This diagnostic is for the standalone neural branch, not the deployed 50/50 fusion. It was descriptive after selection and did not change the frozen model. The untouched-country final supports generalization beyond memorizing training games; it cannot prove absence of overfit on all football or predict competition rank.')
    add(f"Short possessions remain difficult: on PFF development, the mean per-game predicted/reference spell-count ratio is {diagnostic['spell_count_ratio']:.3f}; predicted mean duration is {diagnostic['mean_predicted_spell_seconds']:.2f}s versus reference {diagnostic['mean_reference_spell_seconds']:.2f}s. Long occlusions, incomplete player coverage, ambiguous control, annotation differences and anonymous WCSF provenance remain limitations. The method is offline and uses future context.")
    add('Next work: add independent Metrica-like games, reduce the two-game source\'s sampling weight, and improve short-spell duration modelling. A changed model requires a new untouched reserve.')
    add('Source and full evidence: <a href="https://github.com/8Tigran8/silver-octo-winner" color="#157d69">github.com/8Tigran8/silver-octo-winner</a><br/>Experiment records: docs/experiments/pff-mixed/REPORT.md', 'SmallTeam')
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=48,title='Ruses Cules - Possession from noisy tracking',author='Ruses Cules',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(pdf)


if __name__=='__main__':main()
