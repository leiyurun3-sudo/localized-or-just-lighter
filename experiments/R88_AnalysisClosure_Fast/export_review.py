"""Optional method-anonymized paired review export. Produces no judgments."""
import argparse,csv,hashlib,html,json,random,shutil
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(__file__).resolve().parent/'runtime_base');p.add_argument('--out',type=Path);a=p.parse_args()
 root=a.runtime.resolve();out=(a.out or root/'results'/'blind_review').resolve();plan=read(root/'plan.json')
 if out.exists():raise FileExistsError('Keep previous ratings; choose a new --out directory.')
 jobs={(j['case'],j['seed'],j['arm']):j for j in plan['jobs']};pairs=[]
 for t in plan['tasks']:
  for seed in (0,17):
   for contrast,left,right in [('cross_anchor','REQ_U','U25'),('request_anchor','REQ','GLB_R')]:
    ident=hashlib.sha256(f"R88-review-88103:{t['case']}:{seed}:{contrast}".encode()).hexdigest()[:20]
    arms=[left,right];random.Random(ident).shuffle(arms)
    pairs.append((ident,t,seed,contrast,arms))
 random.Random(88103).shuffle(pairs);rows=[];key=[];cards=[]
 for ident,t,seed,contrast,arms in pairs:
  assets=out/'images'/ident;assets.mkdir(parents=True,exist_ok=True)
  shutil.copy2(root/'data'/t['case']/'source.png',assets/'source.png')
  for label,arm in zip(('A','B'),arms):
   job=jobs[(t['case'],seed,arm)];folder=root/'results'/job['id'];rec=read(folder/'record.json')
   if rec.get('status')!='completed':raise RuntimeError('Incomplete input: '+job['id'])
   # Exact native pixels: locate using the runtime artifact convention.
   candidates=[folder/'output.png',folder/'raw.png',folder/'edited.png']
   image=next((f for f in candidates if f.is_file()),None)
   if image is None:raise FileNotFoundError('Native PNG not found in '+str(folder))
   shutil.copy2(image,assets/(label+'.png'))
  key.append({'pair_id':ident,'case':t['case'],'seed':seed,'contrast':contrast,'A':arms[0],'B':arms[1]})
  rows.append({'pair_id':ident,'rater_id':'','rater_type':'','edit_A':'','edit_B':'','preserve_A':'','preserve_B':'','preservation_preference':'','notes':''})
  pictures=''.join('<figure><img src="images/'+ident+'/'+x+'.png"><figcaption>'+x+'</figcaption></figure>' for x in ('source','A','B'))
  cards.append('<section><h3>'+ident+'</h3><p>Source description: '+html.escape(t['source_prompt'])+'</p><p>Requested target: '+html.escape(t['target_prompt'])+'</p><p>Instruction: '+html.escape(t['edit_rubric'])+'</p><div>'+pictures+'</div></section>')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'ratings_blank.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Paired editing review</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}section{padding:20px;border-bottom:2px solid #999}section div{display:flex}figure{margin:6px;width:32%}img{width:100%;height:auto}</style><h1>Paired editing review</h1><p>Score each output independently: complete requested edit (1/0/unknown), preserve content not requested to change (1/0/unknown). Then choose A/B/tie/unknown for preservation preference. Do not infer editing success from visual similarity. Record rater_type as human or AI; these are separate evaluations.</p>'+''.join(cards),encoding='utf-8')
 (root/'results'/'blind_review_private_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
 print('Exported',len(rows),'pairs. No scores generated. Keep blind_review_private_key.json from raters.')
if __name__=='__main__':main()
