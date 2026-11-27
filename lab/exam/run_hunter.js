// Paper B runner: node run_hunter.js <mode> <out-tag> [ids...]
//  duel<n>  : one-on-one match, both start with n dice (all smaller endgames are solved on the way)
//  table<n> : four players with n dice each, the hunter and three copies of the bot
const fs=require('fs'),{distFor}=require('./lib.js'),H=require('./hunter.js'),{students}=require('./students.js');
const mode=process.argv[2],tag=process.argv[3],n=+mode.replace(/\D/g,''),table=mode.startsWith('table'),ids=process.argv.length>4?process.argv.slice(4):students(n),file=`${__dirname}/out/hunter_${tag}.json`;
const out=fs.existsSync(file)?JSON.parse(fs.readFileSync(file)):{mode,n,res:{}},BUDGET=+process.env.BUDGET||2e8,LIMIT=+process.env.LIMIT||150,EPSS=(process.env.EPS||'1e-10,1e-7,1e-5,1e-4,1e-3').split(',').map(Number);
for(const id of ids){const t0=Date.now(),bot=H.makeBot(id,distFor(id)),memo=new Map(),states={};let nodes=0,fail=false;
 // retry a round with coarser pruning if it is too big; the reported error bound says how much that could hide
 const opt={budget:BUDGET,log:(key,r)=>{states[key]={v:+r.value.toFixed(6),err:+r.err.toExponential(2),nodes:r.nodes,eps:opt.eps};nodes+=r.nodes}};
 const orig=H.hunterRound;const start=table?[n,n,n,n]:[n,n],starters=table?[0,1,2,3]:[0,1],vals=[];let err=0;
 let ei=0;try{for(const s of starters){let r=null;for(;ei<EPSS.length;ei++){try{opt.eps=EPSS[ei];opt.deadline=Date.now()+LIMIT*1000;r=H.hunterMatch(start,s,bot,memo,opt);break}catch(e){if(e.message!=='budget')throw e;console.error('  ',id,'starter',s,'eps',EPSS[ei],'too big, coarser pruning')}}
   if(!r){fail=true;break}vals.push(r.value);err=Math.max(err,r.err)}}catch(e){console.error(id,'failed',e.message);fail=true}
 const R=fail?{fail:true}:{value:vals.reduce((a,b)=>a+b,0)/vals.length,byStarter:vals,err,states,nodes,evals:bot.evals,sec:(Date.now()-t0)/1000};out.res[id]=R;
 console.error(id.padEnd(10),mode,fail?'TOO BIG':`hunter wins ${(R.value*100).toFixed(2)}% ± ${(R.err*100).toFixed(3)}  [by starter ${vals.map(v=>(v*100).toFixed(1)).join(' / ')}]  nodes ${(nodes/1e6).toFixed(2)}M  evals ${bot.evals}  ${R.sec.toFixed(1)}s`);
 fs.writeFileSync(file,JSON.stringify(out))}
