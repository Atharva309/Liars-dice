// Paper A runner: node run_key.js <n0> <n1> <starter> [ids...]  -> out/key_<n0>_<n1>_<s>.json
const fs=require('fs'),{loadKey,grade}=require('./grade.js'),{students,cachedDist}=require('./students.js');
const [a,b,s]=process.argv.slice(2,5).map(Number),ids=process.argv.length>5?process.argv.slice(5):students(Math.max(a,b)),t0=Date.now(),K=loadKey(a,b,s),file=`${__dirname}/out/key_${a}_${b}_${s}.json`;
console.error(`key (${a},${b}) starter ${s}: gap ${K.gap.toExponential(2)}, seat values ${K.value.map(v=>v.toFixed(5))}  loaded in ${((Date.now()-t0)/1000).toFixed(1)}s`);
const out=fs.existsSync(file)&&process.argv.length>5?JSON.parse(fs.readFileSync(file)):{n0:a,n1:b,starter:s,gap:K.gap,value:K.value,res:{}};out.gap=K.gap;out.value=K.value;
const mathD=cachedDist('math');
for(const id of ids)for(const seat of [0,1]){const t1=Date.now(),d=cachedDist(id),R=grade(K,seat,d,mathD,{top:4});R.ms=Date.now()-t1;R.check=K.br[seat]-R.win-R.act.loss;out.res[id+'@'+seat]=R;
 console.error(id.padEnd(10),'seat',seat,`wins round ${(R.win*100).toFixed(2)}% (perfect ${(K.value[seat]*100).toFixed(2)}%)  paper: correct ${(100*R.std.ok/R.std.n).toFixed(1)}%  lost/decision ${(100*R.std.loss/R.std.n).toFixed(2)}  | own games: correct ${(100*R.act.ok/R.act.n).toFixed(1)}% lost/decision ${(100*R.act.loss/R.act.n).toFixed(2)}  identity ${R.check.toExponential(1)}  nodes ${R.nodes} evals ${d.evals} ${R.ms}ms`)}
fs.writeFileSync(file,JSON.stringify(out));
