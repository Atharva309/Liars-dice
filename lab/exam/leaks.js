// Check the two leaks shown on the page by playing them against the shipped (sampling) bot code.
const {CALL,legal,playFor}=require('./lib.js'),rnd=n=>Math.floor(Math.random()*n),roll=n=>Array.from({length:n},()=>1+rnd(6)).sort(),M=20000;
{const P=playFor('new:dqn');let raised=0,won=0;
 for(let g=0;g<M;g++){const me=roll(2),bot=roll(2),f=[1,2,3].find(x=>!me.includes(x)),open=6+(f-1),mv=P([bot,2,2,1,open,-1],legal(open,4),{order:[1,0],total:4,hist:[[0,open]],counts:[2,2]});
  if(mv===12+(f-1)){raised++;const c=me.filter(d=>d===f).length+bot.filter(d=>d===f).length;if(c<3)won++}}
 console.log(`Exploiter: you open "two" of a low number you lack; it answered "three" of that number in ${(100*raised/M).toFixed(2)}% of ${M} rounds; calling then won ${(100*won/M).toFixed(2)}% of all rounds`)}
{const P=playFor('new:ppo');let lacks=0,one=0,iLack=0,won=0;
 for(let g=0;g<M;g++){const me=roll(2),bot=roll(2),mv=P([bot,2,2,1,-1,-1],legal(-1,4),{order:[0,1],total:4,hist:[],counts:[2,2]}),f=mv%6+1,q=((mv/6)|0)+1;
  if(q===1)one++;if(!bot.includes(f))lacks++;if(!me.includes(f)){iLack++;if(me.filter(d=>d===f).length+bot.filter(d=>d===f).length<q)won++}}
 console.log(`Gut Player: its opening claim was "one X" in ${(100*one/M).toFixed(2)}% of rounds and named a number it did not hold in ${(100*lacks/M).toFixed(2)}%; when you lack that number too (${(100*iLack/M).toFixed(1)}% of rounds), calling Liar won ${(100*won/iLack).toFixed(2)}% of the time`)}
