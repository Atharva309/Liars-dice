// Exact solver for one-on-one Liar's Dice rounds (CFR+, alternating updates, linear averaging).
// Same game and storage layout as duel.js. Usage: ./solver n0 n1 starter maxIters targetGap maxMinutes out.f32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define MAXNB 30
#define MAXH 64
#define MAXD (MAXNB+3)
static int n0_,n1_,starter,N,NB,Hn[2],Hmax,nd[2];
static double prior[2][MAXH];static unsigned char cf[2][MAXH][6];
static double baseK[MAXNB+1];static size_t SIZE;
static float *R,*S;
static double rr[2][MAXD][MAXH],vv[MAXD][MAXH],sig[MAXD][MAXNB*MAXH],va[MAXD][MAXNB*MAXH],W;
static int cur[8];
static void gen(int pl,int n,int start,int len){
 if(len==n){int h=Hn[pl]++,c[6]={0,0,0,0,0,0};static const double fact[]={1,1,2,6,24,120};for(int i=0;i<n;i++)c[cur[i]-1]++;double p=fact[n];for(int f=0;f<6;f++){p/=fact[c[f]];cf[pl][h][f]=(unsigned char)c[f];}prior[pl][h]=p/pow(6,n);return;}
 for(int d=start;d<=6;d++){cur[len]=d;gen(pl,n,d,len+1);}
}
static inline size_t off(unsigned mask,int k){return k<0?0:(size_t)(baseK[k]+(double)(mask-(1u<<k))*(NB-k)*Hmax);}
static inline void tails(int pl,const double*reach,int f,double*T){int n=nd[pl],H=Hn[pl];for(int t=0;t<=n+1;t++)T[t]=0;for(int h=0;h<H;h++)T[cf[pl][h][f]]+=reach[h];for(int t=n-1;t>=0;t--)T[t]+=T[t+1];}
static inline double tl(const double*T,int n,int need){return need<=0?T[0]:need>n?0:T[need];}
static inline void strat(const float*X,size_t o0,int H,int A,double*sg){
 for(int h=0;h<H;h++){const float*x=X+o0+(size_t)h*A;double s=0;for(int a=0;a<A;a++)s+=x[a];double*g=sg+h*A;if(s>0){double inv=1/s;for(int a=0;a<A;a++)g[a]=x[a]*inv;}else{double u=1.0/A;for(int a=0;a<A;a++)g[a]=u;}}
}
// one traversal that updates player u; fills vv[d] with u's counterfactual values
static void walk(unsigned mask,int k,int d,int u){
 int p=(starter+d)&1,q=1-u,nb=NB-1-k,A=nb+(k>=0),Hu=Hn[u],Hp=Hn[p];size_t o0=off(mask,k);double*v=vv[d],*sg=sig[d],T[8];
 strat(R,o0,Hp,A,sg);
 if(p==u){double*vA=va[d],*ru=rr[u][d],*ro=rr[q][d];for(int h=0;h<Hu;h++)v[h]=0;
  for(int a=0;a<A;a++){double*cv;
   if(a<nb){double*cu=rr[u][d+1];for(int h=0;h<Hu;h++)cu[h]=ru[h]*sg[h*A+a];memcpy(rr[q][d+1],ro,sizeof(double)*Hn[q]);int j=k+1+a;walk(mask|(1u<<j),j,d+1,u);cv=vv[d+1];}
   else{int f=k%6,qq=k/6+1;tails(q,ro,f,T);cv=vv[d+1];for(int h=0;h<Hu;h++)cv[h]=T[0]-tl(T,nd[q],qq-cf[u][h][f]);}
   for(int h=0;h<Hu;h++){double x=cv[h];vA[a*Hmax+h]=x;v[h]+=sg[h*A+a]*x;}}
  for(int h=0;h<Hu;h++){float*x=R+o0+(size_t)h*A,*s=S+o0+(size_t)h*A;double val=v[h],rw=ru[h]*W;for(int a=0;a<A;a++){double t=x[a]+vA[a*Hmax+h]-val;x[a]=t>0?(float)t:0;s[a]+=(float)(rw*sg[h*A+a]);}}
 }else{double*ro=rr[p][d],*ru=rr[u][d];for(int h=0;h<Hu;h++)v[h]=0;
  for(int a=0;a<A;a++){double*co=rr[p][d+1];for(int h=0;h<Hp;h++)co[h]=ro[h]*sg[h*A+a];double*cv;
   if(a<nb){memcpy(rr[u][d+1],ru,sizeof(double)*Hu);int j=k+1+a;walk(mask|(1u<<j),j,d+1,u);cv=vv[d+1];}
   else{int f=k%6,qq=k/6+1;tails(p,co,f,T);cv=vv[d+1];for(int h=0;h<Hu;h++)cv[h]=tl(T,nd[p],qq-cf[u][h][f]);}
   for(int h=0;h<Hu;h++)v[h]+=cv[h];}}
}
// best reply of `seat` against the average strategy (S) of the other player
static void br(unsigned mask,int k,int d,int seat){
 int p=(starter+d)&1,opp=1-seat,nb=NB-1-k,A=nb+(k>=0),Hb=Hn[seat],Ho=Hn[opp];size_t o0=off(mask,k);double*v=vv[d],*ro=rr[opp][d],T[8];
 if(p==seat){for(int h=0;h<Hb;h++)v[h]=-1;
  for(int a=0;a<A;a++){double*cv=vv[d+1];
   if(a<nb){memcpy(rr[opp][d+1],ro,sizeof(double)*Ho);int j=k+1+a;br(mask|(1u<<j),j,d+1,seat);}
   else{int f=k%6,qq=k/6+1;tails(opp,ro,f,T);for(int h=0;h<Hb;h++)cv[h]=T[0]-tl(T,nd[opp],qq-cf[seat][h][f]);}
   for(int h=0;h<Hb;h++)if(cv[h]>v[h])v[h]=cv[h];}
 }else{double*sg=sig[d];strat(S,o0,Ho,A,sg);for(int h=0;h<Hb;h++)v[h]=0;
  for(int a=0;a<A;a++){double*co=rr[opp][d+1],m=0;for(int h=0;h<Ho;h++){co[h]=ro[h]*sg[h*A+a];m+=co[h];}if(m<=0)continue;double*cv=vv[d+1];
   if(a<nb){int j=k+1+a;br(mask|(1u<<j),j,d+1,seat);}
   else{int f=k%6,qq=k/6+1;tails(opp,co,f,T);for(int h=0;h<Hb;h++)cv[h]=tl(T,nd[opp],qq-cf[seat][h][f]);}
   for(int h=0;h<Hb;h++)v[h]+=cv[h];}}
}
static double brval(int seat){int opp=1-seat;memcpy(rr[opp][0],prior[opp],sizeof(double)*Hn[opp]);br(0,-1,0,seat);double x=0;for(int h=0;h<Hn[seat];h++)x+=prior[seat][h]*vv[0][h];return x;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
 if(argc<8){fprintf(stderr,"usage: solver n0 n1 starter maxIters targetGap maxMinutes out.f32\n");return 1;}
 n0_=atoi(argv[1]);n1_=atoi(argv[2]);starter=atoi(argv[3]);int maxIt=atoi(argv[4]);double target=atof(argv[5]),maxMin=atof(argv[6]);
 nd[0]=n0_;nd[1]=n1_;N=n0_+n1_;NB=6*N;if(NB>MAXNB){fprintf(stderr,"too big\n");return 1;}
 gen(0,n0_,1,0);gen(1,n1_,1,0);Hmax=Hn[0]>Hn[1]?Hn[0]:Hn[1];
 double size=(double)NB*Hmax;for(int k=0;k<NB;k++){baseK[k]=size;size+=pow(2,k)*(NB-k)*Hmax;}SIZE=(size_t)size;
 R=calloc(SIZE,sizeof(float));S=calloc(SIZE,sizeof(float));if(!R||!S){fprintf(stderr,"out of memory\n");return 1;}
 fprintf(stderr,"game (%d,%d) starter %d: %d bids, hands %d/%d, %.1f M slots\n",n0_,n1_,starter,NB,Hn[0],Hn[1],SIZE/1e6);
 double t0=now(),gapv=1,b0=0,b1=0;int it,every=maxIt>=2000?250:maxIt>=200?50:10,next=every;if(NB>=24){every=25;next=25;}
 for(it=1;it<=maxIt;it++){W=it;
  for(int u=0;u<2;u++){memcpy(rr[0][0],prior[0],sizeof(double)*Hn[0]);memcpy(rr[1][0],prior[1],sizeof(double)*Hn[1]);walk(0,-1,0,u);}
  double el=(now()-t0)/60;
  if(it==next||it==maxIt||el>maxMin){b0=brval(0);b1=brval(1);gapv=b0+b1-1;fprintf(stderr,"iter %d  gap %.3e  value(p0) in [%.5f, %.5f]  %.1f min\n",it,gapv,1-b1,b0,el);next+=every;if(gapv<target||el>maxMin)break;}
 }
 FILE*f=fopen(argv[7],"wb");size_t w=0;while(w<SIZE){size_t n=fwrite(S+w,sizeof(float),SIZE-w>(1u<<26)?(1u<<26):SIZE-w,f);if(!n){fprintf(stderr,"write failed\n");return 1;}w+=n;}fclose(f);
 printf("{\"n0\":%d,\"n1\":%d,\"starter\":%d,\"iters\":%d,\"gap\":%.3e,\"lo\":%.6f,\"hi\":%.6f,\"minutes\":%.2f}\n",n0_,n1_,starter,it>maxIt?maxIt:it,gapv,1-b1,b0,(now()-t0)/60);
 return 0;
}
